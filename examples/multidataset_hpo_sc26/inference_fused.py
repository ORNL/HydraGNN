#!/usr/bin/env python3
"""Fused inference: HydraGNN (multi-branch) + BranchWeightMLP.

Loads a pretrained multi-branch HydraGNN model and a trained BranchWeightMLP,
generates random atomistic structures, evaluates all branches, and produces a
weighted-average prediction of energy and forces.  With ``--encoder_reuse`` the
GNN encoder runs once per batch and only the decoder heads are repeated,
avoiding B-1 redundant encoder passes.  Includes detailed latency and
throughput measurements.

Usage:
    python inference_fused.py \\
        --logdir <path_to_training_log_dir> \\
        --num_structures 100 \\
        --batch_size 32

The --logdir should contain config.json, a .pk HydraGNN checkpoint, and a
mlp_weights/ subdirectory with .pt MLP checkpoints.  The script auto-selects
the most recent checkpoint of each type unless overridden.

Public API for reuse (e.g. ``inference_fused_write_json.py``):
``add_fused_cli_arguments``, ``load_fused_stack``, ``generate_structures``,
``run_fused_inference`` (optional ``mlp_device`` / ``profile_stages``), ``print_fused_results``.
"""

import argparse
import copy
import glob
import json
import os
import time
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from hydragnn.train.train_validate_test import (
    move_batch_to_device,
    resolve_precision,
)

from torch.utils.checkpoint import checkpoint as torch_checkpoint

from inference_random_structures import (
    add_edges,
    build_argument_parser,
    build_random_structure,
    load_config_and_model,
)

# ---------------------------------------------------------------------------
# Helpers: MLP checkpoint discovery
# ---------------------------------------------------------------------------


def _find_mlp_checkpoint(logdir: str, mlp_checkpoint: str = None) -> str:
    """Locate a .pt MLP checkpoint, defaulting to the newest in mlp_weights/."""
    if mlp_checkpoint is not None:
        path = (
            mlp_checkpoint
            if os.path.isabs(mlp_checkpoint)
            else os.path.join(logdir, mlp_checkpoint)
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MLP checkpoint not found: {path}")
        return path
    mlp_dir = os.path.join(logdir, "mlp_weights")
    candidates = glob.glob(os.path.join(mlp_dir, "*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt MLP checkpoint found in {mlp_dir}")
    return max(candidates, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Helpers: MLP reconstruction from state dict
# ---------------------------------------------------------------------------


class BranchWeightMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], num_branches: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_branches))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _reconstruct_mlp_from_state_dict(state_dict: dict) -> BranchWeightMLP:
    """Infer BranchWeightMLP architecture from weight tensor shapes."""
    linear_keys = sorted(
        [k for k in state_dict if k.endswith(".weight") and "net." in k]
    )
    if not linear_keys:
        raise ValueError("Cannot infer MLP architecture: no net.*.weight keys found")
    input_dim = state_dict[linear_keys[0]].shape[1]
    hidden_dims = tuple(state_dict[k].shape[0] for k in linear_keys[:-1])
    num_branches = state_dict[linear_keys[-1]].shape[0]
    mlp = BranchWeightMLP(input_dim, hidden_dims, num_branches)
    mlp.load_state_dict(state_dict, strict=True)
    return mlp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_fused_cli_arguments(parser):
    """Add fused-inference-only arguments to a parser from ``build_argument_parser``."""
    parser.add_argument(
        "--mlp_checkpoint",
        default=None,
        help="MLP checkpoint path (defaults to newest .pt in <logdir>/mlp_weights/)",
    )
    parser.add_argument(
        "--mlp_precision",
        type=str,
        default=None,
        help="MLP parameter dtype (fp32, fp64, bf16). Default: same as --precision / config.",
    )
    parser.add_argument(
        "--mlp_device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for BranchWeightMLP (default cuda = same as HydraGNN).",
    )
    parser.add_argument(
        "--profile_stages",
        action="store_true",
        help="Per-batch timing for MLP, branch forwards, and combine (synced on CUDA).",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=1,
        help="Number of warmup batches excluded from timing",
    )
    parser.add_argument(
        "--encoder_reuse",
        action="store_true",
        help="Run GNN encoder once per batch, then only decoder heads B times "
        "(avoids redundant encoder passes).",
    )
    parser.add_argument(
        "--num_streams",
        type=int,
        default=1,
        help="Number of HIP/CUDA streams for parallel branch dispatch. "
        "1 = sequential (default). Set to 16 to run all branches concurrently.",
    )
    parser.add_argument(
        "--weight_threshold",
        type=float,
        default=0.0,
        help="Skip backward pass for branches whose mean weight (across the "
        "batch) is below this value. Energy is still computed (forward-only) "
        "but forces are zeroed for skipped branches. 0.0 = no skipping.",
    )
    parser.add_argument(
        "--fused_energy_grad",
        action="store_true",
        help="Compute weighted energy sum first, then ONE autograd.grad call "
        "for forces (instead of 16 separate backward passes). Mathematically "
        "equivalent when all branches are included.",
    )
    parser.add_argument(
        "--disable_param_grad",
        action="store_true",
        help="Set requires_grad=False on all model parameters before inference. "
        "Only data.pos retains gradients (needed for forces). Reduces backward "
        "computation by skipping parameter gradient accumulation.",
    )
    parser.add_argument(
        "--batched_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vectorize the 16 branch decoder heads into batched matrix "
        "multiplications instead of sequential per-branch forward calls. "
        "Only applies to the fused-reuse path with graph-level heads. "
        "(default: enabled; use --no-batched_decoder to disable)",
    )
    parser.add_argument(
        "--compile_encoder",
        action="store_true",
        default=False,
        help="Apply torch.compile to each PAINN conv block in the encoder. "
        "Requires a compatible PyTorch/ROCm/Triton stack; may cause SIGILL "
        "on environments where Inductor codegen targets unsupported instructions.",
    )
    parser.add_argument(
        "--compile_decoder",
        action="store_true",
        help="Apply torch.compile to each branch decoder head (graph_shared + heads_NN).",
    )
    parser.add_argument(
        "--compile_full",
        action="store_true",
        help="Apply torch.compile to the full model (fullgraph=False). "
        "Uses dynamo.allow_in_graph(autograd.grad) so the backward pass "
        "can also be compiled.",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor). Use 'eager' for debugging.",
    )
    parser.add_argument(
        "--omnistat_fom",
        action="store_true",
        help="POST per-batch throughput FOM to the local Omnistat collector.",
    )
    parser.add_argument(
        "--omnistat_fom_port",
        type=int,
        default=8002,
        help="Port for the Omnistat FOM HTTP endpoint (default: 8002, "
        "matching [omnistat.collectors] port in omnihub.config).",
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mlp(logdir, mlp_checkpoint, mlp_dtype, mlp_device: torch.device):
    """Load BranchWeightMLP from checkpoint.

    Returns
    -------
    mlp : BranchWeightMLP
    mlp_path : str
    mlp_ckpt : dict
    """
    mlp_path = _find_mlp_checkpoint(logdir, mlp_checkpoint)
    mlp_ckpt = torch.load(mlp_path, map_location=mlp_device)
    mlp = _reconstruct_mlp_from_state_dict(mlp_ckpt["mlp_state_dict"])
    mlp = mlp.to(dtype=mlp_dtype, device=mlp_device)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp, mlp_path, mlp_ckpt


def _mlp_bf16_autocast(mlp_device: torch.device):
    """Autocast context for bf16 MLP forward when GNN uses a different precision."""
    if mlp_device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if mlp_device.type == "cpu" and getattr(torch.backends.cpu, "has_bf16", False):
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def _mlp_forward_autocast(mlp_device: torch.device, mlp_prec_str: str):
    prec, _, _ = resolve_precision(mlp_prec_str)
    if prec == "bf16":
        return _mlp_bf16_autocast(mlp_device)
    return nullcontext()


def load_fused_stack(
    logdir,
    checkpoint=None,
    mlp_checkpoint=None,
    precision_override=None,
    mlp_precision_override=None,
    mlp_device_str: str = "cuda",
):
    """Load HydraGNN (via ``load_config_and_model``) and BranchWeightMLP.

    Returns
    -------
    model, mlp, config, device, autocast_ctx, param_dtype, num_branches,
    mlp_device, mlp_autocast_ctx, unified_mlp_gnn_stack, gnn_prec_str, mlp_prec_str
    """
    config_path = os.path.join(logdir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {logdir}")
    with open(config_path, "r") as f:
        config_pre = json.load(f)

    gnn_prec_str = precision_override or config_pre["NeuralNetwork"]["Training"].get(
        "precision", "fp32"
    )
    mlp_prec_str = (
        mlp_precision_override if mlp_precision_override is not None else gnn_prec_str
    )
    _, mlp_dtype, _ = resolve_precision(mlp_prec_str)

    model, config, device, autocast_ctx, param_dtype = load_config_and_model(
        logdir, checkpoint, precision_override
    )

    num_branches = getattr(model, "num_branches", 1)
    print(f"HydraGNN num_branches = {num_branches}")

    mlp_dev = device if mlp_device_str == "cuda" else torch.device("cpu")
    mlp, mlp_path, mlp_ckpt = load_mlp(logdir, mlp_checkpoint, mlp_dtype, mlp_dev)

    unified_mlp_gnn_stack = (mlp_dev == device) and (mlp_prec_str == gnn_prec_str)
    mlp_autocast_ctx = (
        nullcontext()
        if unified_mlp_gnn_stack
        else _mlp_forward_autocast(mlp_dev, mlp_prec_str)
    )

    linear_keys = sorted(
        k for k in mlp_ckpt["mlp_state_dict"] if k.endswith(".weight") and "net." in k
    )
    sd = mlp_ckpt["mlp_state_dict"]
    input_dim = sd[linear_keys[0]].shape[1]
    hidden_dims = [sd[k].shape[0] for k in linear_keys[:-1]]
    mlp_out = sd[linear_keys[-1]].shape[0]
    print(f"MLP checkpoint: {mlp_path}")
    print(
        f"  architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {mlp_out}"
    )
    print(f"  device: {mlp_dev}, dtype: {mlp_dtype}, prec_str: {mlp_prec_str}")
    print(f"  unified_mlp_gnn_stack (single autocast path): {unified_mlp_gnn_stack}")

    if mlp_out != num_branches:
        print(
            f"WARNING: MLP output dim ({mlp_out}) != HydraGNN num_branches "
            f"({num_branches}). Results may be incorrect."
        )

    return (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        mlp_dev,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        gnn_prec_str,
        mlp_prec_str,
    )


# ---------------------------------------------------------------------------
# Structure generation (no dataset_name; fused sets branch per forward)
# ---------------------------------------------------------------------------


def generate_structures(
    num_structures,
    min_atoms,
    max_atoms,
    box_size,
    max_atomic_number,
    radius,
    max_neighbours,
    seed,
):
    """Generate random structures with radius edges (no ``dataset_name`` set)."""
    rng = np.random.default_rng(seed)
    structures = [
        build_random_structure(min_atoms, max_atoms, box_size, max_atomic_number, rng)
        for _ in range(num_structures)
    ]
    return add_edges(structures, radius, max_neighbours)


# ---------------------------------------------------------------------------
# Helpers: per-branch prediction and weighted averaging
# ---------------------------------------------------------------------------


def _reshape_composition(data) -> torch.Tensor:
    """Return composition as [num_graphs, comp_dim]."""
    comp = data.chemical_composition
    if comp.dim() == 1:
        comp = comp.unsqueeze(0)
    if comp.dim() == 2:
        if comp.size(0) == data.num_graphs:
            return comp
        if comp.size(1) == data.num_graphs:
            return comp.t()
        if comp.size(1) == 1 and comp.size(0) % data.num_graphs == 0:
            return comp.view(data.num_graphs, -1)
    if comp.dim() == 3:
        if comp.size(0) == data.num_graphs:
            return comp.view(data.num_graphs, -1)
        if comp.size(1) == data.num_graphs:
            return comp.transpose(0, 1).contiguous().view(data.num_graphs, -1)
    raise ValueError(
        f"Unsupported chemical_composition shape {tuple(comp.shape)} "
        f"for num_graphs={data.num_graphs}"
    )


def _build_dataset_name(data, branch_id: int) -> torch.Tensor:
    if hasattr(data, "dataset_name"):
        return torch.full_like(data.dataset_name, branch_id)
    return torch.full(
        (data.num_graphs, 1),
        branch_id,
        dtype=torch.long,
        device=data.x.device,
    )


def _energy_from_pred(pred) -> torch.Tensor:
    if isinstance(pred, (list, tuple)):
        energy = pred[0]
    elif isinstance(pred, dict) and "graph" in pred:
        energy = pred["graph"][0]
    else:
        energy = pred
    return energy.squeeze(-1)


def _predict_branch_energy_forces(
    model, data, branch_id: int, retain_graph: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)

    pred = model(data)
    energy_pred = _energy_from_pred(pred)
    forces_pred = torch.autograd.grad(
        energy_pred,
        data.pos,
        grad_outputs=torch.ones_like(energy_pred),
        retain_graph=retain_graph,
        create_graph=False,
    )[0]
    forces_pred = -forces_pred

    if original_dataset_name is None and hasattr(data, "dataset_name"):
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name

    return energy_pred.detach(), forces_pred.detach()


def _predict_branch_energy_forces_decoder(
    model, data, encoded_feats, branch_id: int, retain_graph: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict energy/forces for one branch using pre-computed encoder features."""
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)

    pred = _decode_branch(model, data, encoded_feats)
    energy_pred = _energy_from_pred(pred)
    forces_pred = torch.autograd.grad(
        energy_pred,
        data.pos,
        grad_outputs=torch.ones_like(energy_pred),
        retain_graph=retain_graph,
        create_graph=False,
    )[0]
    forces_pred = -forces_pred

    if original_dataset_name is None and hasattr(data, "dataset_name"):
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name

    return energy_pred.detach(), forces_pred.detach()


def _predict_branch_energy_only(
    model, data, branch_id: int
) -> torch.Tensor:
    """Forward-only energy for the no-reuse path (gradient still attached)."""
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)
    pred = model(data)
    energy_pred = _energy_from_pred(pred)
    if original_dataset_name is None and hasattr(data, "dataset_name"):
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name
    return energy_pred


def _predict_branch_energy_only_decoder(
    model, data, encoded_feats, branch_id: int
) -> torch.Tensor:
    """Forward-only energy for the encoder-reuse path (gradient still attached)."""
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)
    pred = _decode_branch(model, data, encoded_feats)
    energy_pred = _energy_from_pred(pred)
    if original_dataset_name is None and hasattr(data, "dataset_name"):
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name
    return energy_pred


def _fused_energy_forces(
    live_energies: list,
    live_weights: list,
    skip_energies: list,
    skip_weights: list,
    batch_pos: torch.Tensor,
    batch_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted energy then ONE autograd.grad for forces.

    *live_energies* have gradients attached; *skip_energies* are detached
    (their force contribution is zero by construction).
    """
    weighted_e = torch.zeros(
        live_energies[0].shape if live_energies else skip_energies[0].shape,
        device=batch_pos.device,
        dtype=batch_pos.dtype,
    )
    for e, w in zip(live_energies, live_weights):
        weighted_e = weighted_e + w * e
    skip_e_sum = torch.zeros_like(weighted_e)
    for e, w in zip(skip_energies, skip_weights):
        skip_e_sum = skip_e_sum + w * e.detach()
    total_weighted_e = weighted_e + skip_e_sum

    if live_energies:
        forces = -torch.autograd.grad(
            weighted_e,
            batch_pos,
            grad_outputs=torch.ones_like(weighted_e),
            retain_graph=False,
            create_graph=False,
        )[0]
    else:
        forces = torch.zeros_like(batch_pos)

    return total_weighted_e.detach(), forces.detach()


def _weighted_average(
    energy_preds: torch.Tensor,
    forces_preds: torch.Tensor,
    weights: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weighted_energy = torch.sum(weights * energy_preds.transpose(0, 1), dim=1)

    node_counts = torch.bincount(batch)
    weighted_forces = torch.zeros_like(forces_preds[0])
    for branch_idx in range(energy_preds.size(0)):
        node_weights = torch.repeat_interleave(weights[:, branch_idx], node_counts)
        weighted_forces = (
            weighted_forces + node_weights.unsqueeze(-1) * forces_preds[branch_idx]
        )

    return weighted_energy, weighted_forces


# ---------------------------------------------------------------------------
# Stream-parallel branch dispatch
# ---------------------------------------------------------------------------


def _run_branches_parallel_streams(
    model,
    batch,
    num_branches: int,
    num_streams: int,
    device: torch.device,
) -> Tuple[list, list]:
    """Dispatch full forward+backward for each branch across multiple streams.

    Used when ``encoder_reuse`` is **False**: each branch runs the complete
    ``model(data)`` call on its assigned stream.
    """
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    default_stream = torch.cuda.current_stream(device)

    energy_preds: list = [None] * num_branches
    forces_preds: list = [None] * num_branches

    for branch_id in range(num_branches):
        s = streams[branch_id % num_streams]
        with torch.cuda.stream(s):
            s.wait_stream(default_stream)
            batch_copy = copy.copy(batch)
            batch_copy.dataset_name = _build_dataset_name(batch, branch_id)
            pred = model(batch_copy)
            energy_pred = _energy_from_pred(pred)
            forces_pred = -torch.autograd.grad(
                energy_pred,
                batch.pos,
                grad_outputs=torch.ones_like(energy_pred),
                retain_graph=True,
                create_graph=False,
            )[0]
            energy_preds[branch_id] = energy_pred.detach()
            forces_preds[branch_id] = forces_pred.detach()

    for s in streams:
        default_stream.wait_stream(s)

    return energy_preds, forces_preds


def _run_branches_parallel_streams_decoder(
    model,
    batch,
    encoded_feats,
    num_branches: int,
    num_streams: int,
    device: torch.device,
) -> Tuple[list, list]:
    """Dispatch decoder-only forward+backward for each branch across streams.

    Used when ``encoder_reuse`` is **True**: the encoder has already run on the
    default stream and *encoded_feats* are shared (read-only) across branches.
    """
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    default_stream = torch.cuda.current_stream(device)

    energy_preds: list = [None] * num_branches
    forces_preds: list = [None] * num_branches

    for branch_id in range(num_branches):
        s = streams[branch_id % num_streams]
        with torch.cuda.stream(s):
            s.wait_stream(default_stream)
            batch_copy = copy.copy(batch)
            batch_copy.dataset_name = _build_dataset_name(batch, branch_id)
            pred = _decode_branch(model, batch_copy, encoded_feats)
            energy_pred = _energy_from_pred(pred)
            forces_pred = -torch.autograd.grad(
                energy_pred,
                batch.pos,
                grad_outputs=torch.ones_like(energy_pred),
                retain_graph=True,
                create_graph=False,
            )[0]
            energy_preds[branch_id] = energy_pred.detach()
            forces_preds[branch_id] = forces_pred.detach()

    for s in streams:
        default_stream.wait_stream(s)

    return energy_preds, forces_preds


# ---------------------------------------------------------------------------
# Encoder-reuse: local encoder / decoder split (matches Base.forward exactly)
# ---------------------------------------------------------------------------


def _encode_once(model, data):
    """Run the encoder portion of ``Base.forward`` and return cached features.

    Reproduces ``Base.forward`` lines 688-719 including the
    ``_apply_graph_conditioning`` call that ``EncoderModel`` (non-MACE path)
    omits.  The returned tensors are **not** detached so that
    ``torch.autograd.grad`` can trace back through the encoder to ``data.pos``.
    """
    inv_node_feat, equiv_node_feat, conv_args = model._embedding(data)
    batch_for_cond = (
        data.batch if hasattr(data, "batch") and data.batch is not None else None
    )
    for conv, feat_layer in zip(model.graph_convs, model.feature_layers):
        if not model.conv_checkpointing:
            inv_node_feat, equiv_node_feat = conv(
                inv_node_feat=inv_node_feat,
                equiv_node_feat=equiv_node_feat,
                **conv_args,
            )
        else:
            inv_node_feat, equiv_node_feat = torch_checkpoint(
                conv,
                use_reentrant=False,
                inv_node_feat=inv_node_feat,
                equiv_node_feat=equiv_node_feat,
                **conv_args,
            )
        inv_node_feat = model._apply_graph_conditioning(
            inv_node_feat, batch_for_cond, data
        )
        inv_node_feat = model.activation_function(feat_layer(inv_node_feat))
    return inv_node_feat, equiv_node_feat, conv_args


def _decode_branch(model, data, encoded_feats):
    """Run the decoder portion of ``Base.forward`` with pre-computed features.

    Reproduces ``Base.forward`` lines 721-837 including the
    ``_apply_graph_pool_conditioning`` call that ``DecoderModel`` omits.
    """
    inv_node_feat, equiv_node_feat, conv_args = encoded_feats
    x = inv_node_feat

    if data.batch is None:
        x_graph = model._pool_graph_features(x, None)
        data.batch = data.x * 0
    else:
        x_graph = model._pool_graph_features(x, data.batch)

    x_graph = model._apply_graph_pool_conditioning(x_graph, data)

    outputs = []
    outputs_var = []
    if not hasattr(data, "dataset_name"):
        setattr(data, "dataset_name", data.batch.unique() * 0)
    datasetIDs = data.dataset_name.unique()
    unique, node_counts = torch.unique_consecutive(data.batch, return_counts=True)

    for head_dim, headloc, type_head in zip(
        model.head_dims, model.heads_NN, model.head_type
    ):
        if type_head == "graph":
            out_dtype = x_graph.dtype
            head = torch.zeros(
                (len(data.dataset_name), head_dim),
                device=x.device,
                dtype=out_dtype,
            )
            headvar = torch.zeros(
                (len(data.dataset_name), head_dim * model.var_output),
                device=x.device,
                dtype=out_dtype,
            )
            if model.num_branches == 1:
                x_graph_head = model.graph_shared["branch-0"](x_graph)
                output_head = headloc["branch-0"](x_graph_head)
                head = output_head[:, :head_dim]
                headvar = output_head[:, head_dim:] ** 2
            else:
                for ID in datasetIDs:
                    mask = data.dataset_name == ID
                    mask = mask[:, 0]
                    branchtype = f"branch-{ID.item()}"
                    x_graph_head = model.graph_shared[branchtype](x_graph[mask, :])
                    output_head = headloc[branchtype](x_graph_head)
                    head[mask] = output_head[:, :head_dim]
                    headvar[mask] = (output_head[:, head_dim:] ** 2).to(
                        dtype=out_dtype
                    )
            outputs.append(head)
            outputs_var.append(headvar)
        else:
            node_NN_type = model.config_heads["node"][0]["architecture"]["type"]
            out_dtype = x.dtype
            head = torch.zeros(
                (x.shape[0], head_dim), device=x.device, dtype=out_dtype
            )
            headvar = torch.zeros(
                (x.shape[0], head_dim * model.var_output),
                device=x.device,
                dtype=out_dtype,
            )
            if model.num_branches == 1:
                branchtype = "branch-0"
                if node_NN_type == "conv":
                    inv_nf = x
                    equiv_nf = equiv_node_feat
                    for cnv, bn in zip(
                        headloc[branchtype][0::2], headloc[branchtype][1::2]
                    ):
                        inv_nf, equiv_nf = cnv(
                            inv_node_feat=inv_nf,
                            equiv_node_feat=equiv_nf,
                            **conv_args,
                        )
                        inv_nf = bn(inv_nf)
                        inv_nf = model.activation_function(inv_nf)
                    x_node = inv_nf
                else:
                    x_node = headloc[branchtype](x=x, batch=data.batch)
                head = x_node[:, :head_dim]
                headvar = x_node[:, head_dim:] ** 2
            else:
                for ID in datasetIDs:
                    mask = data.dataset_name == ID
                    mask_nodes = torch.repeat_interleave(mask, node_counts)
                    branchtype = f"branch-{ID.item()}"
                    if node_NN_type == "conv":
                        inv_nf = x[mask_nodes, :]
                        equiv_nf = equiv_node_feat[mask_nodes, :]
                        for cnv, bn in zip(
                            headloc[branchtype][0::2], headloc[branchtype][1::2]
                        ):
                            inv_nf, equiv_nf = cnv(
                                inv_node_feat=inv_nf,
                                equiv_node_feat=equiv_nf,
                                **conv_args,
                            )
                            inv_nf = bn(inv_nf)
                            inv_nf = model.activation_function(inv_nf)
                        x_node = inv_nf
                    else:
                        x_node = headloc[branchtype](
                            x=x[mask_nodes, :], batch=data.batch[mask_nodes]
                        )
                    head[mask_nodes] = x_node[:, :head_dim]
                    headvar[mask_nodes] = x_node[:, head_dim:] ** 2
            outputs.append(head)
            outputs_var.append(headvar)

    if model.var_output:
        return outputs, outputs_var
    return outputs


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _make_timer(device):
    """Return (start, stop_and_elapsed_ms) callables that use CUDA events on GPU."""
    use_cuda = device.type == "cuda"

    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def start():
            start_event.record()

        def stop():
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event)

        return start, stop

    _t = {}

    def start():
        _t["t0"] = time.perf_counter()

    def stop():
        return (time.perf_counter() - _t["t0"]) * 1000.0

    return start, stop


def _post_omnistat_fom(fom_url: str, gpu_id: int, throughput: float):
    """POST per-batch throughput FOM to the local Omnistat collector."""
    import requests

    payload = {
        "name": f"throughput_structs_per_sec_gpu{gpu_id}",
        "value": throughput,
    }
    try:
        res = requests.post(fom_url, json=payload, timeout=2)
        if not res.ok:
            print(f"  [omnistat-fom] POST failed: {res.status_code}")
    except Exception as exc:
        print(f"  [omnistat-fom] POST error: {exc}")


# ---------------------------------------------------------------------------
# Batched decoder: vectorize 16 identical-architecture branch heads
# ---------------------------------------------------------------------------


def _build_batched_decoder(model, device):
    """Stack weights from all B branch decoder heads into batched tensors.

    All branches must share the same architecture (same layer sizes).
    Returns a dict with stacked weights/biases for graph_shared and heads_NN,
    plus activation function reference.

    The SC26 config has: graph_shared = Linear(337,50)+ReLU+Linear(50,50)+ReLU
                         heads_NN     = Linear(50,776)+ReLU+Linear(776,776)+ReLU+Linear(776,1)
    """
    num_branches = model.num_branches
    branch_keys = [f"branch-{i}" for i in range(num_branches)]

    def _extract_linear_params(sequential):
        """Extract (weight, bias) pairs from a Sequential of Linear+Act layers."""
        params = []
        for m in sequential:
            if isinstance(m, nn.Linear):
                params.append((m.weight.data, m.bias.data))
        return params

    shared_params = [_extract_linear_params(model.graph_shared[k]) for k in branch_keys]
    head_NN_dict = model.heads_NN[0]
    head_params = [_extract_linear_params(head_NN_dict[k]) for k in branch_keys]

    num_shared_layers = len(shared_params[0])
    num_head_layers = len(head_params[0])

    stacked_shared_W = []
    stacked_shared_b = []
    for li in range(num_shared_layers):
        W = torch.stack([shared_params[b][li][0] for b in range(num_branches)])
        b = torch.stack([shared_params[b][li][1] for b in range(num_branches)])
        stacked_shared_W.append(W)
        stacked_shared_b.append(b)

    stacked_head_W = []
    stacked_head_b = []
    for li in range(num_head_layers):
        W = torch.stack([head_params[b][li][0] for b in range(num_branches)])
        b = torch.stack([head_params[b][li][1] for b in range(num_branches)])
        stacked_head_W.append(W)
        stacked_head_b.append(b)

    return {
        "num_branches": num_branches,
        "shared_W": stacked_shared_W,
        "shared_b": stacked_shared_b,
        "head_W": stacked_head_W,
        "head_b": stacked_head_b,
        "act_fn": model.activation_function,
    }


def _batched_decode_all_branches(model, data, encoded_feats, batched_dec):
    """Run all B branch decoders in a single batched pass.

    Instead of B sequential forward calls through separate nn.Module instances,
    this broadcasts the pooled graph features across all B branches and uses
    batched matrix multiplications.

    Returns a list of B energy tensors (with gradients attached for autograd.grad).
    """
    inv_node_feat, equiv_node_feat, conv_args = encoded_feats

    if data.batch is None:
        x_graph = model._pool_graph_features(inv_node_feat, None)
        data.batch = data.x * 0
    else:
        x_graph = model._pool_graph_features(inv_node_feat, data.batch)

    x_graph = model._apply_graph_pool_conditioning(x_graph, data)

    B = batched_dec["num_branches"]
    act_fn = batched_dec["act_fn"]

    # x_graph: [G, H] -> [B, G, H] for batched matmul
    x = x_graph.unsqueeze(0).expand(B, -1, -1)

    for W, b in zip(batched_dec["shared_W"], batched_dec["shared_b"]):
        # W: [B, out, in], x: [B, G, in] -> bmm needs x @ W^T
        x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)
        x = act_fn(x)

    for i, (W, b) in enumerate(zip(batched_dec["head_W"], batched_dec["head_b"])):
        x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)
        if i < len(batched_dec["head_W"]) - 1:
            x = act_fn(x)

    # x: [B, G, output_dim] -- for energy, output_dim=1
    energies = [x[branch_id, :, 0] for branch_id in range(B)]
    return energies


# ---------------------------------------------------------------------------
# torch.compile helpers
# ---------------------------------------------------------------------------


def _apply_torch_compile(model, compile_encoder, compile_decoder, compile_full, backend):
    """Apply torch.compile to model submodules or the full model.

    Returns the (possibly compiled) model.
    """
    import time as _time

    _base = model.module if hasattr(model, "module") else model

    if compile_full:
        try:
            import torch._dynamo as dynamo
            dynamo.allow_in_graph(torch.autograd.grad)
        except Exception:
            pass
        print(f"torch.compile: compiling full model (backend={backend}, fullgraph=False)")
        t0 = _time.perf_counter()
        model = torch.compile(model, dynamic=True, fullgraph=False, backend=backend)
        print(f"  compile setup: {(_time.perf_counter() - t0)*1000:.0f} ms")
        return model

    if compile_encoder:
        n_compiled = 0
        t0 = _time.perf_counter()
        for i, conv in enumerate(_base.graph_convs):
            _base.graph_convs[i] = torch.compile(
                conv, dynamic=True, backend=backend
            )
            n_compiled += 1
        print(
            f"torch.compile: {n_compiled} encoder conv blocks compiled "
            f"(backend={backend}, {(_time.perf_counter() - t0)*1000:.0f} ms)"
        )

    if compile_decoder:
        n_compiled = 0
        t0 = _time.perf_counter()
        for branch_key in _base.graph_shared:
            _base.graph_shared[branch_key] = torch.compile(
                _base.graph_shared[branch_key], dynamic=True, backend=backend
            )
            n_compiled += 1
        for head_dict in _base.heads_NN:
            for branch_key in head_dict:
                head_dict[branch_key] = torch.compile(
                    head_dict[branch_key], dynamic=True, backend=backend
                )
                n_compiled += 1
        print(
            f"torch.compile: {n_compiled} decoder head modules compiled "
            f"(backend={backend}, {(_time.perf_counter() - t0)*1000:.0f} ms)"
        )

    return model


# ---------------------------------------------------------------------------
# Inference and reporting
# ---------------------------------------------------------------------------


def _sync_device(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run_fused_inference(
    model,
    mlp,
    structures,
    batch_size,
    param_dtype,
    autocast_ctx,
    device,
    num_branches,
    num_warmup,
    mlp_device: torch.device,
    mlp_autocast_ctx,
    unified_mlp_gnn_stack: bool = True,
    profile_stages: bool = False,
    encoder_reuse: bool = False,
    num_streams: int = 1,
    weight_threshold: float = 0.0,
    fused_energy_grad: bool = False,
    per_batch_callback=None,
    omnistat_fom_url: str = None,
    omnistat_fom_gpu_id: int = 0,
    disable_param_grad: bool = False,
    batched_decoder: bool = False,
    compile_encoder: bool = False,
    compile_decoder: bool = False,
    compile_full: bool = False,
    compile_backend: str = "inductor",
):
    """Run batched fused inference with timing (excludes warmup batches).

    When *encoder_reuse* is True, the GNN encoder runs once per batch and only
    the branch-specific decoder heads are evaluated B times, avoiding B-1
    redundant encoder passes.

    When *num_streams* > 1, branch forward+backward passes are dispatched
    across multiple HIP/CUDA streams for GPU-level overlap.

    When *weight_threshold* > 0, branches whose batch-mean softmax weight is
    below the threshold are skipped for backward (forces zeroed, energy still
    computed forward-only).

    When *fused_energy_grad* is True, all branch energies are weighted-summed
    first, then a single ``autograd.grad`` computes forces, replacing 16
    separate backward passes.

    When *per_batch_callback* is not None it is called after each batch's
    results are extracted to CPU (but before the next batch starts on the
    GPU).  Signature::

        per_batch_callback(batch_idx, batch_energies, batch_forces,
                           batch_natoms, batch_weights)

    This enables the caller to pipeline I/O (e.g. NVMe writes) with GPU
    inference.

    Returns
    -------
    all_energies : list[float]
    all_forces : list[torch.Tensor]
    all_natoms : list[int]
    all_weights : list[torch.Tensor]
    batch_latencies_ms : list[float]
        Latencies for batches after warmup only.
    total_timed_structures : int
        Number of structures in timed batches (excludes warmup).
    stage_stats : dict | None
        If profile_stages, keys mlp_ms, branches_ms, combine_ms (lists per
        timed batch).  With encoder_reuse, also includes encoder_ms.
    """
    batches = []
    for start in range(0, len(structures), batch_size):
        batches.append(structures[start : start + batch_size])

    num_warmup_effective = min(num_warmup, len(batches))
    timer_start, timer_stop_ms = _make_timer(device)

    all_energies = []
    all_forces = []
    all_natoms = []
    all_weights = []
    batch_latencies_ms = []
    stage_mlp_ms: List[float] = []
    stage_encoder_ms: List[float] = []
    stage_branches_ms: List[float] = []
    stage_combine_ms: List[float] = []

    mlp_param_dtype = next(mlp.parameters()).dtype

    use_parallel_streams = num_streams > 1 and device.type == "cuda"
    if use_parallel_streams:
        print(
            f"Stream parallelism: {num_streams} HIP/CUDA streams for "
            f"{num_branches} branches"
        )

    _base_model = None
    if encoder_reuse:
        _base_model = model.module if hasattr(model, "module") else model
        print(
            f"Encoder-reuse enabled: 1 encoder pass + {num_branches} decoder "
            f"passes per batch (trades memory for compute — encoder graph is "
            f"pinned across all decoder calls)"
        )

    if disable_param_grad:
        n_disabled = 0
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n_disabled += 1
        print(
            f"Parameter grad disabled: {n_disabled} model parameters set to "
            f"requires_grad=False (only data.pos retains grad for forces)"
        )

    if weight_threshold > 0:
        print(
            f"Branch skipping: branches with mean weight < {weight_threshold:.4f} "
            f"get forward-only energy (zero forces)"
        )
    if fused_energy_grad:
        print(
            "Fused energy gradient: single autograd.grad from weighted energy "
            "sum (replaces per-branch backward passes)"
        )

    batched_dec = None
    if batched_decoder and encoder_reuse and fused_energy_grad:
        batched_dec = _build_batched_decoder(_base_model, device)
        print(
            f"Batched decoder: {batched_dec['num_branches']} branches vectorized "
            f"into batched matrix multiplications"
        )
    elif batched_decoder:
        print(
            "WARNING: --batched_decoder requires --encoder_reuse and "
            "--fused_energy_grad; falling back to sequential decoder"
        )

    if compile_encoder or compile_decoder or compile_full:
        if compile_decoder and batched_dec is not None:
            print(
                "NOTE: --compile_decoder skipped (batched decoder uses "
                "pre-stacked weights, not nn.Module forward calls)"
            )
            compile_decoder = False
        model = _apply_torch_compile(
            model, compile_encoder, compile_decoder, compile_full, compile_backend
        )
        if encoder_reuse:
            _base_model = model.module if hasattr(model, "module") else model

    num_skipped_total = 0
    num_evaluated_total = 0

    for batch_idx, batch_list in enumerate(batches):
        is_warmup = batch_idx < num_warmup_effective
        tag = "warmup" if is_warmup else "timed"
        print(
            f"  batch {batch_idx + 1}/{len(batches)} ({tag}, "
            f"{len(batch_list)} structs)",
            flush=True,
        )

        batch = Batch.from_data_list(batch_list)
        batch = move_batch_to_device(batch, param_dtype)
        batch.pos.requires_grad_(True)

        timer_start()

        with torch.enable_grad():
            if unified_mlp_gnn_stack:
                with autocast_ctx:
                    comp = _reshape_composition(batch).to(
                        device=device, dtype=param_dtype
                    )
                    if profile_stages:
                        _sync_device(device)
                        t_mlp0 = time.perf_counter()
                    logits = mlp(comp)
                    weights = F.softmax(logits, dim=-1)
                    if profile_stages:
                        _sync_device(device)
                        t_mlp1 = time.perf_counter()
                    mean_weights = weights.mean(dim=0)

                    energy_preds = []
                    forces_preds = []
                    live_energies: list = []
                    live_weights_list: list = []
                    skip_energies: list = []
                    skip_weights_list: list = []

                    if encoder_reuse:
                        if profile_stages:
                            _sync_device(device)
                            t_enc0 = time.perf_counter()
                        encoded_feats = _encode_once(_base_model, batch)
                        if profile_stages:
                            _sync_device(device)
                            t_enc1 = time.perf_counter()
                        if batched_dec is not None and fused_energy_grad:
                            all_energies_b = _batched_decode_all_branches(
                                _base_model, batch, encoded_feats, batched_dec
                            )
                            for branch_id in range(num_branches):
                                is_skipped = (
                                    weight_threshold > 0
                                    and mean_weights[branch_id].item() < weight_threshold
                                )
                                e = all_energies_b[branch_id]
                                if is_skipped:
                                    skip_energies.append(e)
                                    skip_weights_list.append(weights[:, branch_id])
                                else:
                                    live_energies.append(e)
                                    live_weights_list.append(weights[:, branch_id])
                        elif use_parallel_streams and not fused_energy_grad:
                            energy_preds, forces_preds = (
                                _run_branches_parallel_streams_decoder(
                                    _base_model,
                                    batch,
                                    encoded_feats,
                                    num_branches,
                                    num_streams,
                                    device,
                                )
                            )
                        else:
                            for branch_id in range(num_branches):
                                is_skipped = (
                                    weight_threshold > 0
                                    and mean_weights[branch_id].item() < weight_threshold
                                )
                                if fused_energy_grad:
                                    e = _predict_branch_energy_only_decoder(
                                        _base_model, batch, encoded_feats, branch_id
                                    )
                                    if is_skipped:
                                        skip_energies.append(e)
                                        skip_weights_list.append(weights[:, branch_id])
                                    else:
                                        live_energies.append(e)
                                        live_weights_list.append(weights[:, branch_id])
                                elif is_skipped:
                                    e = _predict_branch_energy_only_decoder(
                                        _base_model, batch, encoded_feats, branch_id
                                    )
                                    energy_preds.append(e.detach())
                                    del e
                                    forces_preds.append(torch.zeros_like(batch.pos))
                                else:
                                    is_last = branch_id == num_branches - 1
                                    e, f = _predict_branch_energy_forces_decoder(
                                        _base_model, batch, encoded_feats, branch_id,
                                        retain_graph=not is_last,
                                    )
                                    energy_preds.append(e)
                                    forces_preds.append(f)
                    else:
                        if use_parallel_streams and not fused_energy_grad:
                            energy_preds, forces_preds = (
                                _run_branches_parallel_streams(
                                    model,
                                    batch,
                                    num_branches,
                                    num_streams,
                                    device,
                                )
                            )
                        else:
                            for branch_id in range(num_branches):
                                is_skipped = (
                                    weight_threshold > 0
                                    and mean_weights[branch_id].item() < weight_threshold
                                )
                                if fused_energy_grad:
                                    e = _predict_branch_energy_only(
                                        model, batch, branch_id
                                    )
                                    if is_skipped:
                                        skip_energies.append(e)
                                        skip_weights_list.append(weights[:, branch_id])
                                    else:
                                        live_energies.append(e)
                                        live_weights_list.append(weights[:, branch_id])
                                elif is_skipped:
                                    e = _predict_branch_energy_only(
                                        model, batch, branch_id
                                    )
                                    energy_preds.append(e.detach())
                                    del e
                                    forces_preds.append(torch.zeros_like(batch.pos))
                                else:
                                    is_last = branch_id == num_branches - 1
                                    e, f = _predict_branch_energy_forces(
                                        model, batch, branch_id,
                                        retain_graph=not is_last,
                                    )
                                    energy_preds.append(e)
                                    forces_preds.append(f)

                    if profile_stages:
                        _sync_device(device)
                        t_br0 = time.perf_counter()

                    if fused_energy_grad:
                        weighted_energy, weighted_forces = _fused_energy_forces(
                            live_energies,
                            live_weights_list,
                            skip_energies,
                            skip_weights_list,
                            batch.pos,
                            batch.batch,
                        )
                        num_skipped_total += len(skip_energies)
                        num_evaluated_total += len(live_energies)
                    else:
                        energy_preds_t = torch.stack(energy_preds, dim=0)
                        forces_preds_t = torch.stack(forces_preds, dim=0)
                        weighted_energy, weighted_forces = _weighted_average(
                            energy_preds_t, forces_preds_t, weights, batch.batch
                        )
                        n_skip = sum(
                            1
                            for b in range(num_branches)
                            if weight_threshold > 0
                            and mean_weights[b].item() < weight_threshold
                        )
                        num_skipped_total += n_skip
                        num_evaluated_total += num_branches - n_skip

                    if profile_stages:
                        _sync_device(device)
                        t_cb0 = time.perf_counter()
                if profile_stages and not is_warmup:
                    stage_mlp_ms.append((t_mlp1 - t_mlp0) * 1000.0)
                    if encoder_reuse:
                        stage_encoder_ms.append((t_enc1 - t_enc0) * 1000.0)
                        stage_branches_ms.append((t_br0 - t_enc1) * 1000.0)
                    else:
                        stage_branches_ms.append((t_br0 - t_mlp1) * 1000.0)
                    stage_combine_ms.append((t_cb0 - t_br0) * 1000.0)
            else:
                comp = _reshape_composition(batch).to(device=device, dtype=param_dtype)
                comp_m = comp.to(device=mlp_device, dtype=mlp_param_dtype)
                if profile_stages:
                    _sync_device(mlp_device)
                    _sync_device(device)
                    t_mlp0 = time.perf_counter()
                with mlp_autocast_ctx:
                    logits = mlp(comp_m)
                weights = F.softmax(logits, dim=-1).to(device=device, dtype=param_dtype)
                if profile_stages:
                    _sync_device(device)
                    t_mlp1 = time.perf_counter()
                with autocast_ctx:
                    mean_weights = weights.mean(dim=0)

                    energy_preds = []
                    forces_preds = []
                    live_energies = []
                    live_weights_list = []
                    skip_energies = []
                    skip_weights_list = []

                    if encoder_reuse:
                        if profile_stages:
                            _sync_device(device)
                            t_enc0 = time.perf_counter()
                        encoded_feats = _encode_once(_base_model, batch)
                        if profile_stages:
                            _sync_device(device)
                            t_enc1 = time.perf_counter()
                        if batched_dec is not None and fused_energy_grad:
                            all_energies_b = _batched_decode_all_branches(
                                _base_model, batch, encoded_feats, batched_dec
                            )
                            for branch_id in range(num_branches):
                                is_skipped = (
                                    weight_threshold > 0
                                    and mean_weights[branch_id].item() < weight_threshold
                                )
                                e = all_energies_b[branch_id]
                                if is_skipped:
                                    skip_energies.append(e)
                                    skip_weights_list.append(weights[:, branch_id])
                                else:
                                    live_energies.append(e)
                                    live_weights_list.append(weights[:, branch_id])
                        elif use_parallel_streams and not fused_energy_grad:
                            energy_preds, forces_preds = (
                                _run_branches_parallel_streams_decoder(
                                    _base_model,
                                    batch,
                                    encoded_feats,
                                    num_branches,
                                    num_streams,
                                    device,
                                )
                            )
                        else:
                            for branch_id in range(num_branches):
                                is_skipped = (
                                    weight_threshold > 0
                                    and mean_weights[branch_id].item() < weight_threshold
                                )
                                if fused_energy_grad:
                                    e = _predict_branch_energy_only_decoder(
                                        _base_model, batch, encoded_feats, branch_id
                                    )
                                    if is_skipped:
                                        skip_energies.append(e)
                                        skip_weights_list.append(weights[:, branch_id])
                                    else:
                                        live_energies.append(e)
                                        live_weights_list.append(weights[:, branch_id])
                                elif is_skipped:
                                    e = _predict_branch_energy_only_decoder(
                                        _base_model, batch, encoded_feats, branch_id
                                    )
                                    energy_preds.append(e.detach())
                                    del e
                                    forces_preds.append(torch.zeros_like(batch.pos))
                                else:
                                    is_last = branch_id == num_branches - 1
                                    e, f = _predict_branch_energy_forces_decoder(
                                        _base_model, batch, encoded_feats, branch_id,
                                        retain_graph=not is_last,
                                    )
                                    energy_preds.append(e)
                                    forces_preds.append(f)
                    else:
                        if use_parallel_streams and not fused_energy_grad:
                            energy_preds, forces_preds = (
                                _run_branches_parallel_streams(
                                    model,
                                    batch,
                                    num_branches,
                                    num_streams,
                                    device,
                                )
                            )
                        else:
                            for branch_id in range(num_branches):
                                is_skipped = (
                                    weight_threshold > 0
                                    and mean_weights[branch_id].item() < weight_threshold
                                )
                                if fused_energy_grad:
                                    e = _predict_branch_energy_only(
                                        model, batch, branch_id
                                    )
                                    if is_skipped:
                                        skip_energies.append(e)
                                        skip_weights_list.append(weights[:, branch_id])
                                    else:
                                        live_energies.append(e)
                                        live_weights_list.append(weights[:, branch_id])
                                elif is_skipped:
                                    e = _predict_branch_energy_only(
                                        model, batch, branch_id
                                    )
                                    energy_preds.append(e.detach())
                                    del e
                                    forces_preds.append(torch.zeros_like(batch.pos))
                                else:
                                    is_last = branch_id == num_branches - 1
                                    e, f = _predict_branch_energy_forces(
                                        model, batch, branch_id,
                                        retain_graph=not is_last,
                                    )
                                    energy_preds.append(e)
                                    forces_preds.append(f)

                    if profile_stages:
                        _sync_device(device)
                        t_br0 = time.perf_counter()

                    if fused_energy_grad:
                        weighted_energy, weighted_forces = _fused_energy_forces(
                            live_energies,
                            live_weights_list,
                            skip_energies,
                            skip_weights_list,
                            batch.pos,
                            batch.batch,
                        )
                        num_skipped_total += len(skip_energies)
                        num_evaluated_total += len(live_energies)
                    else:
                        energy_preds_t = torch.stack(energy_preds, dim=0)
                        forces_preds_t = torch.stack(forces_preds, dim=0)
                        weighted_energy, weighted_forces = _weighted_average(
                            energy_preds_t, forces_preds_t, weights, batch.batch
                        )
                        n_skip = sum(
                            1
                            for b in range(num_branches)
                            if weight_threshold > 0
                            and mean_weights[b].item() < weight_threshold
                        )
                        num_skipped_total += n_skip
                        num_evaluated_total += num_branches - n_skip

                    if profile_stages:
                        _sync_device(device)
                        t_cb0 = time.perf_counter()
                if profile_stages and not is_warmup:
                    stage_mlp_ms.append((t_mlp1 - t_mlp0) * 1000.0)
                    if encoder_reuse:
                        stage_encoder_ms.append((t_enc1 - t_enc0) * 1000.0)
                        stage_branches_ms.append((t_br0 - t_enc1) * 1000.0)
                    else:
                        stage_branches_ms.append((t_br0 - t_mlp1) * 1000.0)
                    stage_combine_ms.append((t_cb0 - t_br0) * 1000.0)

        elapsed_ms = timer_stop_ms()

        if omnistat_fom_url is not None:
            batch_throughput = len(batch_list) / (elapsed_ms / 1000.0)
            _post_omnistat_fom(omnistat_fom_url, omnistat_fom_gpu_id, batch_throughput)

        if not is_warmup:
            batch_latencies_ms.append(elapsed_ms)

        weighted_energy = weighted_energy.detach()
        weighted_forces = weighted_forces.detach()
        weights_cpu = weights.detach().cpu()

        batch_e: list = []
        batch_f: list = []
        batch_n: list = []
        batch_w: list = []
        num_graphs = batch.num_graphs
        for i in range(num_graphs):
            batch_e.append(
                weighted_energy[i].item()
                if weighted_energy.numel() > 1
                else weighted_energy.item()
            )
            mask = batch.batch == i
            n = int(mask.sum().item())
            batch_n.append(n)
            batch_f.append(weighted_forces[mask].cpu())
            batch_w.append(weights_cpu[i])

        all_energies.extend(batch_e)
        all_forces.extend(batch_f)
        all_natoms.extend(batch_n)
        all_weights.extend(batch_w)

        if per_batch_callback is not None:
            per_batch_callback(batch_idx, batch_e, batch_f, batch_n, batch_w)

    total_timed_structures = sum(
        len(batches[i]) for i in range(num_warmup_effective, len(batches))
    )

    if weight_threshold > 0 or fused_energy_grad:
        total_branches = num_skipped_total + num_evaluated_total
        print(
            f"Optimization summary: {num_evaluated_total}/{total_branches} "
            f"branch evaluations had full backward, "
            f"{num_skipped_total}/{total_branches} skipped "
            f"(threshold={weight_threshold}, fused_grad={fused_energy_grad})"
        )

    stage_stats = None
    if profile_stages and stage_mlp_ms:
        stage_stats = {
            "mlp_ms": stage_mlp_ms,
            "branches_ms": stage_branches_ms,
            "combine_ms": stage_combine_ms,
        }
        if encoder_reuse and stage_encoder_ms:
            stage_stats["encoder_ms"] = stage_encoder_ms

    return (
        all_energies,
        all_forces,
        all_natoms,
        all_weights,
        batch_latencies_ms,
        total_timed_structures,
        stage_stats,
    )


def print_fused_results(
    all_energies,
    all_forces,
    all_natoms,
    all_weights,
    num_branches,
    batch_latencies_ms,
    total_timed_structures,
    stage_stats=None,
):
    """Print per-structure table, branch-weight summary, and timing."""
    print("\n" + "=" * 80)
    print("PER-STRUCTURE PREDICTIONS")
    print("=" * 80)
    header = (
        f"{'Idx':>5} | {'Atoms':>5} | {'Energy':>14} | {'E/atom':>14} | "
        f"{'|F|_mean':>10} | {'Top Branch':>10} | {'Top Wt':>8}"
    )
    print(header)
    print("-" * len(header))

    num_results = len(all_energies)
    show_limit = 10
    if num_results > 20:
        for i in range(num_results):
            if i == show_limit:
                print(" " * 5 + "..." + " " * (len(header) - 8) + "...")
            if i < show_limit or i >= num_results - show_limit:
                e = all_energies[i]
                n = all_natoms[i]
                e_per_atom = e / n
                f_norms = all_forces[i].norm(dim=1)
                f_mean = f_norms.mean().item()
                w = all_weights[i]
                top_branch = int(w.argmax().item())
                top_wt = w[top_branch].item()
                print(
                    f"{i:5d} | {n:5d} | {e:14.6f} | {e_per_atom:14.6f} | "
                    f"{f_mean:10.6f} | {top_branch:10d} | {top_wt:8.4f}"
                )
    else:
        for i in range(num_results):
            e = all_energies[i]
            n = all_natoms[i]
            e_per_atom = e / n
            f_norms = all_forces[i].norm(dim=1)
            f_mean = f_norms.mean().item()
            w = all_weights[i]
            top_branch = int(w.argmax().item())
            top_wt = w[top_branch].item()
            print(
                f"{i:5d} | {n:5d} | {e:14.6f} | {e_per_atom:14.6f} | "
                f"{f_mean:10.6f} | {top_branch:10d} | {top_wt:8.4f}"
            )

    print("\n" + "=" * 80)
    print("BRANCH WEIGHT DISTRIBUTION (averaged over all structures)")
    print("=" * 80)
    all_w = torch.stack(all_weights, dim=0)
    mean_w = all_w.mean(dim=0)
    std_w = all_w.std(dim=0)
    print(f"{'Branch':>8} | {'Mean Wt':>10} | {'Std Wt':>10}")
    print("-" * 35)
    for b in range(mean_w.size(0)):
        print(f"{b:8d} | {mean_w[b].item():10.6f} | {std_w[b].item():10.6f}")

    dominant_counts = all_w.argmax(dim=1)
    print("\nDominant branch frequency:")
    for b in range(num_branches):
        count = int((dominant_counts == b).sum().item())
        if count > 0:
            print(
                f"  branch-{b}: {count}/{len(all_weights)} "
                f"({100.0 * count / len(all_weights):.1f}%)"
            )

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)

    if batch_latencies_ms:
        lat = np.array(batch_latencies_ms)
        total_ms = lat.sum()
        print(f"  Timed batches:       {len(lat)}")
        print(f"  Timed structures:    {total_timed_structures}")
        print(f"  Total wall time:     {total_ms:.1f} ms ({total_ms / 1000:.3f} s)")
        print(
            f"  Batch latency (ms):  mean={lat.mean():.1f}  std={lat.std():.1f}  "
            f"min={lat.min():.1f}  max={lat.max():.1f}"
        )
        per_struct_ms = total_ms / total_timed_structures
        print(f"  Per-structure:       {per_struct_ms:.2f} ms")
        throughput = total_timed_structures / (total_ms / 1000.0)
        print(f"  Throughput:          {throughput:.1f} structures/s")
    else:
        print("  No timed batches (all batches used for warmup).")

    if stage_stats:
        print("\n" + "=" * 80)
        has_encoder = "encoder_ms" in stage_stats
        mode_label = "encoder-reuse" if has_encoder else "baseline"
        print(
            f"STAGE TIMING (--profile_stages, {mode_label}, mean over timed batches, ms)"
        )
        print("=" * 80)
        stage_keys = [("mlp_ms", "MLP + softmax")]
        if has_encoder:
            stage_keys.append(("encoder_ms", "Encoder (1x)"))
            stage_keys.append(("branches_ms", "Decoders (Bx)"))
        else:
            stage_keys.append(("branches_ms", "Branch forwards (Bx full)"))
        stage_keys.append(("combine_ms", "Stack + weighted average"))
        for key, label in stage_keys:
            arr = np.array(stage_stats[key])
            print(
                f"  {label:26s}  mean={arr.mean():.2f}  std={arr.std():.2f}  "
                f"min={arr.min():.2f}  max={arr.max():.2f}"
            )

    print("\n" + "=" * 80)
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_argument_parser(
        description="Fused HydraGNN + BranchWeightMLP inference on random structures"
    )
    add_fused_cli_arguments(parser)
    parser.set_defaults(num_structures=100)
    args = parser.parse_args()

    (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        _gnn_prec_str,
        _mlp_prec_str,
    ) = load_fused_stack(
        args.logdir,
        args.checkpoint,
        args.mlp_checkpoint,
        args.precision,
        args.mlp_precision,
        args.mlp_device,
    )

    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)

    structures = generate_structures(
        args.num_structures,
        args.min_atoms,
        args.max_atoms,
        args.box_size,
        args.max_atomic_number,
        radius,
        max_neighbours,
        args.seed,
    )
    print(
        f"Generated {len(structures)} random structures "
        f"(atoms: {args.min_atoms}-{args.max_atoms}, box: {args.box_size} A)"
    )

    n_batches = (len(structures) + args.batch_size - 1) // args.batch_size
    num_warmup_eff = min(args.num_warmup, n_batches)
    print(
        f"Total batches: {n_batches} "
        f"(warmup: {num_warmup_eff}, timed: {n_batches - num_warmup_eff})"
    )

    omnistat_fom_url = None
    if args.omnistat_fom:
        omnistat_fom_url = f"http://localhost:{args.omnistat_fom_port}/fom"
    local_gpu_id = int(
        os.environ.get(
            "SLURM_LOCALID",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"),
        )
    )

    (
        all_energies,
        all_forces,
        all_natoms,
        all_weights,
        batch_latencies_ms,
        total_timed_structures,
        stage_stats,
    ) = run_fused_inference(
        model,
        mlp,
        structures,
        args.batch_size,
        param_dtype,
        autocast_ctx,
        device,
        num_branches,
        args.num_warmup,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        args.profile_stages,
        encoder_reuse=args.encoder_reuse,
        num_streams=args.num_streams,
        weight_threshold=args.weight_threshold,
        fused_energy_grad=args.fused_energy_grad,
        omnistat_fom_url=omnistat_fom_url,
        omnistat_fom_gpu_id=local_gpu_id,
        disable_param_grad=args.disable_param_grad,
        batched_decoder=args.batched_decoder,
        compile_encoder=args.compile_encoder,
        compile_decoder=args.compile_decoder,
        compile_full=args.compile_full,
        compile_backend=args.compile_backend,
    )

    print_fused_results(
        all_energies,
        all_forces,
        all_natoms,
        all_weights,
        num_branches,
        batch_latencies_ms,
        total_timed_structures,
        stage_stats,
    )


if __name__ == "__main__":
    main()
