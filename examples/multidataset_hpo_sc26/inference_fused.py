#!/usr/bin/env python3
"""Fused inference: HydraGNN (multi-branch) + BranchWeightMLP.

Loads a pretrained multi-branch HydraGNN model and a trained BranchWeightMLP,
generates random atomistic structures, runs all branches in parallel, and
produces a weighted-average prediction of energy and forces. Includes detailed
latency and throughput measurements.

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
        mlp_precision_override
        if mlp_precision_override is not None
        else gnn_prec_str
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
    print(f"  architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {mlp_out}")
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
        build_random_structure(
            min_atoms, max_atoms, box_size, max_atomic_number, rng
        )
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
    model, data, branch_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dataset_name = getattr(data, "dataset_name", None)
    data.dataset_name = _build_dataset_name(data, branch_id)

    pred = model(data)
    energy_pred = _energy_from_pred(pred)
    forces_pred = torch.autograd.grad(
        energy_pred,
        data.pos,
        grad_outputs=torch.ones_like(energy_pred),
        retain_graph=True,
        create_graph=False,
    )[0]
    forces_pred = -forces_pred

    if original_dataset_name is None and hasattr(data, "dataset_name"):
        delattr(data, "dataset_name")
    else:
        data.dataset_name = original_dataset_name

    return energy_pred.detach(), forces_pred.detach()


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
):
    """Run batched fused inference with timing (excludes warmup batches).

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
        If profile_stages, keys mlp_ms, branches_ms, combine_ms (lists per timed batch).
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
    stage_branches_ms: List[float] = []
    stage_combine_ms: List[float] = []

    mlp_param_dtype = next(mlp.parameters()).dtype

    for batch_idx, batch_list in enumerate(batches):
        is_warmup = batch_idx < num_warmup_effective

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
                    energy_preds = []
                    forces_preds = []
                    for branch_id in range(num_branches):
                        e, f = _predict_branch_energy_forces(model, batch, branch_id)
                        energy_preds.append(e)
                        forces_preds.append(f)
                    if profile_stages:
                        _sync_device(device)
                        t_br0 = time.perf_counter()
                    energy_preds_t = torch.stack(energy_preds, dim=0)
                    forces_preds_t = torch.stack(forces_preds, dim=0)
                    weighted_energy, weighted_forces = _weighted_average(
                        energy_preds_t, forces_preds_t, weights, batch.batch
                    )
                    if profile_stages:
                        _sync_device(device)
                        t_cb0 = time.perf_counter()
                if profile_stages and not is_warmup:
                    stage_mlp_ms.append((t_mlp1 - t_mlp0) * 1000.0)
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
                weights = F.softmax(logits, dim=-1).to(
                    device=device, dtype=param_dtype
                )
                if profile_stages:
                    _sync_device(device)
                    t_mlp1 = time.perf_counter()
                with autocast_ctx:
                    energy_preds = []
                    forces_preds = []
                    for branch_id in range(num_branches):
                        e, f = _predict_branch_energy_forces(model, batch, branch_id)
                        energy_preds.append(e)
                        forces_preds.append(f)
                    if profile_stages:
                        _sync_device(device)
                        t_br0 = time.perf_counter()
                    energy_preds_t = torch.stack(energy_preds, dim=0)
                    forces_preds_t = torch.stack(forces_preds, dim=0)
                    weighted_energy, weighted_forces = _weighted_average(
                        energy_preds_t, forces_preds_t, weights, batch.batch
                    )
                    if profile_stages:
                        _sync_device(device)
                        t_cb0 = time.perf_counter()
                if profile_stages and not is_warmup:
                    stage_mlp_ms.append((t_mlp1 - t_mlp0) * 1000.0)
                    stage_branches_ms.append((t_br0 - t_mlp1) * 1000.0)
                    stage_combine_ms.append((t_cb0 - t_br0) * 1000.0)

        elapsed_ms = timer_stop_ms()

        if not is_warmup:
            batch_latencies_ms.append(elapsed_ms)

        weighted_energy = weighted_energy.detach()
        weighted_forces = weighted_forces.detach()
        weights_cpu = weights.detach().cpu()

        num_graphs = batch.num_graphs
        for i in range(num_graphs):
            all_energies.append(
                weighted_energy[i].item()
                if weighted_energy.numel() > 1
                else weighted_energy.item()
            )
            mask = batch.batch == i
            n = int(mask.sum().item())
            all_natoms.append(n)
            all_forces.append(weighted_forces[mask].cpu())
            all_weights.append(weights_cpu[i])

    total_timed_structures = sum(
        len(batches[i]) for i in range(num_warmup_effective, len(batches))
    )

    stage_stats = None
    if profile_stages and stage_mlp_ms:
        stage_stats = {
            "mlp_ms": stage_mlp_ms,
            "branches_ms": stage_branches_ms,
            "combine_ms": stage_combine_ms,
        }

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
        print("STAGE TIMING (--profile_stages, mean over timed batches, ms)")
        print("=" * 80)
        for key, label in (
            ("mlp_ms", "MLP + softmax"),
            ("branches_ms", "Branch forwards"),
            ("combine_ms", "Stack + weighted average"),
        ):
            arr = np.array(stage_stats[key])
            print(
                f"  {label:22s}  mean={arr.mean():.2f}  std={arr.std():.2f}  "
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
