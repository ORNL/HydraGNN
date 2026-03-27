#!/usr/bin/env python3
"""Fused inference: HydraGNN (multi-branch) + BranchWeightMLP.

Loads a pretrained multi-branch HydraGNN model and a trained BranchWeightMLP,
generates random atomistic structures, runs all branches in parallel, and
produces a weighted-average prediction of energy and forces. Includes detailed
latency and throughput measurements.

Usage:
    python inference_fused.py \
        --logdir <path_to_training_log_dir> \
        --num_structures 100 \
        --batch_size 32

The --logdir should contain config.json, a .pk HydraGNN checkpoint, and a
mlp_weights/ subdirectory with .pt MLP checkpoints.  The script auto-selects
the most recent checkpoint of each type unless overridden.
"""

import argparse
import glob
import json
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RadiusGraph

from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import (
    resolve_precision,
    get_autocast_and_scaler,
)
from hydragnn.utils.distributed import get_device


# ---------------------------------------------------------------------------
# Helpers: checkpoint discovery
# ---------------------------------------------------------------------------


def _find_checkpoint(logdir: str, checkpoint: str = None) -> str:
    """Locate a .pk HydraGNN checkpoint inside *logdir*."""
    if checkpoint is not None:
        path = (
            checkpoint
            if os.path.isabs(checkpoint)
            else os.path.join(logdir, checkpoint)
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = sorted(glob.glob(os.path.join(logdir, "*.pk")))
    if not candidates:
        raise FileNotFoundError(f"No .pk checkpoint found in {logdir}")
    return candidates[-1]


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
# Helpers: random structure generation
# ---------------------------------------------------------------------------


def _build_random_structure(
    min_atoms: int,
    max_atoms: int,
    box_size: float,
    max_atomic_number: int,
    rng: np.random.Generator,
) -> Data:
    """Create a single random atomistic graph (no edges yet)."""
    n_atoms = rng.integers(min_atoms, max_atoms + 1)
    atomic_numbers = rng.integers(1, max_atomic_number + 1, size=n_atoms)
    positions = rng.uniform(0.0, box_size, size=(n_atoms, 3))

    dtype = torch.get_default_dtype()
    x = torch.tensor(atomic_numbers, dtype=dtype).unsqueeze(1)
    pos = torch.tensor(positions, dtype=dtype)

    hist, _ = np.histogram(atomic_numbers, bins=range(1, 118 + 2))
    chemical_composition = torch.tensor(hist, dtype=torch.float32).unsqueeze(1)

    graph_attr = torch.tensor([0.0, 0.0], dtype=torch.float32)

    return Data(
        x=x,
        pos=pos,
        chemical_composition=chemical_composition,
        graph_attr=graph_attr,
        natoms=torch.tensor([n_atoms], dtype=torch.long),
    )


def _add_edges(data_list, radius, max_neighbours):
    """Add radius-graph edges to each structure."""
    transform = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_neighbours)
    processed = []
    for data in data_list:
        data = transform(data)
        if not hasattr(data, "edge_shifts") or data.edge_shifts is None:
            n_edges = data.edge_index.size(1)
            data.edge_shifts = torch.zeros(n_edges, 3, dtype=data.pos.dtype)
        processed.append(data)
    return processed


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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fused HydraGNN + BranchWeightMLP inference on random structures"
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Training log directory (config.json, .pk checkpoint, mlp_weights/)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="HydraGNN checkpoint filename or path (defaults to latest .pk in --logdir)",
    )
    parser.add_argument(
        "--mlp_checkpoint",
        default=None,
        help="MLP checkpoint path (defaults to newest .pt in <logdir>/mlp_weights/)",
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=100,
        help="Number of random structures to generate",
    )
    parser.add_argument(
        "--min_atoms", type=int, default=2, help="Minimum atoms per structure"
    )
    parser.add_argument(
        "--max_atoms", type=int, default=20, help="Maximum atoms per structure"
    )
    parser.add_argument(
        "--box_size",
        type=float,
        default=10.0,
        help="Side length of the cubic box for random positions (Angstrom)",
    )
    parser.add_argument(
        "--max_atomic_number",
        type=int,
        default=94,
        help="Maximum atomic number for random atom types",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Override precision (fp32, fp64, bf16)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=1,
        help="Number of warmup batches excluded from timing",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config_path = os.path.join(args.logdir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {args.logdir}")
    with open(config_path, "r") as f:
        config = json.load(f)

    precision_str = args.precision or config["NeuralNetwork"]["Training"].get(
        "precision", "fp32"
    )
    precision, param_dtype, _ = resolve_precision(precision_str)
    torch.set_default_dtype(param_dtype)

    device = get_device()
    print(f"Device: {device}, precision: {precision_str}")

    autocast_ctx, _ = get_autocast_and_scaler(precision)

    # ---- Load HydraGNN model ----
    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )
    model = model.to(dtype=param_dtype, device=device)

    ckpt_path = _find_checkpoint(args.logdir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_branches = getattr(model, "num_branches", 1)
    print(f"HydraGNN checkpoint: {ckpt_path}")
    print(f"  num_branches = {num_branches}")

    # ---- Load BranchWeightMLP ----
    mlp_path = _find_mlp_checkpoint(args.logdir, args.mlp_checkpoint)
    mlp_ckpt = torch.load(mlp_path, map_location=device)
    mlp = _reconstruct_mlp_from_state_dict(mlp_ckpt["mlp_state_dict"])
    mlp = mlp.to(dtype=param_dtype, device=device)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    linear_keys = sorted(
        k for k in mlp_ckpt["mlp_state_dict"] if k.endswith(".weight") and "net." in k
    )
    sd = mlp_ckpt["mlp_state_dict"]
    input_dim = sd[linear_keys[0]].shape[1]
    hidden_dims = [sd[k].shape[0] for k in linear_keys[:-1]]
    mlp_out = sd[linear_keys[-1]].shape[0]
    print(f"MLP checkpoint: {mlp_path}")
    print(f"  architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {mlp_out}")

    if mlp_out != num_branches:
        print(
            f"WARNING: MLP output dim ({mlp_out}) != HydraGNN num_branches "
            f"({num_branches}). Results may be incorrect."
        )

    # ---- Architecture parameters for graph construction ----
    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)

    # ---- Generate random structures ----
    rng = np.random.default_rng(args.seed)
    structures = [
        _build_random_structure(
            args.min_atoms, args.max_atoms, args.box_size, args.max_atomic_number, rng
        )
        for _ in range(args.num_structures)
    ]
    structures = _add_edges(structures, radius, max_neighbours)

    print(
        f"Generated {len(structures)} random structures "
        f"(atoms: {args.min_atoms}-{args.max_atoms}, box: {args.box_size} A)"
    )

    # ---- Prepare batches ----
    batches = []
    for start in range(0, len(structures), args.batch_size):
        batches.append(structures[start : start + args.batch_size])

    num_warmup = min(args.num_warmup, len(batches))
    print(f"Total batches: {len(batches)} (warmup: {num_warmup}, timed: {len(batches) - num_warmup})")

    # ---- Fused inference ----
    timer_start, timer_stop_ms = _make_timer(device)

    all_energies = []
    all_forces = []
    all_natoms = []
    all_weights = []
    batch_latencies_ms = []

    for batch_idx, batch_list in enumerate(batches):
        is_warmup = batch_idx < num_warmup

        batch = Batch.from_data_list(batch_list)
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
                batch[key] = val.to(device=device, dtype=param_dtype)
            elif isinstance(val, torch.Tensor):
                batch[key] = val.to(device=device)
        batch.pos.requires_grad_(True)

        timer_start()

        with torch.enable_grad(), autocast_ctx:
            comp = _reshape_composition(batch).to(device=device, dtype=param_dtype)
            logits = mlp(comp)
            weights = F.softmax(logits, dim=-1)

            energy_preds = []
            forces_preds = []
            for branch_id in range(num_branches):
                e, f = _predict_branch_energy_forces(model, batch, branch_id)
                energy_preds.append(e)
                forces_preds.append(f)

            energy_preds_t = torch.stack(energy_preds, dim=0)
            forces_preds_t = torch.stack(forces_preds, dim=0)

            weighted_energy, weighted_forces = _weighted_average(
                energy_preds_t, forces_preds_t, weights, batch.batch
            )

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

    # ---- Per-structure results ----
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
        show_indices = list(range(show_limit)) + list(range(num_results - show_limit, num_results))
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

    # ---- Branch weight distribution summary ----
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
    print(f"\nDominant branch frequency:")
    for b in range(num_branches):
        count = int((dominant_counts == b).sum().item())
        if count > 0:
            print(f"  branch-{b}: {count}/{len(all_weights)} ({100.0 * count / len(all_weights):.1f}%)")

    # ---- Timing statistics ----
    total_timed_structures = sum(
        len(batches[i]) for i in range(num_warmup, len(batches))
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
        print(f"  Batch latency (ms):  mean={lat.mean():.1f}  std={lat.std():.1f}  "
              f"min={lat.min():.1f}  max={lat.max():.1f}")
        per_struct_ms = total_ms / total_timed_structures
        print(f"  Per-structure:       {per_struct_ms:.2f} ms")
        throughput = total_timed_structures / (total_ms / 1000.0)
        print(f"  Throughput:          {throughput:.1f} structures/s")
    else:
        print("  No timed batches (all batches used for warmup).")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
