#!/usr/bin/env python3
"""Inference on randomly generated atomistic structures using a trained HydraGNN model.

Generates random structures with variable numbers of atoms, atom types, and
positions, then runs the frozen HydraGNN model to predict energy (and forces
via autograd when the model uses interatomic potentials).

Usage:
    python inference_random_structures.py \
        --logdir <path_to_training_log_dir> \
        --num_structures 10 \
        --min_atoms 2 --max_atoms 20 \
        --box_size 10.0

The --logdir should contain a config.json and a .pk checkpoint file produced
by a prior HydraGNN training run.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RadiusGraph

from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import (
    resolve_precision,
    move_batch_to_device,
    get_autocast_and_scaler,
)
from hydragnn.utils.distributed import get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_checkpoint(logdir: str, checkpoint: str = None) -> str:
    """Locate a checkpoint file inside logdir."""
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


def _build_random_structure(
    min_atoms: int,
    max_atoms: int,
    box_size: float,
    max_atomic_number: int,
    rng: np.random.Generator,
) -> Data:
    """Create a single random atomistic graph (no edges yet)."""
    n_atoms = rng.integers(min_atoms, max_atoms + 1)
    # Random atom types (atomic numbers from 1 to max_atomic_number)
    atomic_numbers = rng.integers(1, max_atomic_number + 1, size=n_atoms)
    # Random 3-D positions inside a cubic box
    positions = rng.uniform(0.0, box_size, size=(n_atoms, 3))

    dtype = torch.get_default_dtype()
    x = torch.tensor(atomic_numbers, dtype=dtype).unsqueeze(1)  # (N, 1)
    pos = torch.tensor(positions, dtype=dtype)

    # Chemical composition histogram (118 elements)
    hist, _ = np.histogram(atomic_numbers, bins=range(1, 118 + 2))
    chemical_composition = torch.tensor(hist, dtype=torch.float32).unsqueeze(1)

    # Graph-level attributes (charge=0, spin=0 as neutral default)
    graph_attr = torch.tensor([0.0, 0.0], dtype=torch.float32)

    data = Data(
        x=x,
        pos=pos,
        chemical_composition=chemical_composition,
        graph_attr=graph_attr,
        natoms=torch.tensor([n_atoms], dtype=torch.long),
    )
    return data


def _add_edges(data_list, radius, max_neighbours):
    """Add radius-graph edges to each structure and batch them."""
    transform = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_neighbours)
    processed = []
    for data in data_list:
        data = transform(data)
        # Ensure edge_shifts exists (zeros for non-periodic systems)
        if not hasattr(data, "edge_shifts") or data.edge_shifts is None:
            n_edges = data.edge_index.size(1)
            data.edge_shifts = torch.zeros(n_edges, 3, dtype=data.pos.dtype)
        processed.append(data)
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="HydraGNN inference on randomly generated atomistic structures"
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Training log directory containing config.json and a .pk checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename or path (defaults to latest .pk in --logdir)",
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=10,
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
        "--branch_id",
        type=int,
        default=0,
        help="Branch index to use for multi-branch models (default: 0)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Override precision (fp32, fp64, bf16)",
    )
    args = parser.parse_args()

    # ----- Load config -----
    config_path = os.path.join(args.logdir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {args.logdir}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Resolve precision
    precision_str = args.precision or config["NeuralNetwork"]["Training"].get(
        "precision", "fp32"
    )
    precision, param_dtype, _ = resolve_precision(precision_str)
    torch.set_default_dtype(param_dtype)

    device = get_device()
    print(f"Device: {device}, precision: {precision}")

    autocast_ctx, _ = get_autocast_and_scaler(precision)

    # ----- Build and load model -----
    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )
    # Restore default dtype: create_model_config overwrites it with the training precision.
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=device)

    ckpt_path = _find_checkpoint(args.logdir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ----- Read architecture parameters needed for graph construction -----
    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)
    enable_ip = arch.get("enable_interatomic_potential", False)

    # ----- Generate random structures -----
    rng = np.random.default_rng(args.seed)
    structures = [
        _build_random_structure(
            args.min_atoms, args.max_atoms, args.box_size, args.max_atomic_number, rng
        )
        for _ in range(args.num_structures)
    ]
    structures = _add_edges(structures, radius, max_neighbours)

    # Assign branch ID (for multi-branch models)
    for s in structures:
        s.dataset_name = torch.tensor([[args.branch_id]], dtype=torch.long)

    print(
        f"Generated {len(structures)} random structures (atoms: {args.min_atoms}-{args.max_atoms}, box: {args.box_size} A)"
    )

    # ----- Run inference in batches -----
    all_energies = []
    all_forces = []  # list of per-structure force tensors
    all_natoms = []

    for start in range(0, len(structures), args.batch_size):
        batch_list = structures[start : start + args.batch_size]
        batch = Batch.from_data_list(batch_list)
        batch = move_batch_to_device(batch, param_dtype)
        batch.pos.requires_grad_(True)

        with torch.enable_grad(), autocast_ctx:
            outputs = model(batch)

        # First head is energy for interatomic-potential models
        energy_pred = outputs[0].squeeze()
        if energy_pred.dim() == 0:
            energy_pred = energy_pred.unsqueeze(0)

        # Compute forces via autograd if interatomic potential is enabled
        if enable_ip and batch.pos.requires_grad:
            forces_pred = -torch.autograd.grad(
                energy_pred.sum(),
                batch.pos,
                create_graph=False,
                retain_graph=False,
            )[0].detach()
        else:
            forces_pred = None

        energy_pred = energy_pred.detach()
        # Collect per-structure results
        num_graphs = batch.num_graphs
        for i in range(num_graphs):
            all_energies.append(
                energy_pred[i].item() if energy_pred.numel() > 1 else energy_pred.item()
            )
            mask = batch.batch == i
            n = int(mask.sum().item())
            all_natoms.append(n)
            if forces_pred is not None:
                all_forces.append(forces_pred[mask].cpu())
            else:
                all_forces.append(None)

    # ----- Print results -----
    print("\n" + "=" * 70)
    print(
        f"{'Struct':>6} | {'Atoms':>5} | {'Energy':>16} | {'Energy/atom':>16} | {'|F|_mean':>12}"
    )
    print("-" * 70)
    for i in range(len(all_energies)):
        e = all_energies[i]
        n = all_natoms[i]
        e_per_atom = e / n
        if all_forces[i] is not None:
            f_norms = all_forces[i].norm(dim=1)
            f_mean = f_norms.mean().item()
            f_str = f"{f_mean:12.6f}"
        else:
            f_str = "       N/A"
        print(f"{i:6d} | {n:5d} | {e:16.6f} | {e_per_atom:16.6f} | {f_str}")
    print("=" * 70)


if __name__ == "__main__":
    main()
