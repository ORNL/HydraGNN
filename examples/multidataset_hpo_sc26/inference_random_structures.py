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

The public functions in this module (find_checkpoint, build_random_structure,
add_edges, build_argument_parser, load_config_and_model,
generate_structures, run_inference) can be imported and reused by other
scripts.
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


def find_checkpoint(logdir: str, checkpoint: str = None) -> str:
    """Locate a checkpoint file inside *logdir*."""
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


def build_random_structure(
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


def add_edges(data_list, radius, max_neighbours):
    """Add radius-graph edges to each structure."""
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
# Reusable building blocks
# ---------------------------------------------------------------------------


def build_argument_parser(description=None):
    """Return an :class:`argparse.ArgumentParser` with the common arguments.

    Callers can extend the returned parser with additional arguments before
    calling ``parse_args()``.
    """
    parser = argparse.ArgumentParser(
        description=description
        or "HydraGNN inference on randomly generated atomistic structures"
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
    return parser


def load_config_and_model(logdir, checkpoint=None, precision_override=None):
    """Load config, build the model, and restore a checkpoint.

    Returns
    -------
    model : torch.nn.Module
    config : dict
    device : torch.device
    autocast_ctx : context-manager
    param_dtype : torch.dtype
    """
    config_path = os.path.join(logdir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {logdir}")
    with open(config_path, "r") as f:
        config = json.load(f)

    precision_str = precision_override or config["NeuralNetwork"]["Training"].get(
        "precision", "fp32"
    )
    precision, param_dtype, _ = resolve_precision(precision_str)
    torch.set_default_dtype(param_dtype)

    device = get_device()
    autocast_ctx, _ = get_autocast_and_scaler(precision)

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )
    # Restore default dtype: create_model_config overwrites it with the training precision.
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=device)

    ckpt_path = find_checkpoint(logdir, checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Device: {device}, precision: {precision}")

    return model, config, device, autocast_ctx, param_dtype


def generate_structures(
    num_structures,
    min_atoms,
    max_atoms,
    box_size,
    max_atomic_number,
    radius,
    max_neighbours,
    branch_id,
    seed,
):
    """Generate random atomistic structures with edges and branch labels.

    Returns
    -------
    structures : list[Data]
    """
    rng = np.random.default_rng(seed)
    structures = [
        build_random_structure(min_atoms, max_atoms, box_size, max_atomic_number, rng)
        for _ in range(num_structures)
    ]
    structures = add_edges(structures, radius, max_neighbours)
    for s in structures:
        s.dataset_name = torch.tensor([[branch_id]], dtype=torch.long)
    return structures


def run_inference(model, structures, batch_size, param_dtype, autocast_ctx, enable_ip):
    """Run batched inference and return per-structure energies and forces.

    Returns
    -------
    all_energies : list[float]
    all_forces : list[torch.Tensor | None]
        Each entry is an (N_atoms, 3) CPU tensor, or *None* when forces are
        unavailable.
    all_natoms : list[int]
    """
    all_energies = []
    all_forces = []
    all_natoms = []

    for start in range(0, len(structures), batch_size):
        batch_list = structures[start : start + batch_size]
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

    return all_energies, all_forces, all_natoms


def print_results(all_energies, all_forces, all_natoms):
    """Print a table of inference results to stdout."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = build_argument_parser().parse_args()

    model, config, device, autocast_ctx, param_dtype = load_config_and_model(
        args.logdir,
        args.checkpoint,
        args.precision,
    )

    arch = config["NeuralNetwork"]["Architecture"]
    radius = arch.get("radius", 5.0)
    max_neighbours = arch.get("max_neighbours", 20)
    enable_ip = arch.get("enable_interatomic_potential", False)

    structures = generate_structures(
        args.num_structures,
        args.min_atoms,
        args.max_atoms,
        args.box_size,
        args.max_atomic_number,
        radius,
        max_neighbours,
        args.branch_id,
        args.seed,
    )
    print(
        f"Generated {len(structures)} random structures "
        f"(atoms: {args.min_atoms}-{args.max_atoms}, box: {args.box_size} A)"
    )

    all_energies, all_forces, all_natoms = run_inference(
        model,
        structures,
        args.batch_size,
        param_dtype,
        autocast_ctx,
        enable_ip,
    )

    print_results(all_energies, all_forces, all_natoms)


if __name__ == "__main__":
    main()
