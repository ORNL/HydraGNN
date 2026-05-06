"""Generate FT1 feasibility-classification dataset from existing OPF HDF5 data.

Infeasible OPF instances are synthesised by scaling all load features (Pd, Qd)
by a large overload factor, so that total demand exceeds total generation
capacity — guaranteeing AC-OPF infeasibility without running a solver.

The output is a balanced HDF5 dataset (50 % feasible, 50 % infeasible) with a
graph-level binary label stored as ``data.y = torch.tensor([1.0])`` (feasible)
or ``data.y = torch.tensor([0.0])`` (infeasible).

Usage (single rank, no MPI required)::

    python generate_infeasible_samples.py \\
        --src_dir  ../dataset/FT3_contingency_HeteroSAGE_data.h5 \\
        --out_dir  ../dataset/FT1_feasibility_data.h5 \\
        --overload_factor 6.0 \\
        --max_samples 5000

The output directory can then be used directly by train_opf_ft1_classify.py
via the config key ``"ft_data_modelname": "FT1_feasibility_data"``.
"""

import os
import sys
import copy
import random
import argparse

import torch
from mpi4py import MPI

# Make examples/opf importable (for opf_solution_utils if ever needed)
_OPF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _OPF_DIR)

from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset, HDF5Writer


# ---------------------------------------------------------------------------
# Sample manipulation helpers
# ---------------------------------------------------------------------------

def strip_node_targets(data):
    """Remove node-level prediction targets and y_loc; keep all node features."""
    for node_type in list(data.node_types):
        node_store = data[node_type]
        if hasattr(node_store, "y"):
            del node_store["y"]
    if hasattr(data, "y_loc"):
        del data["y_loc"]
    # Wipe top-level y so we can set the graph-level label cleanly
    if hasattr(data, "y"):
        del data["y"]
    return data


def make_infeasible(data, overload_factor: float):
    """Return a deep copy of *data* with load features scaled by *overload_factor*.

    Only the ``load`` node-type features are scaled (indices 0 = Pd, 1 = Qd).
    All other features (bus, generator, shunt, edge_attr) are unchanged.
    """
    infeasible = copy.deepcopy(data)
    if "load" in infeasible.node_types and infeasible["load"].x is not None:
        infeasible["load"].x = infeasible["load"].x * overload_factor
    return infeasible


def label_samples(samples, label: float):
    """Assign graph-level feasibility label to a list of samples in-place."""
    y = torch.tensor([label], dtype=torch.float32)
    for s in samples:
        s.y = y.clone()
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate FT1 feasibility classification HDF5 dataset.",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help=(
            "Source HDF5 directory containing feasible OPF samples "
            "(e.g. ../dataset/FT3_contingency_HeteroSAGE_data.h5)."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output HDF5 directory for the FT1 mixed feasibility dataset.",
    )
    parser.add_argument(
        "--overload_factor",
        type=float,
        default=6.0,
        help=(
            "Factor by which load features (Pd, Qd) are multiplied to create "
            "infeasible samples.  A value >=5 reliably causes infeasibility for "
            "typical pglib-opf test cases."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of feasible samples to use (randomly subsampled). "
            "The same number of infeasible samples is generated, giving a "
            "perfectly balanced dataset of size 2 * max_samples."
        ),
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.70,
        help="Fraction of the mixed dataset assigned to the train split.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
        help="Fraction assigned to the val split (remainder goes to test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Only rank-0 does the work (single-process preprocessing)
    if rank != 0:
        comm.Barrier()
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load all feasible samples from the source HDF5 ─────────────────────
    if not os.path.isdir(args.src_dir):
        raise FileNotFoundError(
            f"Source HDF5 directory not found: '{args.src_dir}'. "
            "Run the FT3 (or any other OPF) preprocessing step first to "
            "generate a feasible-sample HDF5 dataset."
        )

    # ── Reservoir-sample up to max_samples to avoid loading full dataset ───
    reservoir = []
    n_seen = 0
    cap = args.max_samples  # None means no cap

    for split in ("trainset", "valset", "testset"):
        try:
            ds = HDF5Dataset(args.src_dir, split)
            for i in range(len(ds)):
                sample = ds[i]
                strip_node_targets(sample)
                if cap is None or n_seen < cap:
                    reservoir.append(sample)
                else:
                    j = random.randint(0, n_seen)
                    if j < cap:
                        reservoir[j] = sample
                n_seen += 1
                if (n_seen % 5000) == 0:
                    print(f"  Scanned {n_seen} samples, kept {len(reservoir)}...")
        except Exception as exc:
            print(f"  Warning: could not load split '{split}' from {args.src_dir}: {exc}")

    all_feasible = reservoir
    print(f"Loaded {len(all_feasible)} feasible samples from {args.src_dir} (scanned {n_seen} total)")

    if len(all_feasible) == 0:
        raise RuntimeError("No feasible samples found in the source dataset.")

    n_base = len(all_feasible)

    # ── Build feasible samples (label = 1.0) ───────────────────────────────
    feasible = label_samples(all_feasible, label=1.0)

    # ── Build infeasible samples (label = 0.0) ─────────────────────────────
    print(
        f"Generating {n_base} infeasible samples "
        f"(overload_factor={args.overload_factor})..."
    )
    infeasible = [make_infeasible(s, args.overload_factor) for s in all_feasible]
    infeasible = label_samples(infeasible, label=0.0)

    # ── Mix and shuffle ────────────────────────────────────────────────────
    mixed = feasible + infeasible
    random.shuffle(mixed)
    n_total = len(mixed)
    print(f"Total mixed samples: {n_total}  ({n_base} feasible + {n_base} infeasible)")

    # ── Split ──────────────────────────────────────────────────────────────
    n_train = int(args.train_frac * n_total)
    n_val   = int(args.val_frac   * n_total)
    splits = {
        "trainset": mixed[:n_train],
        "valset":   mixed[n_train : n_train + n_val],
        "testset":  mixed[n_train + n_val :],
    }
    for name, s in splits.items():
        label_counts = {
            0: sum(1 for x in s if x.y.item() < 0.5),
            1: sum(1 for x in s if x.y.item() >= 0.5),
        }
        print(
            f"  {name}: {len(s)} samples  "
            f"(feasible={label_counts[1]}, infeasible={label_counts[0]})"
        )

    # ── Write HDF5 ─────────────────────────────────────────────────────────
    print(f"Writing dataset to {args.out_dir} ...")
    writer = HDF5Writer(args.out_dir, comm=MPI.COMM_SELF)
    for split_name, split_samples in splits.items():
        writer.add(split_name, split_samples)
    writer.save()
    print("Done.")

    comm.Barrier()


if __name__ == "__main__":
    main()
