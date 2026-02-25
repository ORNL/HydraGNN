#!/usr/bin/env python3
"""Compute element occurrence probabilities from multiple HydraGNN ADIOS datasets.

This script follows the same dataset-loading pattern used in
examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py:
- dataset files are expected as <dataset_name>-v2.bp
- ADIOS data is read through hydragnn.utils.datasets.adiosdataset.AdiosDataset
- train/val/test splits are processed

For each dataset, a JSON file is created containing probabilities for all 118
periodic-table elements.
"""

import argparse
import json
import os
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpi4py import MPI

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError as exc:
    raise ImportError(
        "AdiosDataset is unavailable; install adios2 and HydraGNN dataset dependencies."
    ) from exc


PERIODIC_TABLE_SYMBOLS: List[str] = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


COMMON_VARIABLE_NAMES = [
    "pbc",
    "edge_attr",
    "energy_per_atom",
    "forces",
    "pos",
    "edge_index",
    "cell",
    "edge_shifts",
    "y",
    "chemical_composition",
    "natoms",
    "x",
    "energy",
    "graph_attr",
    "atomic_numbers",
]


def _to_numpy(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def _hist_from_data(data) -> np.ndarray:
    if hasattr(data, "chemical_composition"):
        comp = _to_numpy(data.chemical_composition).reshape(-1)
        if comp.size == 118:
            return comp.astype(np.float64)
        if comp.size % 118 == 0:
            return comp.reshape(-1, 118).sum(axis=0).astype(np.float64)

    if hasattr(data, "atomic_numbers"):
        atomic_numbers = _to_numpy(data.atomic_numbers).reshape(-1)
    elif hasattr(data, "x"):
        x = _to_numpy(data.x)
        if x.ndim < 2 or x.shape[1] < 1:
            raise ValueError("Cannot infer atomic numbers from data.x")
        atomic_numbers = x[:, 0].reshape(-1)
    else:
        raise ValueError("Data object has neither atomic_numbers nor x")

    atomic_numbers = np.rint(atomic_numbers).astype(np.int64)
    valid = (atomic_numbers >= 1) & (atomic_numbers <= 118)
    atomic_numbers = atomic_numbers[valid]
    if atomic_numbers.size == 0:
        return np.zeros(118, dtype=np.float64)

    hist = np.bincount(atomic_numbers, minlength=119)[1:119]
    return hist.astype(np.float64)


def _parse_model_list(multi_model_list: str) -> List[str]:
    models = [name.strip() for name in multi_model_list.split(",") if name.strip()]
    if len(models) == 0:
        raise ValueError("--multi_model_list resulted in zero entries")
    return models


def _iter_splits(splits: Iterable[str]) -> List[str]:
    normalized = []
    for split in splits:
        split_value = split.strip()
        if split_value:
            normalized.append(split_value)
    if len(normalized) == 0:
        raise ValueError("At least one split must be provided")
    return normalized


def _compute_dataset_histogram(dataset_path: str, splits: List[str]) -> np.ndarray:
    total_hist = np.zeros(118, dtype=np.float64)
    comm_self = MPI.COMM_SELF

    for split in splits:
        dataset = AdiosDataset(
            dataset_path,
            split,
            comm_self,
            keys=COMMON_VARIABLE_NAMES,
        )
        for sample_idx in range(len(dataset)):
            data = dataset[sample_idx]
            total_hist += _hist_from_data(data)

    return total_hist


def _probability_dict(hist: np.ndarray, global_total_atoms: float):
    if global_total_atoms <= 0.0:
        probs = np.zeros(118, dtype=np.float64)
    else:
        probs = hist / global_total_atoms

    by_symbol = {
        symbol: float(prob) for symbol, prob in zip(PERIODIC_TABLE_SYMBOLS, probs)
    }
    by_atomic_number = {str(i + 1): float(prob) for i, prob in enumerate(probs)}
    return by_symbol, by_atomic_number


def _plot_histogram(values: np.ndarray, title: str, output_path: str, ylabel: str):
    atomic_numbers = np.arange(1, 119)
    plt.figure(figsize=(16, 4))
    plt.bar(atomic_numbers, values, width=0.85)
    plt.title(title)
    plt.xlabel("Atomic number")
    plt.ylabel(ylabel)
    plt.xlim(0.5, 118.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing ADIOS dataset files",
    )
    parser.add_argument(
        "--multi_model_list",
        type=str,
        required=True,
        help="Comma-separated dataset names (e.g., Alexandria,ANI1x,MPTrj)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="-v2.bp",
        help="Suffix appended to each dataset name to form the ADIOS filename",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["trainset", "valset", "testset"],
        help="Dataset splits to aggregate",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    models = _parse_model_list(args.multi_model_list)
    splits = _iter_splits(args.splits)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.dataset_dir, "element_probabilities")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        dataset_histograms = {}
        dataset_paths = {}
        for model in models:
            dataset_path = os.path.join(args.dataset_dir, f"{model}{args.file_suffix}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            dataset_paths[model] = dataset_path
            print(f"Processing dataset: {model}")
            dataset_histograms[model] = _compute_dataset_histogram(dataset_path, splits)

        global_hist = np.zeros(118, dtype=np.float64)
        for model in models:
            global_hist += dataset_histograms[model]
        global_total_atoms = float(global_hist.sum())

        global_plot_path = os.path.join(output_dir, "global_element_probability_histogram.png")
        _plot_histogram(
            global_hist,
            "Global Element Histogram (Counts)",
            global_plot_path,
            "Count",
        )
        print(f"Wrote: {global_plot_path}")

        for model in models:
            hist = dataset_histograms[model]
            by_symbol, by_atomic_number = _probability_dict(hist, global_total_atoms)

            model_plot_path = os.path.join(
                output_dir, f"{model}_element_probability_histogram.png"
            )
            _plot_histogram(
                hist,
                f"Element Histogram (Counts) - {model}",
                model_plot_path,
                "Count",
            )
            print(f"Wrote: {model_plot_path}")

            output = {
                "dataset": model,
                "dataset_path": dataset_paths[model],
                "splits": splits,
                "dataset_total_atoms": int(hist.sum()),
                "global_total_atoms": int(global_total_atoms),
                "normalization": "global_across_all_datasets",
                "element_probabilities": by_symbol,
                "element_probabilities_by_atomic_number": by_atomic_number,
            }

            output_path = os.path.join(output_dir, f"{model}_element_probabilities.json")
            with open(output_path, "w", encoding="utf-8") as fout:
                json.dump(output, fout, indent=2)

            print(f"Wrote: {output_path}")

        global_output = {
            "datasets": models,
            "splits": splits,
            "global_total_atoms": int(global_total_atoms),
            "global_element_counts_by_atomic_number": {
                str(i + 1): int(v) for i, v in enumerate(global_hist.tolist())
            },
            "global_element_counts_by_symbol": {
                sym: int(global_hist[i]) for i, sym in enumerate(PERIODIC_TABLE_SYMBOLS)
            },
        }
        global_output_path = os.path.join(output_dir, "global_element_counts.json")
        with open(global_output_path, "w", encoding="utf-8") as fout:
            json.dump(global_output, fout, indent=2)
        print(f"Wrote: {global_output_path}")

    comm.Barrier()


if __name__ == "__main__":
    main()
