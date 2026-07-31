import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mpi4py import MPI
from torch_geometric.data import Data

from hydragnn.utils.datasets.hdf5dataset import HDF5Writer

INPUT_SUFFIXES = {"pl", "ql"}
OUTPUT_SUFFIXES = {
    "pg",
    "qg",
    "vm_gen",
    "vm_bus",
    "va_bus_rad",
    "p_to",
    "p_fr",
    "q_to",
    "q_fr",
}


def _suffix(column: str) -> str:
    if ":" not in column:
        return ""
    return column.rsplit(":", 1)[1]


def select_io_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Select OPFLearn input and output columns from processed table columns."""
    input_cols = [c for c in columns if _suffix(c) in INPUT_SUFFIXES]
    output_cols = [c for c in columns if _suffix(c) in OUTPUT_SUFFIXES]

    if not input_cols:
        raise ValueError("No input columns detected (expected suffixes: :pl and :ql).")
    if not output_cols:
        raise ValueError(
            "No output columns detected (expected OPF output suffixes such as :pg/:qg/:vm_bus/:va_bus_rad)."
        )

    return sorted(input_cols), sorted(output_cols)


def _split_indices(n_rows: int, train_frac: float, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to form train/val/test splits.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)

    n_train = max(1, int(train_frac * n_rows))
    n_val = max(1, int(val_frac * n_rows))
    if n_train + n_val >= n_rows:
        n_val = max(1, n_rows - n_train - 1)
    n_test = n_rows - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _row_to_data(x_row: np.ndarray, y_row: np.ndarray, feasible_flag: float) -> Data:
    # OPFLearn tabular rows are represented as single-node graphs.
    x = torch.tensor(x_row.reshape(1, -1), dtype=torch.float32)
    y = torch.tensor(y_row.reshape(1, -1), dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1), dtype=torch.float32)

    sample = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    sample.graph_attr = torch.tensor([feasible_flag], dtype=torch.float32)
    return sample


def _build_split_dataset(
    x_values: np.ndarray,
    y_values: np.ndarray,
    split_idx: np.ndarray,
    feasible_flag: float,
    rank: int,
    world_size: int,
) -> list[Data]:
    local_idx = split_idx[rank::world_size]
    dataset: list[Data] = []
    for i in local_idx:
        dataset.append(_row_to_data(x_values[i], y_values[i], feasible_flag=feasible_flag))
    return dataset


def serialize_parquet_to_hdf5(
    input_parquet: Path,
    output_hdf5_dir: Path,
    max_samples: int | None,
    include_duals: bool,
    train_frac: float,
    val_frac: float,
    seed: int,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, int | str]:
    """Convert one processed OPFLearn parquet file to HydraGNN HDF5 PyG dataset.

    The output is a directory containing HDF5 shards compatible with HDF5Dataset.
    """
    del include_duals  # reserved for future extension

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    frame = pd.read_parquet(input_parquet)
    n_total = int(frame.shape[0])
    if max_samples is not None and max_samples > 0:
        frame = frame.iloc[: max_samples].copy()

    input_cols, output_cols = select_io_columns(list(frame.columns))
    x_values = frame[input_cols].to_numpy(dtype=np.float32)
    y_values = frame[output_cols].to_numpy(dtype=np.float32)

    n_rows = int(frame.shape[0])
    train_idx, val_idx, test_idx = _split_indices(n_rows, train_frac, val_frac, seed)

    feasible_flag = 0.0 if input_parquet.name.startswith("INFEASIBLE_") else 1.0

    trainset = _build_split_dataset(x_values, y_values, train_idx, feasible_flag, rank, world_size)
    valset = _build_split_dataset(x_values, y_values, val_idx, feasible_flag, rank, world_size)
    testset = _build_split_dataset(x_values, y_values, test_idx, feasible_flag, rank, world_size)

    if rank == 0 and output_hdf5_dir.exists() and overwrite:
        if output_hdf5_dir.is_dir():
            shutil.rmtree(output_hdf5_dir, ignore_errors=True)
        else:
            output_hdf5_dir.unlink()
    comm.Barrier()

    if output_hdf5_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_hdf5_dir}. Use --overwrite.")

    writer = HDF5Writer(str(output_hdf5_dir), comm)
    writer.add("trainset", trainset)
    writer.add("valset", valset)
    writer.add("testset", testset)
    writer.save()

    logger.info(
        "Serialized %s -> %s (rows=%d/%d, in_dim=%d, out_dim=%d)",
        input_parquet,
        output_hdf5_dir,
        n_rows,
        n_total,
        len(input_cols),
        len(output_cols),
    )

    return {
        "input": str(input_parquet),
        "output": str(output_hdf5_dir),
        "rows": n_rows,
        "input_dim": len(input_cols),
        "output_dim": len(output_cols),
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "test_rows": int(test_idx.size),
    }


def serialize_directory(
    input_dir: Path,
    output_dir: Path,
    include_infeasible: bool,
    max_samples: int | None,
    include_duals: bool,
    train_frac: float,
    val_frac: float,
    seed: int,
    overwrite: bool,
    logger: logging.Logger,
) -> list[dict[str, int | str]]:
    """Serialize all OPFLearn parquet files from directory to HDF5 PyG datasets."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not include_infeasible:
        parquet_files = [p for p in parquet_files if not p.name.startswith("INFEASIBLE_")]

    if not parquet_files:
        raise FileNotFoundError("No matching parquet files found to serialize.")

    results: list[dict[str, int | str]] = []
    for parquet_path in parquet_files:
        output_name = parquet_path.stem + ".h5"
        out_path = output_dir / output_name
        results.append(
            serialize_parquet_to_hdf5(
                input_parquet=parquet_path,
                output_hdf5_dir=out_path,
                max_samples=max_samples,
                include_duals=include_duals,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=seed,
                overwrite=overwrite,
                logger=logger,
            )
        )
    return results
