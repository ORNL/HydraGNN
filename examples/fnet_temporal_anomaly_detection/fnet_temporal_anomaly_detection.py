#!/usr/bin/env python3
##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""
fnet_temporal_anomaly_detection.py
==================================

Multi-feature spatiotemporal forecasting on FNET PMU data, mirroring the
``fnet-multi-features`` branch of the standalone T-GCN-FNET repository, but
built on HydraGNN with TemporalGCN (or any other Temporal* backbone).

Pipeline
--------
1. Load one day of FNET parquet files from GRID-UTK-data
   (one file per device: ``[FDRID]-[device_name]-[date].parquet``).
2. Per-device dynamic feature extraction (4 channels):
     - freq_dev    = Frequency - 60
     - rocof       = first difference of freq_dev
     - angle_delta = first difference of unwrap(VoltageAngle)
     - volt_dev    = VoltageMagnitude - per-sensor mean
3. Multi-device timestamp inner-join -> X[T, N, F_dyn=4].
4. Geographic k-NN graph from FDRLocation.xlsx using haversine distance
   with edge weights ``exp(-d_km / sigma_km)``, symmetrized.
5. Static node features (4-dim grid embedding):
     - look up GridName per device from FDRLocation.xlsx
     - assign each unique GridName a fixed 4-dim random projection vector
       (deterministic via numpy seed)
6. Sliding-window torch_geometric Data objects:
     - x_seq = [N, Tin, F_dyn + F_static = 8]   (static features tiled in time)
     - y     = [N, H * F_out]                   (multi-step, F_out=3 per step)
7. Time-ordered 80/10/10 train/val/test split.
8. Optionally cache pre-processed splits (pickle or ADIOS).
9. Train HydraGNN's Temporal* model.
10. Score val + test windows; save predictions and ground-truth tensors.

Usage
-----
Pre-process and cache (pickle):
    python fnet_temporal_anomaly_detection.py --preonly --format pickle \\
        --data_root ../../../GRID-UTK-data/dataset/FNETDATAforOrnl \\
        --date 2024-06-01

Train from cache:
    python fnet_temporal_anomaly_detection.py --format pickle \\
        --date 2024-06-01

End-to-end (preprocess -> cache -> train) in one shot:
    python fnet_temporal_anomaly_detection.py --do_all \\
        --data_root ../../../GRID-UTK-data/dataset/FNETDATAforOrnl \\
        --date 2024-06-01 --limit_devices 30
"""

import os
import json
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)

# ADIOS is optional; only required when --format adios is selected.
try:
    from mpi4py import MPI
    import adios2  # noqa: F401  (verifies adios2 is importable)
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset

    ADIOS_AVAILABLE = True
except ImportError:
    ADIOS_AVAILABLE = False


EARTH_RADIUS_KM = 6371.0
F_DYN = 4  # freq_dev, rocof, angle_delta, volt_dev
F_OUT = 3  # freq_dev, angle_delta, volt_dev (RoCoF is input-only)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility helper
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FNET parquet loader
# ─────────────────────────────────────────────────────────────────────────────


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path)


def _parse_file_name(path: Path):
    """``[FDRID]-[device_name]-[date].parquet`` -> dict."""
    parts = path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected file name: {path.name}")
    return {
        "fdr_id": int(parts[0]),
        "device_name": "-".join(parts[1:-1]),
        "date_str": parts[-1],
    }


def _read_device_file(path: Path) -> pd.DataFrame:
    meta = _parse_file_name(path)
    df = _read_parquet(path)
    # Timestamps are stored as the DataFrame index (not a column).
    if "DateTime" not in df.columns and "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "DateTime"})
    df = df.rename(
        columns={
            "DateTime": "timestamp",
            "Frequency": "frequency_hz",
            "VoltageAngle": "voltage_angle",
            "VoltageMagnitude": "voltage_magnitude",
            "ReceivedTime": "received_time",
        }
    )
    if "received_time" in df.columns:
        df = df.drop(columns=["received_time"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("frequency_hz", "voltage_angle", "voltage_magnitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with any NaN in the feature columns *before* computing
    # finite differences / unwrap so RoCoF and Delta-theta are well defined,
    # and so the multi-device timestamp inner-join below cannot leak NaNs
    # into the loss tensor.
    df = df.dropna(
        subset=["timestamp", "frequency_hz", "voltage_angle", "voltage_magnitude"]
    ).sort_values("timestamp")
    df["fdr_id"] = meta["fdr_id"]
    df["device_name"] = meta["device_name"]
    return df


def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 dynamic features per timestamp.

    Returns columns: timestamp, freq_dev, rocof, angle_delta, volt_dev.
    """
    df = df.copy()
    freq = df["frequency_hz"].to_numpy(dtype=np.float64)
    angle = df["voltage_angle"].to_numpy(dtype=np.float64)
    volt = df["voltage_magnitude"].to_numpy(dtype=np.float64)

    freq_dev = freq - 60.0

    rocof = np.zeros_like(freq_dev)
    rocof[1:] = np.diff(freq_dev)
    if len(rocof) > 1:
        rocof[0] = rocof[1]

    angle_unwrap = np.unwrap(angle)
    angle_delta = np.zeros_like(angle_unwrap)
    angle_delta[1:] = np.diff(angle_unwrap)
    if len(angle_delta) > 1:
        angle_delta[0] = angle_delta[1]

    volt_baseline = float(np.nanmean(volt))
    volt_dev = volt - volt_baseline

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"].values,
            "freq_dev": freq_dev.astype(np.float32),
            "rocof": rocof.astype(np.float32),
            "angle_delta": angle_delta.astype(np.float32),
            "volt_dev": volt_dev.astype(np.float32),
        }
    )
    return out


def load_day_folder(date_dir: Path, limit: int = None, min_samples: int = 1000):
    """Load and preprocess every parquet file in a day folder.

    Returns
    -------
    fdr_ids : list[int]                ordered FDR ids
    sites   : list[str]                "{fdr_id}-{device_name}" labels
    data    : dict[int, pd.DataFrame]  per-device feature dataframes
    dt      : float                    median sampling interval (s)
    """
    files = sorted(date_dir.glob("*.parquet"))
    fdr_ids, sites, data, dt_list = [], [], {}, []
    for i, f in enumerate(files):
        if limit is not None and i >= limit:
            break
        try:
            raw = _read_device_file(f)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        if len(raw) < min_samples:
            print(f"[skip] {f.name}: only {len(raw)} rows")
            continue
        feats = compute_dynamic_features(raw)
        fid = int(raw["fdr_id"].iloc[0])
        site = f"{fid}-{raw['device_name'].iloc[0]}"
        fdr_ids.append(fid)
        sites.append(site)
        data[fid] = feats
        dt_s = pd.Series(feats["timestamp"]).diff().dt.total_seconds().dropna()
        if len(dt_s):
            dt_list.append(float(np.median(dt_s)))
    if not fdr_ids:
        raise FileNotFoundError(
            f"No usable parquet files under {date_dir} "
            f"(with >= {min_samples} samples each)."
        )
    dt = float(np.nanmedian(dt_list))
    return fdr_ids, sites, data, dt


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multi-device alignment + feature stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_node_features(fdr_ids, data, dt: float):
    """Inner-merge all device timelines on timestamp.

    PMU streams are not phase-locked across sensors, so positional stacking
    (row i of sensor A vs row i of sensor B) would silently desynchronize
    nodes whenever any device drops samples. We instead inner-join on the
    parsed timestamp so every column of ``X[t, :, :]`` is the same instant.

    The ``dt`` argument is unused but kept for signature symmetry with
    callers that may want to fall back to positional alignment.

    Returns
    -------
    tvec : np.ndarray  (T,)        seconds since first common timestamp
    X    : np.ndarray  (T, N, 4)   dynamic features per node per timestep
    """
    del dt  # see docstring
    feats = ("freq_dev", "rocof", "angle_delta", "volt_dev")
    M = data[fdr_ids[0]][["timestamp"]].copy()
    for fid in fdr_ids:
        sub = data[fid][["timestamp", *feats]].copy()
        sub.columns = ["timestamp"] + [f"{fid}:{c}" for c in feats]
        M = M.merge(sub, on="timestamp", how="inner")
    if len(M) == 0:
        raise RuntimeError(
            "No common timestamps across devices after inner-join. "
            "Check that the selected day has overlapping data."
        )
    M = M.sort_values("timestamp").reset_index(drop=True)
    t0 = M["timestamp"].iloc[0]
    tvec = (M["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=np.float32)
    ordered = [f"{fid}:{c}" for fid in fdr_ids for c in feats]
    X = (
        M[ordered]
        .to_numpy(dtype=np.float32)
        .reshape(len(tvec), len(fdr_ids), len(feats))
    )
    return tvec, X


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Geographic k-NN graph + static node embeddings
# ─────────────────────────────────────────────────────────────────────────────


def load_metadata(metadata_file: Path) -> pd.DataFrame:
    df = pd.read_excel(metadata_file)
    needed = {"FDRID", "GridName", "Latitude", "Longitude"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"FDRLocation.xlsx missing columns: {missing}")
    df["FDRID"] = df["FDRID"].astype(int)
    return df[["FDRID", "GridName", "Latitude", "Longitude"]].copy()


def filter_to_active(meta_df: pd.DataFrame, fdr_ids: list):
    """Restrict metadata to FDR ids present in `fdr_ids` and report unmatched."""
    matched = meta_df[meta_df["FDRID"].isin(fdr_ids)].drop_duplicates("FDRID")
    unmatched = sorted(set(fdr_ids) - set(matched["FDRID"]))
    if unmatched:
        print(
            f"[meta] WARNING: {len(unmatched)} sensor(s) have no metadata; "
            f"dropping them. First: {unmatched[:5]}"
        )
    matched = matched.set_index("FDRID")
    kept = [fid for fid in fdr_ids if fid in matched.index]
    return kept, matched.loc[kept].reset_index()


def build_geo_knn_graph(meta_active: pd.DataFrame, k: int, sigma_km: float):
    """k-NN haversine graph with weights exp(-d / sigma_km).

    Returns
    -------
    edge_index  : LongTensor [2, E]   directed (symmetric) edges
    edge_weight : FloatTensor [E]
    A_hat       : np.ndarray [N, N]   GCN-normalized adjacency (D^-0.5 (A+I) D^-0.5)
    """
    coords = meta_active[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    coords_rad = np.radians(coords)
    n = len(coords_rad)

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n), algorithm="ball_tree", metric="haversine"
    )
    nbrs.fit(coords_rad)
    dists_rad, indices = nbrs.kneighbors(coords_rad)
    dists_km = dists_rad * EARTH_RADIUS_KM

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Skip first neighbor (self).
        for j_idx in range(1, indices.shape[1]):
            j = int(indices[i, j_idx])
            d = float(dists_km[i, j_idx])
            A[i, j] = np.exp(-d / sigma_km)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)

    src, dst = np.nonzero(A)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(A[src, dst], dtype=torch.float32)

    # GCN-normalized adjacency for downstream tools that expect A_hat.
    A_self = A + np.eye(n)
    deg = A_self.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-12, None)))
    A_hat = D_inv_sqrt @ A_self @ D_inv_sqrt
    return edge_index, edge_weight, A_hat


def build_grid_embeddings(
    meta_active: pd.DataFrame, embed_dim: int = 4, seed: int = 42
):
    """Deterministic random projection per unique GridName.

    Returns
    -------
    grid_embed       : np.ndarray [N, embed_dim]   per-node static features
    grid_name_to_idx : dict[str, int]
    grid_names       : list[str]                   per-node grid names
    """
    grid_names = meta_active["GridName"].astype(str).tolist()
    unique = sorted(set(grid_names))
    rng = np.random.RandomState(seed)
    table = rng.normal(0.0, 1.0, size=(len(unique), embed_dim)).astype(np.float32)
    name_to_idx = {g: i for i, g in enumerate(unique)}
    embed = np.stack([table[name_to_idx[g]] for g in grid_names], axis=0)
    return embed, name_to_idx, grid_names


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sliding-window dataset (multi-step targets, time-ordered split)
# ─────────────────────────────────────────────────────────────────────────────


def make_window_dataset(X, grid_embed, edge_index, Tin: int, H: int):
    """Build sliding-window torch_geometric Data objects.

    Per window:
      data.x_seq : [N, Tin, F_dyn + F_static]   (static features tiled across time)
      data.y     : [N, H * F_out]               (multi-step target per node, flat)
      data.y_loc : [[0, N]]                     single output head spans all N nodes
    """
    T, N, F = X.shape
    assert F == F_DYN
    assert grid_embed.shape == (N, grid_embed.shape[1])
    F_static = grid_embed.shape[1]

    # Pre-tiled static features as a torch tensor [N, Tin, F_static].
    static_t = (
        torch.from_numpy(grid_embed).float().unsqueeze(1).expand(N, Tin, F_static)
    )
    pos = torch.zeros(N, 3)
    batch = torch.zeros(N, dtype=torch.long)
    target_dim = H * F_OUT
    # HydraGNN uses y_loc to slice y per output head; for a node-level head with
    # `output_dim` channels per node, dim_item = (y_loc[0,1] - y_loc[0,0]) / N,
    # so y_loc must span N * output_dim entries.
    y_loc = torch.tensor([[0, N * target_dim]], dtype=torch.int64)
    # Output indices: freq_dev (0), angle_delta (2), volt_dev (3)
    out_idx = np.array([0, 2, 3], dtype=np.int64)

    dataset = []
    for s in range(Tin - 1, T - H):
        # Dynamic input: shape [Tin, N, F_dyn] -> [N, Tin, F_dyn]
        x_dyn = torch.from_numpy(X[s - Tin + 1 : s + 1]).permute(1, 0, 2).contiguous()
        x_seq = torch.cat(
            [x_dyn, static_t], dim=-1
        ).contiguous()  # [N, Tin, F_dyn+F_static]

        # Multi-step target: shape [H, N, F_out] -> [N, H*F_out]
        y_block = X[s + 1 : s + 1 + H][:, :, out_idx]  # [H, N, F_out]
        y = (
            torch.from_numpy(y_block)
            .permute(1, 0, 2)  # [N, H, F_out]
            .reshape(N, H * F_OUT)
            .contiguous()
        )

        x_last = torch.from_numpy(X[s]).contiguous()  # [N, F_dyn]
        dataset.append(
            Data(
                x=x_last,
                x_seq=x_seq,
                edge_index=edge_index.clone(),
                y=y,
                y_loc=y_loc.clone(),
                pos=pos,
                batch=batch,
                num_nodes=N,
            )
        )
    return dataset


def time_ordered_split(dataset, train_frac: float, val_frac: float):
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = dataset[:n_train]
    val = dataset[n_train : n_train + n_val]
    test = dataset[n_train + n_val :]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Cache I/O (pickle / ADIOS)
# ─────────────────────────────────────────────────────────────────────────────


def _cache_basename(date: str, fmt: str) -> str:
    return f"fnet_{date}.{'pickle' if fmt == 'pickle' else 'bp'}"


def _meta_path(cache_dir: Path, date: str) -> Path:
    """Auxiliary metadata (X, tvec, sites, ...) is always pickled."""
    return cache_dir / f"fnet_{date}_meta.pkl"


def write_cache(
    cache_dir: Path, date: str, fmt: str, trainset, valset, testset, meta: dict
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle":
        basedir = str(cache_dir / _cache_basename(date, fmt))
        SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
        SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
        SimplePickleWriter(testset, basedir, "testset", use_subdir=True)
        print(f"[cache] pickle splits saved under {basedir}/")
    elif fmt == "adios":
        if not ADIOS_AVAILABLE:
            raise RuntimeError(
                "ADIOS format requested but mpi4py / adios2 are not installed."
            )
        comm = MPI.COMM_WORLD
        fname = str(cache_dir / _cache_basename(date, fmt))
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        adwriter.save()
        print(f"[cache] adios splits saved to {fname}")
    else:
        raise ValueError(f"Unknown --format: {fmt}")
    with open(_meta_path(cache_dir, date), "wb") as f:
        pickle.dump(meta, f)
    print(f"[cache] metadata saved to {_meta_path(cache_dir, date)}")


def read_cache(cache_dir: Path, date: str, fmt: str):
    meta_p = _meta_path(cache_dir, date)
    if not meta_p.exists():
        raise FileNotFoundError(
            f"Missing metadata cache: {meta_p}. Run with --preonly first."
        )
    with open(meta_p, "rb") as f:
        meta = pickle.load(f)
    if fmt == "pickle":
        basedir = str(cache_dir / _cache_basename(date, fmt))
        trainset = SimplePickleDataset(basedir=basedir, label="trainset")
        valset = SimplePickleDataset(basedir=basedir, label="valset")
        testset = SimplePickleDataset(basedir=basedir, label="testset")
    elif fmt == "adios":
        if not ADIOS_AVAILABLE:
            raise RuntimeError(
                "ADIOS format requested but mpi4py / adios2 are not installed."
            )
        comm = MPI.COMM_WORLD
        fname = str(cache_dir / _cache_basename(date, fmt))
        opt = {"preload": True, "shmem": False}
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    else:
        raise ValueError(f"Unknown --format: {fmt}")
    return trainset, valset, testset, meta


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_stage(args, cache_dir: Path):
    """Steps 1-7 + cache write."""
    # SimplePickleWriter / AdiosWriter call dist.get_rank(); init the group.
    hydragnn.utils.distributed.setup_ddp()

    data_root = Path(args.data_root)
    date_dir = data_root / args.date
    metadata_file = (
        Path(args.metadata_file)
        if args.metadata_file
        else (data_root / "FDRLocation.xlsx")
    )

    print(f"[load]  reading day folder: {date_dir}")
    fdr_ids, sites, data, dt = load_day_folder(
        date_dir, limit=args.limit_devices, min_samples=args.min_samples
    )
    print(f"[load]  {len(fdr_ids)} devices loaded, dt = {dt:.4f} s")

    print(f"[meta]  loading metadata: {metadata_file}")
    meta_df = load_metadata(metadata_file)
    fdr_ids, meta_active = filter_to_active(meta_df, fdr_ids)
    sites = [s for s in sites if int(s.split("-", 1)[0]) in set(fdr_ids)]
    if len(fdr_ids) < 2:
        raise RuntimeError(
            f"Only {len(fdr_ids)} device(s) have both data and metadata; "
            f"need >= 2 to build a graph."
        )
    print(f"[meta]  {len(fdr_ids)} devices retained after metadata filtering")

    tvec, X = stack_node_features(fdr_ids, data, dt)
    if args.stride > 1:
        tvec = tvec[:: args.stride]
        X = X[:: args.stride]
        dt = dt * args.stride
        print(f"[align] applied stride={args.stride}: dt -> {dt:.4f} s")
    T, N, F = X.shape
    print(f"[align] X shape = {(T, N, F)} (after timestamp inner-join)")

    edge_index, edge_weight, A_hat = build_geo_knn_graph(
        meta_active, k=args.k, sigma_km=args.sigma_km
    )
    print(
        f"[graph] {N} nodes  |  {edge_index.shape[1]} directed edges  "
        f"(k={args.k}, sigma_km={args.sigma_km})"
    )

    grid_embed, grid_name_to_idx, grid_names = build_grid_embeddings(
        meta_active, embed_dim=args.grid_embed_dim, seed=args.seed
    )
    print(
        f"[embed] grid embedding shape = {grid_embed.shape}, "
        f"unique grids = {len(grid_name_to_idx)}"
    )

    full_dataset = make_window_dataset(
        X, grid_embed, edge_index, Tin=args.Tin, H=args.horizon
    )
    print(
        f"[ds]    {len(full_dataset)} windows  " f"(Tin={args.Tin}, H={args.horizon})"
    )
    if args.max_windows is not None and len(full_dataset) > args.max_windows:
        idx = np.linspace(0, len(full_dataset) - 1, args.max_windows, dtype=int)
        full_dataset = [full_dataset[i] for i in idx]
        print(f"[ds]    capped to {len(full_dataset)} windows via --max_windows")
    if len(full_dataset) < 8:
        raise RuntimeError(
            "Too few windows; reduce --Tin/--horizon or use a longer day."
        )

    train_data, val_data, test_data = time_ordered_split(
        full_dataset, train_frac=args.train_frac, val_frac=args.val_frac
    )
    print(
        f"[split] train/val/test = {len(train_data)}/{len(val_data)}/{len(test_data)}  "
        f"(time-ordered)"
    )

    meta = {
        "fdr_ids": fdr_ids,
        "sites": sites,
        "tvec": tvec,
        "X": X,
        "dt": float(dt),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "A_hat": A_hat,
        "grid_embed": grid_embed,
        "grid_name_to_idx": grid_name_to_idx,
        "grid_names": grid_names,
        "Tin": int(args.Tin),
        "horizon": int(args.horizon),
        "F_dyn": F_DYN,
        "F_out": F_OUT,
        "F_static": int(grid_embed.shape[1]),
    }
    write_cache(
        cache_dir, args.date, args.format, train_data, val_data, test_data, meta
    )


def train_stage(args, cache_dir: Path):
    """Steps 8-10: load cache, train HydraGNN, score val/test, save outputs."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    cfg_path = Path(__file__).resolve().parent / "fnet_temporal_anomaly_detection.json"
    with open(cfg_path) as f:
        config = json.load(f)

    # CLI overrides (also used as HPO knobs).
    if args.mpnn_type:
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = args.mpnn_type
    if args.backbone:
        config["NeuralNetwork"]["Architecture"]["temporal_backbone"] = args.backbone
    if args.mode:
        config["NeuralNetwork"]["Architecture"]["temporal_mode"] = args.mode
    if args.hidden_dim is not None:
        config["NeuralNetwork"]["Architecture"]["hidden_dim"] = int(args.hidden_dim)
    if args.num_conv_layers is not None:
        config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = int(
            args.num_conv_layers
        )
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = int(args.num_epoch)
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = int(args.batch_size)
    if args.learning_rate is not None:
        config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"] = float(
            args.learning_rate
        )

    verbosity = config["Verbosity"]["level"]

    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    mpnn_label = config["NeuralNetwork"]["Architecture"]["mpnn_type"]
    log_name = args.log if args.log else f"fnet_temporal_{args.date}_{mpnn_label}"
    hydragnn.utils.print.print_utils.setup_log(log_name)

    print(f"[cache] reading splits ({args.format}) from {cache_dir}")
    trainset, valset, testset, meta = read_cache(cache_dir, args.date, args.format)
    print(
        f"[cache] train/val/test = {len(trainset)}/{len(valset)}/{len(testset)}  "
        f"| sites={len(meta['sites'])}, Tin={meta['Tin']}, H={meta['horizon']}, "
        f"F_dyn={meta['F_dyn']}, F_static={meta['F_static']}, F_out={meta['F_out']}"
    )

    # Patch JSON config to match cached tensor shapes.
    F_total = meta["F_dyn"] + meta["F_static"]
    target_dim = meta["horizon"] * meta["F_out"]
    config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"] = list(
        range(F_total)
    )
    config["NeuralNetwork"]["Variables_of_interest"]["output_dim"] = [target_dim]

    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=verbosity
    )
    lr = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    tr.initialize()
    tr.disable()

    print(
        f"\n[train] Multi-feature spatiotemporal forecasting "
        f"({mpnn_label}, Tin={meta['Tin']}, H={meta['horizon']}, "
        f"{config['NeuralNetwork']['Training']['num_epoch']} epochs) ..."
    )
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
    )

    raw_model = (model.module if hasattr(model, "module") else model).to(device)
    print("\n[score] Scoring val + test windows ...")
    preds_val, ys_val = _score_split(raw_model, val_loader, meta, device)
    preds_test, ys_test = _score_split(raw_model, test_loader, meta, device)
    val_mse = float(np.mean((preds_val - ys_val) ** 2))
    test_mse = float(np.mean((preds_test - ys_test) ** 2))
    print(f"[score] val MSE = {val_mse:.6e}")
    print(f"[score] test MSE = {test_mse:.6e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tvec.npy", meta["tvec"])
    np.save(out_dir / "X.npy", meta["X"])
    np.save(out_dir / "A_hat.npy", meta["A_hat"])
    np.save(out_dir / "grid_embed.npy", meta["grid_embed"])
    np.save(out_dir / "preds_val.npy", preds_val)
    np.save(out_dir / "ys_val.npy", ys_val)
    np.save(out_dir / "preds_test.npy", preds_test)
    np.save(out_dir / "ys_test.npy", ys_test)
    pd.DataFrame(
        {
            "site": meta["sites"],
            "fdr_id": meta["fdr_ids"],
            "grid_name": meta["grid_names"],
        }
    ).to_csv(out_dir / "sites.csv", index=False)
    print(f"[save]  outputs written to {out_dir}/")

    tr.save(log_name)
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def _score_split(model, loader, meta, device):
    """Run model over a dataloader and return stacked (preds, ys).

    Each window's per-node output is reshaped back to [N, H, F_out].
    Returns arrays of shape [num_windows, N, H, F_out].
    """
    N = len(meta["fdr_ids"])
    H = meta["horizon"]
    Fo = meta["F_out"]
    preds, ys = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            # Single-head per-node output -> [B*N, H*Fo]
            y_pred = outputs[0].cpu().numpy()
            y_true = data.y.cpu().numpy()
            # Reshape to [B, N, H, Fo]
            B = y_pred.shape[0] // N
            y_pred = y_pred.reshape(B, N, H, Fo)
            y_true = y_true.reshape(B, N, H, Fo)
            preds.append(y_pred)
            ys.append(y_true)
    if not preds:
        return np.zeros((0, N, H, Fo), dtype=np.float32), np.zeros(
            (0, N, H, Fo), dtype=np.float32
        )
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HydraGNN multi-feature T-GCN on real FNET data (parquet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to GRID-UTK-data root containing date subfolders "
        "(required for --preonly / --do_all)",
    )
    p.add_argument(
        "--date",
        type=str,
        default="2024-06-01",
        help="Date subfolder to load (e.g. 2024-06-01)",
    )
    p.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to FDRLocation.xlsx (default: <data_root>/FDRLocation.xlsx)",
    )
    p.add_argument(
        "--limit_devices",
        type=int,
        default=None,
        help="If set, load at most this many parquet files (smoke tests)",
    )
    p.add_argument(
        "--min_samples",
        type=int,
        default=1000,
        help="Skip devices with fewer rows than this",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal subsampling stride applied after alignment",
    )
    # Cache
    p.add_argument(
        "--format",
        type=str,
        choices=["pickle", "adios"],
        default="pickle",
        help="On-disk format for cached pre-processed splits",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to write/read cached splits "
        "(default: <example_dir>/dataset)",
    )
    p.add_argument(
        "--preonly", action="store_true", help="Pre-process + cache only; skip training"
    )
    p.add_argument(
        "--do_all",
        action="store_true",
        help="Pre-process, cache, and immediately train in one run",
    )
    # Graph
    p.add_argument("--k", type=int, default=5, help="k in geographic k-NN graph")
    p.add_argument(
        "--sigma_km",
        type=float,
        default=100.0,
        help="Haversine bandwidth (km) for edge weight exp(-d/sigma)",
    )
    p.add_argument(
        "--grid_embed_dim",
        type=int,
        default=4,
        help="Dimensionality of the per-grid static embedding",
    )
    # Sliding window
    p.add_argument(
        "--Tin", type=int, default=100, help="Input history window length (steps)"
    )
    p.add_argument(
        "--horizon", type=int, default=10, help="Forecast horizon length (steps)"
    )
    p.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Cap total windows (evenly spaced subset). Useful for fast iteration.",
    )
    # Split
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    # Model overrides (also used as HPO knobs)
    p.add_argument(
        "--mpnn_type", type=str, default=None, help="Override mpnn_type from JSON"
    )
    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override temporal_backbone (gru / lstm)",
    )
    p.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override temporal_mode (post_gcn / pre_gcn / interleaved)",
    )
    p.add_argument(
        "--hidden_dim", type=int, default=None, help="Override Architecture.hidden_dim"
    )
    p.add_argument(
        "--num_conv_layers",
        type=int,
        default=None,
        help="Override Architecture.num_conv_layers",
    )
    # Training overrides
    p.add_argument(
        "--num_epoch", type=int, default=None, help="Override Training.num_epoch"
    )
    p.add_argument(
        "--batch_size", type=int, default=None, help="Override Training.batch_size"
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override Training.Optimizer.learning_rate",
    )
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Override the log-name prefix used for outputs / writer",
    )
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs_fnet_temporal",
        help="Directory for saved numpy outputs (post-training)",
    )
    return p


def main():
    args = build_argparser().parse_args()
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else (Path(__file__).resolve().parent / "dataset")
    )

    if args.preonly and args.do_all:
        raise SystemExit("--preonly and --do_all are mutually exclusive.")

    if args.preonly or args.do_all:
        if args.data_root is None:
            raise SystemExit(
                "--data_root is required when running --preonly or --do_all"
            )
        preprocess_stage(args, cache_dir)
        if args.preonly:
            print("[done]  pre-processing finished; re-run without --preonly to train.")
            return

    train_stage(args, cache_dir)


if __name__ == "__main__":
    main()
