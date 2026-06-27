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
F_MASK = 1  # observed-mask input channel (1 = observed, 0 = imputed/missing)


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


def stack_node_features(
    fdr_ids,
    data,
    dt: float,
    min_device_coverage: float = 0.5,
    min_step_coverage: float = 0.8,
):
    """Resample every device onto a common uniform time grid, KEEPING gaps.

    PMU streams are not phase-locked across sensors and individual streams have
    small gaps. We build a single regular grid at step ``dt`` over the common
    window and reindex each device onto it with nearest-neighbor matching within
    tolerance ``dt / 2``.

    cov80 "keep-and-mask": rather than dropping every slot where a device is
    missing (which discards exactly the disturbance-adjacent data the downstream
    detector needs), missing cells are KEPT as NaN and recorded in
    ``observed_mask``. Devices covering less than ``min_device_coverage`` of the
    grid are still dropped (a near-dead sensor would be almost entirely imputed
    and would pollute its graph neighbors). Timesteps where fewer than
    ``min_step_coverage`` of the retained nodes are present are dropped. The
    remaining NaNs are imputed downstream; ``observed_mask`` lets the loss ignore
    the imputed targets.

    Returns
    -------
    tvec          : np.ndarray  (T,)       seconds since first kept slot
    X             : np.ndarray  (T, N, 4)  dynamic features; NaN at gaps (pre-impute)
    observed_mask : np.ndarray  (T, N)     1.0 where the node was actually observed
    keep_ids      : list[int]              FDR ids that survived alignment
    """
    feats = ("freq_dev", "rocof", "angle_delta", "volt_dev")
    starts = pd.Series([data[fid]["timestamp"].iloc[0] for fid in fdr_ids])
    ends = pd.Series([data[fid]["timestamp"].iloc[-1] for fid in fdr_ids])
    # Use inner percentiles instead of strict min/max so a few late-starting
    # / early-ending sensors do not collapse the common window. Devices
    # whose span does not cover the chosen window are dropped below.
    grid_start = starts.quantile(0.9, interpolation="higher")
    grid_end = ends.quantile(0.1, interpolation="lower")
    if grid_end <= grid_start:
        # Fall back to strict intersection if percentile-based window is empty.
        grid_start = starts.max()
        grid_end = ends.min()
    if grid_end <= grid_start:
        raise RuntimeError(
            f"Empty intersection of device time ranges: "
            f"grid_start={grid_start}, grid_end={grid_end}."
        )
    freq = pd.Timedelta(int(round(float(dt) * 1e9)), unit="ns")
    grid = pd.date_range(start=grid_start, end=grid_end, freq=freq)
    if len(grid) == 0:
        raise RuntimeError("Common time grid is empty.")
    print(
        f"[align] common time window {grid_start} -> {grid_end} "
        f"({(grid_end - grid_start).total_seconds():.1f} s, {len(grid)} slots)"
    )
    tol = freq / 2

    # Reindex each device onto `grid` with nearest-neighbor within tolerance.
    keep_ids, columns, dropped = [], [], []
    for fid in fdr_ids:
        sub = (
            data[fid][["timestamp", *feats]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        re = sub.reindex(grid, method="nearest", tolerance=tol)
        coverage = re[list(feats)].notna().all(axis=1).mean()
        if coverage < min_device_coverage:
            dropped.append((fid, float(coverage)))
            continue
        re.columns = [f"{fid}:{c}" for c in feats]
        columns.append(re)
        keep_ids.append(fid)
    if not keep_ids:
        raise RuntimeError(
            f"No device covered at least {min_device_coverage:.0%} of the common "
            f"grid; lower --min_device_coverage or use a different day."
        )
    if dropped:
        print(
            f"[align] WARNING: {len(dropped)} device(s) dropped for coverage "
            f"< {min_device_coverage:.0%}. First: {[d[0] for d in dropped[:5]]}"
        )

    M = pd.concat(columns, axis=1)

    # Per-(timestep, node) observed mask: a node is "observed" at t when all of
    # its features are present after the nearest-reindex (a gap leaves all NaN).
    present = np.stack(
        [
            M[[f"{fid}:{c}" for c in feats]].notna().all(axis=1).to_numpy()
            for fid in keep_ids
        ],
        axis=1,
    )  # [n_slots, N] bool

    # Keep timesteps with enough node coverage (vs. requiring ALL nodes present).
    keep_steps = present.mean(axis=1) >= min_step_coverage
    if not keep_steps.any():
        raise RuntimeError(
            f"No timestep reached {min_step_coverage:.0%} node coverage; "
            f"lower --min_step_coverage or use a different day."
        )
    M = M[keep_steps]
    observed_mask = present[keep_steps].astype(np.float32)  # [T, N]
    print(
        f"[align] kept {int(keep_steps.sum())}/{int(keep_steps.size)} timesteps "
        f"(>= {min_step_coverage:.0%} node coverage); "
        f"observed fraction = {float(observed_mask.mean()):.3f}"
    )

    tvec = (M.index - M.index[0]).total_seconds().to_numpy(dtype=np.float32)
    ordered = [f"{fid}:{c}" for fid in keep_ids for c in feats]
    X = (
        M[ordered]
        .to_numpy(dtype=np.float32)
        .reshape(len(tvec), len(keep_ids), len(feats))
    )  # contains NaN at gaps (imputed downstream)
    return tvec, X, observed_mask, keep_ids


def robust_impute_feature_matrix(
    feature_matrix, observed_mask, train_end, short_gap_steps, medium_gap_steps
):
    """Per-sensor gap imputation: linear interpolate (short gaps) -> ffill/bfill
    (medium gaps) -> train-only median fallback.

    The fallback median is computed from observed *training* values only, so
    imputation never leaks future/val/test information into the train segment.
    Ported from the standalone T-GCN (train.py). Operates on one feature channel
    at a time: ``feature_matrix`` and ``observed_mask`` are both [T, N].
    """
    out = feature_matrix.copy()
    _, N = out.shape
    for j in range(N):
        col = pd.Series(out[:, j], dtype=float)
        train_obs = observed_mask[:train_end, j] > 0.5
        train_vals = out[:train_end, j][train_obs]
        fallback = float(np.nanmedian(train_vals)) if train_vals.size > 0 else 0.0
        col = col.interpolate(
            method="linear", limit=short_gap_steps, limit_direction="both"
        )
        col = col.ffill(limit=medium_gap_steps).bfill(limit=medium_gap_steps)
        col = col.fillna(fallback)
        out[:, j] = col.to_numpy(dtype=np.float32)
    return out


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


def make_window_dataset(
    X,
    grid_embed,
    edge_index,
    Tin: int,
    H: int,
    observed_mask=None,
    edge_weight=None,
    predict_delta: bool = False,
    max_windows: int | None = None,
):
    """Build sliding-window torch_geometric Data objects.

    Per window:
      data.x_seq : [N, Tin, F_dyn (+1 mask) + F_static]   (mask + static tiled in time)
      data.y     : [N, H * F_out]               (multi-step target per node, flat)
      data.y_loc : [[0, N]]                     single output head spans all N nodes

    If ``observed_mask`` ([T, N], 1=observed / 0=imputed) is given, it is tiled as
    an extra input channel between the dynamic and static features so the model
    can condition on which inputs are real.

    If ``predict_delta`` is True, the target is the per-step increment
    ``X[s+1+h] - X[s]`` (in the standardized feature space) rather than the
    absolute level ``X[s+1+h]``. Mathematically equivalent to wiring an
    input->output residual skip in the model. data.x still carries the last
    observed *level* x_last so scoring can recover absolute predictions.
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
    start_indices = range(Tin - 1, T - H)
    if max_windows is not None and len(start_indices) > max_windows:
        # Subsample evenly *before* materializing windows to keep memory bounded
        # for long days (full day = ~735k windows = ~190 GB of Data objects).
        start_indices = np.linspace(
            Tin - 1, T - H - 1, max_windows, dtype=np.int64
        ).tolist()
    for s in start_indices:
        # Dynamic input: shape [Tin, N, F_dyn] -> [N, Tin, F_dyn]
        x_dyn = torch.from_numpy(X[s - Tin + 1 : s + 1]).permute(1, 0, 2).contiguous()
        channels = [x_dyn]
        if observed_mask is not None:
            # Observed-mask channel (1=observed, 0=imputed/missing), placed between
            # the dynamic and static features. [Tin, N] -> [N, Tin, 1].
            mask_win = (
                torch.from_numpy(observed_mask[s - Tin + 1 : s + 1])
                .permute(1, 0)
                .unsqueeze(-1)
                .contiguous()
            )
            channels.append(mask_win)
        channels.append(static_t)
        x_seq = torch.cat(
            channels, dim=-1
        ).contiguous()  # [N, Tin, F_dyn(+1)+F_static]

        # Multi-step target: shape [H, N, F_out] -> [N, H*F_out]
        y_block = X[s + 1 : s + 1 + H][:, :, out_idx]  # [H, N, F_out]
        if predict_delta:
            # Subtract last observed level (broadcast over horizons) so the
            # model only needs to learn the increment.
            y_block = y_block - X[s : s + 1, :, out_idx]  # [H, N, F_out]
        y = (
            torch.from_numpy(y_block)
            .permute(1, 0, 2)  # [N, H, F_out]
            .reshape(N, H * F_OUT)
            .contiguous()
        )

        x_last = torch.from_numpy(X[s]).contiguous()  # [N, F_dyn]
        data = Data(
            x=x_last,
            x_seq=x_seq,
            edge_index=edge_index.clone(),
            y=y,
            y_loc=y_loc.clone(),
            pos=pos,
            batch=batch,
            num_nodes=N,
        )
        if edge_weight is not None:
            # Geographic edge weights exp(-d/sigma). First dim is E (num edges),
            # so PyG's DataLoader concatenates it edge-aligned with edge_index
            # across a batch. Consumed by GCNConv via TemporalGCN's conv string.
            data.edge_weight = edge_weight.clone()
        if observed_mask is not None:
            # Forecast-window LOSS mask, aligned element-for-element with y (same
            # permute/reshape). The per-(t, node) observed flag is broadcast over
            # the F_OUT output channels. Consumed by Base.loss(mask=...) so imputed
            # targets are excluded from the objective. NOTE: distinct from the
            # input-window mask channel baked into x_seq above (different window).
            m_block = np.repeat(
                observed_mask[s + 1 : s + 1 + H][:, :, None], F_OUT, axis=2
            )  # [H, N, F_OUT]
            y_mask = (
                torch.from_numpy(m_block)
                .permute(1, 0, 2)
                .reshape(N, H * F_OUT)
                .contiguous()
            )
            assert y_mask.shape == y.shape
            data.observed_mask = y_mask
        dataset.append(data)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Cache I/O (pickle / ADIOS)
# ─────────────────────────────────────────────────────────────────────────────


def _cache_basename(date: str, fmt: str) -> str:
    return f"fnet_{date}.{'pickle' if fmt == 'pickle' else 'bp'}"


def _meta_path(cache_dir: Path, date: str) -> Path:
    """Auxiliary metadata (X, tvec, sites, ...) is always pickled."""
    return cache_dir / f"fnet_{date}_meta.pkl"


def write_cache(
    cache_dir: Path, date: str, fmt: str, trainset, valset, testset, predset, meta: dict
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle":
        basedir = str(cache_dir / _cache_basename(date, fmt))
        SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True)
        SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
        SimplePickleWriter(testset, basedir, "testset", use_subdir=True)
        SimplePickleWriter(predset, basedir, "predset", use_subdir=True)
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
        adwriter.add("predset", predset)
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
        predset = SimplePickleDataset(basedir=basedir, label="predset")
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
        predset = AdiosDataset(fname, "predset", comm, **opt)
    else:
        raise ValueError(f"Unknown --format: {fmt}")
    return trainset, valset, testset, predset, meta


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

    tvec, X, observed_mask, kept_after_align = stack_node_features(
        fdr_ids,
        data,
        dt,
        min_device_coverage=args.min_device_coverage,
        min_step_coverage=args.min_step_coverage,
    )
    if len(kept_after_align) != len(fdr_ids):
        keep_set = set(kept_after_align)
        sites = [
            s
            for s, fid in zip(sites, fdr_ids)
            if fid in keep_set
        ]
        meta_active = meta_active[meta_active["FDRID"].isin(keep_set)].copy()
        # Reorder meta_active to match kept_after_align order.
        meta_active = (
            meta_active.set_index("FDRID").loc[kept_after_align].reset_index()
        )
        fdr_ids = kept_after_align
    if args.stride > 1:
        tvec = tvec[:: args.stride]
        X = X[:: args.stride]
        observed_mask = observed_mask[:: args.stride]
        dt = dt * args.stride
        print(f"[align] applied stride={args.stride}: dt -> {dt:.4f} s")
    T, N, F = X.shape
    print(f"[align] X shape = {(T, N, F)} (gaps kept; observed_mask attached)")

    # Impute the gaps (NaN) left by keep-and-mask alignment so the model sees a
    # dense input. Train-only median fallback avoids leakage; observed_mask still
    # marks these cells so the masked loss can ignore the imputed *targets*.
    train_end_t = max(int(T * args.train_frac), 2)
    for f in range(F):
        X[:, :, f] = robust_impute_feature_matrix(
            X[:, :, f],
            observed_mask,
            train_end_t,
            args.short_gap_steps,
            args.medium_gap_steps,
        )

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

    # Per-(node, channel) standardization computed only on the train time
    # slice. This acts as a frozen per-node bias + scale: each PMU gets its
    # own zero-point and variance for each of the 4 dynamic features, so the
    # model only has to predict residuals around the node's own baseline.
    # Shapes: feat_mean / feat_std are (N, F_dyn); broadcast over time.
    train_end_t = max(int(T * args.train_frac), 2)
    feat_mean = X[:train_end_t].mean(axis=0).astype(np.float32)  # (N, F_dyn)
    feat_std = X[:train_end_t].std(axis=0).astype(np.float32)  # (N, F_dyn)
    feat_std = np.where(feat_std < 1e-8, np.float32(1.0), feat_std)
    X = ((X - feat_mean[None, :, :]) / feat_std[None, :, :]).astype(np.float32)
    print(
        f"[scale] per-node mean shape={feat_mean.shape}  "
        f"channel-avg={np.array2string(feat_mean.mean(axis=0), precision=4)}  "
        f"channel-spread(std-of-means)={np.array2string(feat_mean.std(axis=0), precision=4)}"
    )
    print(
        f"[scale] per-node std  shape={feat_std.shape}   "
        f"channel-avg={np.array2string(feat_std.mean(axis=0), precision=4)}  "
        f"channel-spread(std-of-stds)={np.array2string(feat_std.std(axis=0), precision=4)}"
    )

    # Slice the timeline into 4 contiguous segments, THEN window each segment
    # independently. The previous order (window the whole series, then split the
    # window list by index) let overlapping sliding windows straddle the
    # train/val/test boundaries, leaking recent history across splits. Windowing
    # per-segment guarantees no window crosses a boundary.
    T = X.shape[0]
    n_train = int(T * args.train_frac)
    n_val = int(T * args.val_frac)
    n_test = int(T * args.test_frac)
    seg_bounds = [
        ("train", 0, n_train),
        ("val", n_train, n_train + n_val),
        ("test", n_train + n_val, n_train + n_val + n_test),
        ("pred", n_train + n_val + n_test, T),
    ]
    seg_data = {
        name: make_window_dataset(
            X[lo:hi],
            grid_embed,
            edge_index,
            Tin=args.Tin,
            H=args.horizon,
            observed_mask=observed_mask[lo:hi],
            edge_weight=edge_weight,
            predict_delta=bool(args.predict_delta),
            max_windows=args.max_windows,
        )
        for name, lo, hi in seg_bounds
    }
    train_data, val_data, test_data, pred_data = (
        seg_data["train"],
        seg_data["val"],
        seg_data["test"],
        seg_data["pred"],
    )
    if args.predict_delta:
        print("[ds]    target = delta from x_last (predict_delta=True)")
    print(
        f"[split] slice-then-window train/val/test/pred = "
        f"{len(train_data)}/{len(val_data)}/{len(test_data)}/{len(pred_data)}  "
        f"(Tin={args.Tin}, H={args.horizon}; no window crosses a boundary)"
    )
    if min(len(train_data), len(val_data), len(test_data)) < 1:
        raise RuntimeError(
            "A split produced 0 windows; each segment length must exceed Tin + H. "
            "Reduce --Tin/--horizon, change split fractions, or use a longer day."
        )

    meta = {
        "fdr_ids": fdr_ids,
        "sites": sites,
        "tvec": tvec,
        "X": X,
        "observed_mask": observed_mask,
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
        "F_mask": F_MASK,
        "F_out": F_OUT,
        "F_static": int(grid_embed.shape[1]),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "out_idx": np.array([0, 2, 3], dtype=np.int64),
        "predict_delta": bool(args.predict_delta),
    }
    write_cache(
        cache_dir,
        args.date,
        args.format,
        train_data,
        val_data,
        test_data,
        pred_data,
        meta,
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
    trainset, valset, testset, predset, meta = read_cache(
        cache_dir, args.date, args.format
    )
    print(
        f"[cache] train/val/test/pred = "
        f"{len(trainset)}/{len(valset)}/{len(testset)}/{len(predset)}  "
        f"| sites={len(meta['sites'])}, Tin={meta['Tin']}, H={meta['horizon']}, "
        f"F_dyn={meta['F_dyn']}, F_mask={meta.get('F_mask', 0)}, "
        f"F_static={meta['F_static']}, F_out={meta['F_out']}"
    )

    # Patch JSON config to match cached tensor shapes.
    F_total = meta["F_dyn"] + meta.get("F_mask", 0) + meta["F_static"]
    target_dim = meta["horizon"] * meta["F_out"]
    config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"] = list(
        range(F_total)
    )
    config["NeuralNetwork"]["Variables_of_interest"]["output_dim"] = [target_dim]

    bs = config["NeuralNetwork"]["Training"]["batch_size"]
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, bs
    )
    # Separate loader for the final held-out 'pred' split (scored after training).
    pred_loader, _, _ = hydragnn.preprocess.create_dataloaders(
        predset, predset, predset, bs
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
    print("\n[score] Scoring val + test + pred windows ...")
    preds_val, ys_val = _score_split(raw_model, val_loader, meta, device)
    preds_test, ys_test = _score_split(raw_model, test_loader, meta, device)
    preds_pred, ys_pred = _score_split(raw_model, pred_loader, meta, device)
    val_mse = float(np.mean((preds_val - ys_val) ** 2))
    test_mse = float(np.mean((preds_test - ys_test) ** 2))
    pred_mse = float(np.mean((preds_pred - ys_pred) ** 2)) if preds_pred.size else float("nan")
    print(f"[score] val MSE  = {val_mse:.6e}")
    print(f"[score] test MSE = {test_mse:.6e}")
    print(f"[score] pred MSE = {pred_mse:.6e}")

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
    np.save(out_dir / "preds_pred.npy", preds_pred)
    np.save(out_dir / "ys_pred.npy", ys_pred)
    if "feat_mean" in meta and "feat_std" in meta:
        np.save(out_dir / "feat_mean.npy", meta["feat_mean"])
        np.save(out_dir / "feat_std.npy", meta["feat_std"])
        np.save(out_dir / "out_idx.npy", meta["out_idx"])
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
    out_idx = meta["out_idx"]
    predict_delta = bool(meta.get("predict_delta", False))
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
            if predict_delta:
                # Re-add last observed level so saved arrays are in absolute
                # (standardized) space, matching the no-delta convention.
                x_last = data.x.cpu().numpy().reshape(B, N, -1)[:, :, out_idx]
                x_last = x_last[:, :, None, :]  # [B, N, 1, Fo]
                y_pred = y_pred + x_last
                y_true = y_true + x_last
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
    # Coverage / imputation (cov80 keep-and-mask)
    p.add_argument(
        "--min_device_coverage",
        type=float,
        default=0.5,
        help="Drop a device covering less than this fraction of the time grid "
        "(retained devices' gaps are imputed + masked)",
    )
    p.add_argument(
        "--min_step_coverage",
        type=float,
        default=0.8,
        help="Keep a timestep only if at least this fraction of nodes are observed",
    )
    p.add_argument(
        "--short_gap_steps",
        type=int,
        default=10,
        help="Max gap length (steps) filled by linear interpolation during impute",
    )
    p.add_argument(
        "--medium_gap_steps",
        type=int,
        default=300,
        help="Max gap length (steps) filled by ffill/bfill during impute",
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
        "--predict_delta",
        action="store_true",
        help="Train the model to predict the per-step delta (X[s+1+h] - X[s]) "
        "instead of the absolute level. Mathematically equivalent to an "
        "input->output residual skip. Saved preds/ys remain in level space.",
    )
    p.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Cap total windows (evenly spaced subset). Useful for fast iteration.",
    )
    # Split (4-way, time-ordered; each segment is windowed independently)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument(
        "--pred_frac",
        type=float,
        default=0.10,
        help="Final held-out segment, scored once after training",
    )
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
