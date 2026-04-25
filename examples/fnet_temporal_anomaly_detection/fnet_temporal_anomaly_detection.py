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

Reproduces the data pre-processing, loading, and self-supervised training
pipeline of the original standalone T-GCN-FNET repository, but built on
the HydraGNN framework using TemporalGCN (and any other Temporal* backbone).

Pipeline
--------
1. Load one day of FNET parquet files from GRID-UTK-data
   (one file per device: ``[ID]-[device_name]-[date].parquet``).
2. Per-device preprocessing: extract frequency, compute residual r = f-60,
   rate-of-change roc, align all devices on a common timestamp grid,
   and drop devices that are too sparse / misaligned.
3. Estimate global disturbance onset time t_event from per-device |roc|
   peaks (15th-percentile across devices, as in the original repo).
4. Build a signal-driven correlation graph from the pre-event window
   (Pearson correlation x exp(-|lag|/lambda), k-NN sparsified).
5. Build sliding-window torch_geometric Data objects with
   ``data.x_seq = [N, lookback, F]`` and self-supervised label
   ``data.y = r(t+1)``.
6. Optionally cache pre-processed splits to disk (pickle or ADIOS) so
   training can resume without re-running steps 1-5.
7. Train HydraGNN's TemporalGCN on pre-event windows only.
8. Score the full day, fit per-node z-scores, estimate arrival times.

Usage
-----
Pre-process and cache (pickle):
    python fnet_temporal_anomaly_detection.py --preonly --format pickle \\
        --data_root ../../../GRID-UTK-data/dataset/FNETDATAforOrnl \\
        --date 2024-06-01

Pre-process and cache (ADIOS, requires mpi4py + adios2):
    python fnet_temporal_anomaly_detection.py --preonly --format adios \\
        --data_root ../../../GRID-UTK-data/dataset/FNETDATAforOrnl \\
        --date 2024-06-01

Train from cache:
    python fnet_temporal_anomaly_detection.py --format pickle \\
        --data_root ../../../GRID-UTK-data/dataset/FNETDATAforOrnl \\
        --date 2024-06-01

End-to-end (preprocess -> cache -> train) in one shot:
    python fnet_temporal_anomaly_detection.py --format pickle --do_all \\
        --data_root ../../../GRID-UTK-data/dataset/FNETDATAforOrnl \\
        --date 2024-06-01 --limit_devices 30
"""

import os
import sys
import json
import glob
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy.signal import correlate, correlation_lags
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
# 1.  FNET parquet loader  (mirrors GRID-UTK-data/read_fnet_data.py)
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_RENAMES = {
    "DateTime": "timestamp",
    "Frequency": "frequency_hz",
    "VoltageAngle": "voltage_angle_rad",
    "VoltageMagnitude": "voltage_magnitude_v",
    "ReceivedTime": "received_time",
}


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path)


def _parse_file_name(path: Path):
    """``[ID]-[device_name]-[date].parquet`` -> dict."""
    parts = path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected file name: {path.name}")
    return {
        "device_id": parts[0],
        "device_name": "-".join(parts[1:-1]),
        "date_str": parts[-1],
    }


def _read_device_file(path: Path) -> pd.DataFrame:
    meta = _parse_file_name(path)
    df = _read_parquet(path)
    # Timestamps are stored as the DataFrame index (not a column).
    if "DateTime" not in df.columns and "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "DateTime"})
    df = df.rename(columns=COLUMN_RENAMES)
    if "received_time" in df.columns:
        df = df.drop(columns=["received_time"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["frequency_hz"] = pd.to_numeric(df["frequency_hz"], errors="coerce")
    df = df.dropna(subset=["timestamp", "frequency_hz"]).sort_values("timestamp")
    df["device_id"] = meta["device_id"]
    df["device_name"] = meta["device_name"]
    return df


def preprocess_device(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual r = f - 60 Hz and rate-of-change roc."""
    df = df.copy()
    # Reference time = first sample; relative seconds.
    t0 = df["timestamp"].iloc[0]
    df["t"] = (df["timestamp"] - t0).dt.total_seconds().astype(np.float64)
    # Sampling interval (median for robustness against missing samples).
    dt_s = df["timestamp"].diff().dt.total_seconds().dropna()
    dt = float(np.median(dt_s)) if len(dt_s) else np.nan
    df.attrs["dt"] = dt
    # Residuals.
    df["r"] = (df["frequency_hz"] - 60.0).astype(np.float32)
    df["roc"] = (
        df["r"].diff().fillna(0.0) / (dt if np.isfinite(dt) and dt > 0 else 1.0)
    ).astype(np.float32)
    return df


def load_day_folder(date_dir: Path, limit: int = None, min_samples: int = 1000):
    """Load and preprocess every parquet file in a day folder.

    Returns
    -------
    sites   : list[str]               site identifiers in load order
    data    : dict[str, pd.DataFrame] preprocessed dataframes per site
    dt      : float                   median sampling interval (s)
    """
    files = sorted(date_dir.glob("*.parquet"))
    sites, data, dt_list = [], {}, []
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
        dfp = preprocess_device(raw)
        sid = f"{dfp['device_id'].iloc[0]}-{dfp['device_name'].iloc[0]}"
        sites.append(sid)
        data[sid] = dfp
        dt_list.append(dfp.attrs.get("dt", np.nan))
    if not sites:
        raise FileNotFoundError(
            f"No usable parquet files under {date_dir} (with >= {min_samples} samples each)."
        )
    dt = float(np.nanmedian(dt_list))
    return sites, data, dt


def estimate_event_time(data: dict, percentile: float = 15.0) -> float:
    """Per-device |roc| argmax -> percentile across devices."""
    times = []
    for sid, df in data.items():
        idx = df["roc"].abs().idxmax()
        times.append(float(df.loc[idx, "t"]))
    return float(np.percentile(times, percentile))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multi-device alignment + feature stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_node_features(sites, data, feature_cols=("r", "roc")):
    """Inner-merge all device timelines on timestamp; return tvec, X[T,N,F]."""
    ref = data[sites[0]][["timestamp", "t"]].rename(columns={"t": "tref"})
    M = ref
    for sid in sites:
        cols = ["timestamp"] + list(feature_cols)
        sub = data[sid][cols].copy()
        sub.columns = ["timestamp"] + [f"{sid}:{c}" for c in feature_cols]
        M = M.merge(sub, on="timestamp", how="inner")
    if len(M) == 0:
        raise RuntimeError(
            "No common timestamps across devices after inner-join. "
            "Check that the selected day has overlapping data."
        )
    tvec = M["tref"].to_numpy(dtype=np.float32)
    ordered = [f"{sid}:{c}" for sid in sites for c in feature_cols]
    X = (
        M[ordered]
        .to_numpy(dtype=np.float32)
        .reshape(len(tvec), len(sites), len(feature_cols))
    )
    return tvec, X


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Signal-driven correlation graph
# ─────────────────────────────────────────────────────────────────────────────


def build_signal_graph(
    X, tvec, t_event, pre_window=120.0, guard=10.0, k=6, lag_lambda=2.0, dt=0.1
):
    """Pearson similarity x exp(-|lag|/lambda), k-NN sparsified."""
    T, N, _ = X.shape
    t_lo = max(0.0, t_event - pre_window)
    t_hi = max(0.0, t_event - guard)
    seg = (tvec >= t_lo) & (tvec <= t_hi)
    if seg.sum() < 16:
        raise RuntimeError(
            f"Pre-event window too short: only {seg.sum()} samples in "
            f"[{t_lo:.1f}, {t_hi:.1f}] s. Try a larger --pre_window or smaller --guard."
        )
    X_pre = X[seg, :, 0]  # 'r' only
    L = X_pre.shape[0]

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        xi = X_pre[:, i] - X_pre[:, i].mean()
        for j in range(i + 1, N):
            xj = X_pre[:, j] - X_pre[:, j].mean()
            denom = (xi.std() * xj.std() + 1e-12) * L
            rho = float(np.dot(xi, xj) / denom)
            full_corr = correlate(xi, xj, mode="full")
            lags = correlation_lags(len(xi), len(xj), mode="full")
            lag_sec = float(lags[np.argmax(full_corr)] * dt)
            W[i, j] = W[j, i] = max(0.0, rho) ** 2 * np.exp(-abs(lag_sec) / lag_lambda)

    for i in range(N):
        order = np.argsort(W[i])[::-1]
        kill = np.ones(N, dtype=bool)
        kill[order[:k]] = False
        kill[i] = False
        W[i, kill] = 0.0
    W = np.maximum(W, W.T)

    A = W + np.eye(N)
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, 1e-12, None)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    src, dst = np.nonzero(W)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(W[src, dst], dtype=torch.float32)
    return edge_index, edge_weight, A_hat


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sliding-window dataset
# ─────────────────────────────────────────────────────────────────────────────


def make_dataset(X, tvec, edge_index, lookback=30, horizon=1, t_max=None):
    T, N, F = X.shape
    pos = torch.zeros(N, 3)
    batch = torch.zeros(N, dtype=torch.long)
    dataset = []
    for s in range(lookback - 1, T - horizon):
        if t_max is not None and tvec[s] > t_max:
            break
        x_seq = torch.from_numpy(X[s - lookback + 1 : s + 1]).permute(
            1, 0, 2
        )  # [N,L,F]
        x_last = torch.from_numpy(X[s])  # [N,F]
        y_next = torch.from_numpy(X[s + 1, :, 0:1])  # [N,1]
        y_loc = torch.tensor([[0, N]], dtype=torch.int64)
        dataset.append(
            Data(
                x=x_last,
                x_seq=x_seq,
                edge_index=edge_index.clone(),
                y=y_next,
                y_loc=y_loc,
                pos=pos,
                batch=batch,
                num_nodes=N,
            )
        )
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Anomaly scoring (full-day rollout)
# ─────────────────────────────────────────────────────────────────────────────


def score_signal(model, X, tvec, edge_index, lookback, device, horizon=1):
    T, N, F = X.shape
    pos = torch.zeros(N, 3, device=device)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    pred_r = np.full((T, N), np.nan, dtype=np.float32)
    err = np.full((T, N), np.nan, dtype=np.float32)
    edge_index_dev = edge_index.to(device)
    model.eval()
    with torch.no_grad():
        for s in range(lookback - 1, T - horizon):
            x_seq = (
                torch.from_numpy(X[s - lookback + 1 : s + 1])
                .permute(1, 0, 2)
                .to(device)
            )
            x_last = torch.from_numpy(X[s]).to(device)
            data = Data(
                x=x_last,
                x_seq=x_seq,
                edge_index=edge_index_dev,
                pos=pos,
                batch=batch,
                num_nodes=N,
            )
            outputs = model(data)
            r_pred = outputs[0].cpu().numpy().reshape(N)
            pred_r[s + 1] = r_pred
            err[s + 1] = np.abs(r_pred - X[s + 1, :, 0])
    return pred_r, err


def fit_z_scores(err, tvec, t_event, guard=10.0):
    baseline = tvec <= (t_event - guard)
    if baseline.sum() < 8:
        # Fallback: use first quartile of pre-event samples.
        baseline = tvec <= (tvec[0] + 0.25 * (t_event - tvec[0]))
    mu = np.nanmean(err[baseline], axis=0)
    sd = np.nanstd(err[baseline], axis=0) + 1e-9
    return (err - mu) / sd, mu, sd


def estimate_arrival_times(z, tvec, tau=3.0, persist_s=2.0, dt=0.1):
    persist_steps = max(1, int(persist_s / dt))
    T, N = z.shape
    arrivals = np.full(N, np.nan)
    for n in range(N):
        seq = np.nan_to_num(z[:, n], nan=0.0)
        above = (seq > tau).astype(np.int32)
        roll = np.convolve(above, np.ones(persist_steps, dtype=np.int32), mode="same")
        idx = np.where(roll >= persist_steps)[0]
        if len(idx) > 0:
            arrivals[n] = tvec[idx[0]]
    return arrivals


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Cache I/O (pickle / ADIOS)
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
    # Always pickle the small auxiliary metadata.
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
# 7.  Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_stage(args, cache_dir: Path):
    """Steps 1-5 + cache write."""
    # SimplePickleWriter / AdiosWriter call dist.get_rank(); init the group.
    hydragnn.utils.distributed.setup_ddp()

    print(f"[load]  reading day folder: {Path(args.data_root) / args.date}")
    sites, data, dt = load_day_folder(
        Path(args.data_root) / args.date,
        limit=args.limit_devices,
        min_samples=args.min_samples,
    )
    print(f"[load]  {len(sites)} devices loaded, dt = {dt:.4f} s")

    t_event = estimate_event_time(data)
    print(f"[event] estimated onset at t = {t_event:.2f} s")

    tvec, X = stack_node_features(sites, data, feature_cols=("r", "roc"))
    if args.stride > 1:
        tvec = tvec[:: args.stride]
        X = X[:: args.stride]
        dt = dt * args.stride
        print(f"[align] applied stride={args.stride}: dt -> {dt:.4f} s")
    T, N, F = X.shape
    print(f"[align] X shape = {(T, N, F)} after timestamp inner-join")

    edge_index, edge_weight, A_hat = build_signal_graph(
        X,
        tvec,
        t_event,
        pre_window=args.pre_window,
        guard=args.guard,
        k=args.k,
        lag_lambda=args.lag_lambda,
        dt=dt,
    )
    print(f"[graph] {N} nodes  |  {edge_index.shape[1]} directed edges  (k={args.k})")

    t_train_max = t_event - args.guard
    full_dataset = make_dataset(
        X,
        tvec,
        edge_index,
        lookback=args.lookback,
        horizon=1,
        t_max=t_train_max,
    )
    print(
        f"[ds]    {len(full_dataset)} pre-event windows  "
        f"(lookback={args.lookback}, guard={args.guard} s)"
    )
    if args.max_windows is not None and len(full_dataset) > args.max_windows:
        # Take an evenly-spaced subset to retain temporal coverage.
        idx = np.linspace(0, len(full_dataset) - 1, args.max_windows, dtype=int)
        full_dataset = [full_dataset[i] for i in idx]
        print(f"[ds]    capped to {len(full_dataset)} windows via --max_windows")
    if len(full_dataset) < 8:
        raise RuntimeError(
            "Too few pre-event windows; reduce --lookback or pick a day "
            "with later disturbance onset."
        )

    perc_train = 0.7
    train_data, val_data, test_data = hydragnn.preprocess.split_dataset(
        full_dataset, perc_train, False
    )
    print(
        f"[split] train/val/test = {len(train_data)}/{len(val_data)}/{len(test_data)}"
    )

    meta = {
        "sites": sites,
        "tvec": tvec,
        "X": X,
        "dt": float(dt),
        "t_event": float(t_event),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "A_hat": A_hat,
        "lookback": int(args.lookback),
    }
    write_cache(
        cache_dir, args.date, args.format, train_data, val_data, test_data, meta
    )


def train_stage(args, cache_dir: Path):
    """Steps 6-8: load cache, train HydraGNN, score, save outputs."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    cfg_path = Path(__file__).resolve().parent / "fnet_temporal_anomaly_detection.json"
    with open(cfg_path) as f:
        config = json.load(f)
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
        f"| sites={len(meta['sites'])}, lookback={meta['lookback']}, "
        f"t_event={meta['t_event']:.2f}s"
    )

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
        config=config["NeuralNetwork"],
        verbosity=verbosity,
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
        f"\n[train] Self-supervised pre-event training "
        f"({mpnn_label}, lookback={meta['lookback']}, "
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
    print("\n[score] Scoring full day (pre + post-event windows) ...")
    pred_r, err = score_signal(
        raw_model,
        meta["X"],
        meta["tvec"],
        meta["edge_index"],
        meta["lookback"],
        device,
    )
    z, mu, sd = fit_z_scores(err, meta["tvec"], meta["t_event"], guard=args.guard)
    arrivals = estimate_arrival_times(
        z,
        meta["tvec"],
        tau=args.tau,
        persist_s=args.persist_s,
        dt=meta["dt"],
    )
    print("\n[arrivals] Estimated disturbance arrival times (s):")
    detected = 0
    for n, t_arr in enumerate(arrivals):
        if np.isnan(t_arr):
            continue
        detected += 1
        delay = t_arr - meta["t_event"]
        if detected <= 20:
            print(
                f"  {meta['sites'][n]:40s}  t = {t_arr:8.2f} s  "
                f"(delay {delay:+.2f} s)"
            )
    print(
        f"[arrivals] {detected}/{len(arrivals)} sites exceeded "
        f"tau={args.tau} for >= {args.persist_s}s"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tvec.npy", meta["tvec"])
    np.save(out_dir / "X.npy", meta["X"])
    np.save(out_dir / "pred_r.npy", pred_r)
    np.save(out_dir / "err.npy", err)
    np.save(out_dir / "z.npy", z)
    np.save(out_dir / "arrivals.npy", arrivals)
    np.save(out_dir / "A_hat.npy", meta["A_hat"])
    pd.DataFrame(
        {
            "site": meta["sites"],
            "arrival_sec": arrivals,
            "delay_sec": arrivals - meta["t_event"],
        }
    ).to_csv(out_dir / "arrival_times.csv", index=False)
    with open(out_dir / "sites.txt", "w") as f:
        f.writelines(s + "\n" for s in meta["sites"])
    print(f"[save]  outputs written to {out_dir}/")

    tr.save(log_name)
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HydraGNN T-GCN on real FNET data (parquet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to GRID-UTK-data root containing date subfolders "
        "(required for --preonly / --do_all; ignored when "
        "training from an existing cache)",
    )
    p.add_argument(
        "--date",
        type=str,
        default="2024-06-01",
        help="Date subfolder to load (e.g. 2024-06-01)",
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
        help="Temporal subsampling stride applied after alignment "
        "(e.g. 10 turns 10 Hz data into 1 Hz)",
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
    p.add_argument(
        "--pre_window",
        type=float,
        default=120.0,
        help="Pre-event seconds used to build the correlation graph",
    )
    p.add_argument(
        "--guard",
        type=float,
        default=10.0,
        help="Exclude last N seconds before event (anti-leakage)",
    )
    p.add_argument("--k", type=int, default=6, help="k in k-NN graph sparsification")
    p.add_argument(
        "--lag_lambda", type=float, default=2.0, help="Lag-penalty decay constant (s)"
    )
    # Sequence
    p.add_argument(
        "--lookback", type=int, default=30, help="Lookback window length (steps)"
    )
    p.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Cap total pre-event windows (evenly spaced subset). "
        "Useful for fast iteration on large days.",
    )
    # Detection
    p.add_argument(
        "--tau", type=float, default=3.0, help="Z-score threshold for arrival detection"
    )
    p.add_argument(
        "--persist_s",
        type=float,
        default=2.0,
        help="Required sustained exceedance duration (s)",
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
    # Training overrides (also used as HPO knobs)
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
