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
synthetic_temporal_anomaly_detection.py
=======================================

Self-supervised autoregressive training of TemporalGCN on synthetic
correlated time-series data — reproducing the T-GCN pipeline from:

    Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic
    Forecasting", IEEE TITS 2020.

within the HydraGNN framework using TemporalGCNStack.

Workflow
--------
1. Generate synthetic correlated time-series on N nodes (event at t_event).
2. Build a signal-driven correlation graph from the pre-event steady-state
   window (Pearson correlation × exponential lag-penalty, k-NN sparsified).
3. Create sliding-window Data objects (data.x_seq = [N, lookback, F]) with
   the next-step residual as the self-supervised label (data.y = [N, 1]).
4. Train TemporalGCNStack as a 1-step-ahead node forecaster using only
   pre-event windows so the model learns normal signal dynamics.
5. Score the full signal post-training: per-node forecast errors → z-scores
   relative to the pre-event baseline → estimated event-arrival times.

Usage
-----
    python synthetic_temporal_anomaly_detection.py                                  # defaults
    python synthetic_temporal_anomaly_detection.py --mpnn_type TemporalGIN          # swap backbone
    python synthetic_temporal_anomaly_detection.py --backbone lstm --mode pre_gcn   # LSTM, pre_gcn mode
    python synthetic_temporal_anomaly_detection.py --num_nodes 30 --T 5000          # larger signal
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
from scipy.signal import correlate, correlation_lags
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr


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
# 1.  Synthetic correlated time-series data
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_signal(
    N: int = 20,
    T: int = 3000,
    dt: float = 0.1,
    t_event: float = 200.0,
    noise_std: float = 0.02,
    seed: int = 42,
):
    """Generate synthetic correlated time-series data.

    Each node carries two features:
      r   — signal residual (deviation from nominal)
      roc — rate-of-change of the signal

    The pre-event portion (t < t_event) is dominated by correlated sinusoidal
    oscillations plus white noise, mimicking normal steady-state dynamics.
    At t_event a decaying step disturbance originates at node 0 and propagates
    sequentially across the graph (constant inter-node propagation delay).

    Returns
    -------
    tvec    : np.ndarray [T]        elapsed time in seconds
    X       : np.ndarray [T, N, 2]  features: (r, roc)
    t_event : float                 disturbance onset time
    """
    rng = np.random.default_rng(seed)
    tvec = np.arange(T, dtype=np.float32) * dt

    # Slight inter-node frequency variation — normal steady-state dynamics.
    base_freq = 0.08 + 0.005 * np.arange(N)
    r = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        r[:, n] = (
            0.05 * np.sin(2 * np.pi * base_freq[n] * tvec)
            + noise_std * rng.standard_normal(T).astype(np.float32)
        )

    # Post-event: decaying step disturbance travels from node 0 to node N-1.
    # Grid traversal time: 30 s  →  delay per node = 30/N seconds.
    prop_speed = N / 30.0
    for n in range(N):
        t_arr = t_event + n / prop_speed
        mask = tvec >= t_arr
        r[mask, n] += 0.5 * np.exp(-0.02 * (tvec[mask] - t_arr))

    roc = np.gradient(r, dt, axis=0).astype(np.float32)
    X = np.stack([r, roc], axis=-1)  # [T, N, 2]
    return tvec, X, t_event


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Signal-driven correlation graph
#     (mirrors build_signal_graph in the standalone gcn_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_graph(
    X: np.ndarray,
    tvec: np.ndarray,
    t_event: float,
    pre_window: float = 120.0,
    guard: float = 10.0,
    k: int = 6,
    lag_lambda: float = 2.0,
    dt: float = 0.1,
):
    """Build a k-NN Pearson-correlation graph from pre-event steady-state data.

    Edge weight:  w_{ij} = max(0, rho)^2 * exp(-|lag_sec| / lag_lambda)

    The pre-event window [t_event - pre_window, t_event - guard] is used so
    that no event-contaminated samples influence the graph topology.

    Returns
    -------
    edge_index  : LongTensor  [2, E]   directed edge list (both directions kept)
    edge_weight : FloatTensor [E]      non-negative pairwise similarity weight
    A_hat       : np.ndarray  [N, N]   D^{-1/2}(W+I)D^{-1/2} for reference
    """
    T, N, _ = X.shape
    t_lo = max(0.0, t_event - pre_window)
    t_hi = max(0.0, t_event - guard)
    seg = (tvec >= t_lo) & (tvec <= t_hi)
    X_pre = X[seg, :, 0]  # [L, N]  use the 'r' feature only for similarity
    L = X_pre.shape[0]

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        xi = X_pre[:, i] - X_pre[:, i].mean()
        for j in range(i + 1, N):
            xj = X_pre[:, j] - X_pre[:, j].mean()
            rho = float(np.dot(xi, xj) / ((xi.std() * xj.std() + 1e-12) * L))
            full_corr = correlate(xi, xj, mode="full")
            lags = correlation_lags(len(xi), len(xj), mode="full")
            lag_sec = float(lags[np.argmax(full_corr)] * dt)
            W[i, j] = W[j, i] = max(0.0, rho) ** 2 * np.exp(
                -abs(lag_sec) / lag_lambda
            )

    # k-NN sparsify: keep only the k strongest connections per row.
    for i in range(N):
        order = np.argsort(W[i])[::-1]
        kill = np.ones(N, dtype=bool)
        kill[order[:k]] = False
        kill[i] = False
        W[i, kill] = 0.0
    W = np.maximum(W, W.T)  # symmetrise again after sparsification

    # GCN normalised adjacency: A_hat = D^{-1/2}(W+I)D^{-1/2}
    A = W + np.eye(N)
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, 1e-12, None)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    src, dst = np.nonzero(W)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(W[src, dst], dtype=torch.float32)
    return edge_index, edge_weight, A_hat


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sliding-window dataset
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(
    X: np.ndarray,
    tvec: np.ndarray,
    edge_index: torch.Tensor,
    lookback: int = 30,
    horizon: int = 1,
    t_max: float = None,
):
    """Build a list of Data objects for self-supervised 1-step-ahead prediction.

    For every admissible window ending at timestep s:

        data.x      [N, F]         last timestep (required by HydraGNN Base)
        data.x_seq  [N, lookback, F]  full lookback window (activates temporal path)
        data.y      [N, 1]         next-step residual r (self-supervised label)
        data.y_loc  [1, 2]         cumulative byte offsets: head 0 spans y[0 : N]
        data.pos    [N, 3]         dummy 3-D positions (GCNConv ignores them)
        data.batch  [N]            all-zero: single graph per sample

    Parameters
    ----------
    t_max : float or None
        If set, only include windows whose last timestep t ≤ t_max.
        Set to ``t_event - guard`` to restrict training to pre-event data.
    """
    T, N, F = X.shape
    pos = torch.zeros(N, 3)
    batch = torch.zeros(N, dtype=torch.long)

    dataset = []
    for s in range(lookback - 1, T - horizon):
        if t_max is not None and tvec[s] > t_max:
            break
        x_seq = torch.from_numpy(
            X[s - lookback + 1 : s + 1]   # [lookback, N, F]
        ).permute(1, 0, 2)                 # → [N, lookback, F]
        x_last = torch.from_numpy(X[s])    # [N, F]
        y_next = torch.from_numpy(         # [N, 1]  self-supervised target
            X[s + 1, :, 0:1]
        )

        # y_loc stores cumulative element offsets into data.y per output head.
        # For 1 node-level head outputting 1 value per node: y[0..N] = head 0.
        y_loc = torch.tensor([[0, N]], dtype=torch.int64)  # [1, num_heads+1]

        data = Data(
            x=x_last,
            x_seq=x_seq,
            edge_index=edge_index.clone(),
            y=y_next,
            y_loc=y_loc,
            pos=pos,
            batch=batch,
            num_nodes=N,
        )
        dataset.append(data)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Post-training anomaly scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(
    model: torch.nn.Module,
    X: np.ndarray,
    tvec: np.ndarray,
    edge_index: torch.Tensor,
    lookback: int,
    device: str,
    horizon: int = 1,
):
    """Run the trained model over every admissible window of the full signal.

    Returns
    -------
    pred_r : np.ndarray [T, N]   predicted next-step frequency residual (NaN before first window)
    err    : np.ndarray [T, N]   absolute per-node forecast error
    """
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
                .permute(1, 0, 2)               # [N, lookback, F]
                .to(device)
            )
            x_last = torch.from_numpy(X[s]).to(device)  # [N, F]
            data = Data(
                x=x_last,
                x_seq=x_seq,
                edge_index=edge_index_dev,
                pos=pos,
                batch=batch,
                num_nodes=N,
            )
            # TemporalBase.forward returns a list of per-head output tensors.
            # For a single node head, pred[0] has shape [N, head_dim=1].
            outputs = model(data)
            r_pred = outputs[0].cpu().numpy().reshape(N)
            r_true = X[s + 1, :, 0]
            pred_r[s + 1] = r_pred
            err[s + 1] = np.abs(r_pred - r_true)

    return pred_r, err


def fit_z_scores(
    err: np.ndarray,
    tvec: np.ndarray,
    t_event: float,
    guard: float = 10.0,
):
    """Fit per-node mean+std on the pre-event baseline, return z-scores.

    Returns
    -------
    z  : np.ndarray [T, N]  standardised anomaly scores
    mu : np.ndarray [N]     per-node baseline mean
    sd : np.ndarray [N]     per-node baseline std
    """
    baseline = tvec <= (t_event - guard)
    mu = np.nanmean(err[baseline], axis=0)   # [N]
    sd = np.nanstd(err[baseline], axis=0) + 1e-9
    z = (err - mu) / sd
    return z, mu, sd


def estimate_arrival_times(
    z: np.ndarray,
    tvec: np.ndarray,
    tau: float = 3.0,
    persist_s: float = 2.0,
    dt: float = 0.1,
):
    """Estimate the first time z > tau sustained for at least persist_s seconds.

    Returns
    -------
    arrivals : np.ndarray [N]  arrival time (s) per node; NaN if not detected
    """
    persist_steps = max(1, int(persist_s / dt))
    T, N = z.shape
    arrivals = np.full(N, np.nan)
    for n in range(N):
        for t in range(T - persist_steps):
            if np.all(z[t : t + persist_steps, n] > tau):
                arrivals[n] = tvec[t]
                break
    return arrivals


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    # ── Load JSON config ──────────────────────────────────────────────────────
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "synthetic_temporal_anomaly_detection.json"
    )
    with open(cfg_path) as f:
        config = json.load(f)

    # Allow CLI overrides for rapid experimentation.
    if args.mpnn_type:
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = args.mpnn_type
    if args.backbone:
        config["NeuralNetwork"]["Architecture"]["temporal_backbone"] = args.backbone
    if args.mode:
        config["NeuralNetwork"]["Architecture"]["temporal_mode"] = args.mode

    verbosity = config["Verbosity"]["level"]

    # ── Distributed / single-process setup (falls back cleanly) ──────────────
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    mpnn_label = config["NeuralNetwork"]["Architecture"]["mpnn_type"]
    log_name = f"synthetic_temporal_anomaly_detection_{mpnn_label}"
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # ── 1. Generate synthetic signal data ────────────────────────────────────
    tvec, X, t_event = generate_synthetic_signal(
        N=args.num_nodes,
        T=args.T,
        dt=args.dt,
        t_event=args.t_event,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(
        f"[data]  X shape = {X.shape}  (T={args.T}, N={args.num_nodes}, F=2)"
        f"  t_event = {t_event:.1f} s"
    )

    # ── 2. Build correlation graph ─────────────────────────────────────────────
    edge_index, edge_weight, A_hat = build_signal_graph(
        X,
        tvec,
        t_event,
        pre_window=args.pre_window,
        guard=args.guard,
        k=args.k,
        lag_lambda=args.lag_lambda,
        dt=args.dt,
    )
    print(
        f"[graph] {args.num_nodes} nodes  |  {edge_index.shape[1]} directed edges"
        f"  (k={args.k} per node)"
    )

    # ── 3. Build pre-event training dataset ────────────────────────────────────
    # Only use windows whose last timestep ends before (t_event - guard) to
    # prevent any event-contaminated signal from leaking into training.
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
        f"[dataset] {len(full_dataset)} pre-event windows"
        f"  (lookback={args.lookback}, guard={args.guard} s)"
    )

    train_data, val_data, test_data = hydragnn.preprocess.split_dataset(
        full_dataset,
        config["NeuralNetwork"]["Training"]["perc_train"],
        False,
    )
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        train_data,
        val_data,
        test_data,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )

    # ── 4. Update config with dataset-inferred dimensions ─────────────────────
    # update_config reads the first batch to set input_dim, output_dim, etc.
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    # ── 5. Build the model ────────────────────────────────────────────────────
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

    # ── 6. Self-supervised pre-event training ─────────────────────────────────
    # The model learns to predict the next-step frequency residual from a
    # lookback window.  Labels are drawn from the same signal (self-supervised).
    # Only pre-event windows are used so the model captures normal dynamics.
    print(
        f"\n[train] Self-supervised pre-event training  "
        f"({mpnn_label}, {args.lookback}-step lookback, "
        f"{config['NeuralNetwork']['Training']['num_epoch']} epochs) …"
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

    # ── 7. Post-training anomaly scoring ─────────────────────────────────────
    # Unwrap DDP wrapper (single-process: model is already the raw module).
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = raw_model.to(device)

    print("\n[score] Scoring full signal (pre- and post-event) …")
    pred_r, err = score_signal(
        raw_model, X, tvec, edge_index, args.lookback, device
    )

    z, mu, sd = fit_z_scores(err, tvec, t_event, guard=args.guard)
    arrivals = estimate_arrival_times(
        z, tvec, tau=args.tau, persist_s=args.persist_s, dt=args.dt
    )

    print("\n[arrivals] Estimated disturbance arrival times (s):")
    for n, t_arr in enumerate(arrivals):
        if np.isnan(t_arr):
            print(f"  node {n:2d}: not detected (z never exceeded tau={args.tau})")
        else:
            delay = t_arr - t_event
            print(
                f"  node {n:2d}: t = {t_arr:7.1f} s  "
                f"(delay {delay:+.1f} s after event onset)"
            )

    # ── 8. Save outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "tvec.npy"), tvec)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "pred_r.npy"), pred_r)
    np.save(os.path.join(args.out_dir, "err.npy"), err)
    np.save(os.path.join(args.out_dir, "z.npy"), z)
    np.save(os.path.join(args.out_dir, "arrivals.npy"), arrivals)
    np.save(os.path.join(args.out_dir, "A_hat.npy"), A_hat)
    print(f"[save]  outputs written to {args.out_dir}/")

    tr.save(log_name)
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Power-grid temporal T-GCN example (self-supervised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Signal parameters
    parser.add_argument(
        "--num_nodes", type=int, default=20, help="Number of sensor nodes"
    )
    parser.add_argument(
        "--T", type=int, default=3000, help="Total time-steps"
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Sampling interval (s)"
    )
    parser.add_argument(
        "--t_event", type=float, default=200.0, help="Event onset time (s)"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.02, help="Signal noise standard deviation"
    )
    # Graph construction
    parser.add_argument(
        "--pre_window",
        type=float,
        default=120.0,
        help="Duration of pre-event window used for graph construction (s)",
    )
    parser.add_argument(
        "--guard",
        type=float,
        default=10.0,
        help="Guard band before event (s) — excluded from graph and training",
    )
    parser.add_argument(
        "--k", type=int, default=6, help="k in k-NN graph sparsification"
    )
    parser.add_argument(
        "--lag_lambda",
        type=float,
        default=2.0,
        help="Lag-penalty decay constant (s)",
    )
    # Training window
    parser.add_argument(
        "--lookback", type=int, default=30, help="Lookback window length (steps)"
    )
    # Anomaly detection thresholds
    parser.add_argument(
        "--tau",
        type=float,
        default=3.0,
        help="Z-score threshold for event detection",
    )
    parser.add_argument(
        "--persist_s",
        type=float,
        default=2.0,
        help="Minimum sustained exceedance duration for arrival detection (s)",
    )
    # Model overrides (optional)
    parser.add_argument(
        "--mpnn_type",
        type=str,
        default=None,
        help="Override mpnn_type from JSON (e.g. TemporalGIN, TemporalGCN)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override temporal_backbone (gru or lstm)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override temporal_mode (post_gcn / pre_gcn / interleaved)",
    )
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_synthetic_temporal",
        help="Directory for saved numpy outputs",
    )

    args = parser.parse_args()
    main(args)
