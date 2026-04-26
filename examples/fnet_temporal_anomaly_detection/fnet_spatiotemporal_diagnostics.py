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
fnet_spatiotemporal_diagnostics.py
==================================

Spatiotemporal sanity plots for the FNET multi-feature T-GCN example.

Loads the ``.npy`` artefacts written by ``fnet_temporal_anomaly_detection.py``
at the end of training and produces three diagnostic figures per output
feature (freq_dev, angle_delta, volt_dev):

  1. Predicted-vs-true time-series overlay for a few representative nodes,
     centered on the largest-residual window. One subplot per node, traces
     for forecast horizons h=1, h=H/2, h=H.

  2. Absolute-error heatmap over (time-window, node) for a fixed horizon.
     Highlights nodes / time slices where the model is systematically off.

  3. Error-vs-graph-distance from the event origin. The "event origin" is
     defined as the node whose ground-truth target has the largest
     instantaneous magnitude in the held-out split; graph distance is the
     unweighted shortest-path hop count on the geographic k-NN graph.

Inputs (all produced by the trainer's ``--out_dir``):
    preds_test.npy  [W, N, H, F_out]
    ys_test.npy     [W, N, H, F_out]
    A_hat.npy       [N, N]   (used only to recover the unweighted graph)
    sites.csv       (site, fdr_id, grid_name)

Usage
-----
    python fnet_spatiotemporal_diagnostics.py \\
        --in_dir outputs_fnet_temporal \\
        --out_dir outputs_fnet_temporal/diagnostics \\
        --split test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


FEATURE_NAMES = ["freq_dev", "angle_delta", "volt_dev"]
FEATURE_LABELS = {
    "freq_dev": r"$\Delta f$ (Hz)",
    "angle_delta": r"$\Delta\theta$ (rad)",
    "volt_dev": r"$\Delta V$ (pu)",
}


def _load_artifacts(in_dir: Path, split: str):
    preds = np.load(in_dir / f"preds_{split}.npy")
    ys = np.load(in_dir / f"ys_{split}.npy")
    A_hat = np.load(in_dir / "A_hat.npy")
    sites_csv = in_dir / "sites.csv"
    if sites_csv.exists():
        sites_df = pd.read_csv(sites_csv)
    else:
        N = preds.shape[1]
        sites_df = pd.DataFrame(
            {"site": [f"node_{i}" for i in range(N)], "fdr_id": np.arange(N)}
        )
    return preds, ys, A_hat, sites_df


def _graph_hop_distances(A_hat: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path hops from the GCN-normalized adjacency.

    A_hat is D^-1/2 (A+I) D^-1/2 with self-loops; an edge exists iff the
    off-diagonal entry is nonzero. Returns an int matrix of hop counts;
    unreachable pairs are encoded as -1.
    """
    n = A_hat.shape[0]
    A = A_hat.copy()
    np.fill_diagonal(A, 0.0)
    A = (np.abs(A) > 1e-12).astype(np.float64)
    dist = shortest_path(csr_matrix(A), directed=False, unweighted=True)
    dist[~np.isfinite(dist)] = -1
    return dist.astype(int)


def _select_horizons(H: int):
    if H <= 1:
        return [(0, "h=1")]
    if H == 2:
        return [(0, "h=1"), (1, "h=2")]
    return [(0, "h=1"), (H // 2, f"h={H // 2 + 1}"), (H - 1, f"h={H}")]


def plot_overlay(
    preds: np.ndarray,
    ys: np.ndarray,
    feat_idx: int,
    feat_name: str,
    sites_df: pd.DataFrame,
    out_path: Path,
    n_nodes: int = 4,
):
    """Predicted-vs-true overlay for the worst-residual window.

    For the picked window, plot per-node series across the forecast horizon
    at three checkpoints (h=1, H/2, H) plus the ground-truth horizon trace.
    """
    W, N, H, _ = preds.shape
    err = np.abs(preds[..., feat_idx] - ys[..., feat_idx])  # [W, N, H]
    # Worst window in any (node, h).
    w_star = int(np.argmax(err.reshape(W, -1).max(axis=1)))
    # Pick top-n worst nodes within that window.
    node_err = err[w_star].max(axis=1)
    node_order = np.argsort(-node_err)[: min(n_nodes, N)]

    horizons = _select_horizons(H)
    cols = len(horizons)
    fig, axes = plt.subplots(
        len(node_order),
        cols,
        figsize=(4.2 * cols, 2.6 * len(node_order)),
        sharex=False,
        squeeze=False,
    )
    horizon_axis = np.arange(1, H + 1)
    for r, ni in enumerate(node_order):
        site = sites_df["site"].iloc[ni] if ni < len(sites_df) else f"node_{ni}"
        for c, (h_idx, h_label) in enumerate(horizons):
            ax = axes[r, c]
            ax.plot(
                horizon_axis,
                ys[w_star, ni, :, feat_idx],
                "-",
                color="black",
                lw=1.6,
                label="truth",
            )
            ax.plot(
                horizon_axis,
                preds[w_star, ni, :, feat_idx],
                "--",
                color="C0",
                lw=1.4,
                label="pred",
            )
            ax.axvline(h_idx + 1, color="C3", lw=0.8, alpha=0.6, label=h_label)
            ax.set_title(f"{site}  |  worst-{h_label}", fontsize=9)
            ax.set_xlabel("horizon step")
            if c == 0:
                ax.set_ylabel(FEATURE_LABELS.get(feat_name, feat_name))
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{feat_name}: predicted vs truth, window {w_star} (max-residual window)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_heatmap(
    preds: np.ndarray,
    ys: np.ndarray,
    feat_idx: int,
    feat_name: str,
    out_path: Path,
    horizons=None,
):
    """One row of subplots: |y - y_hat| heatmap [time-window, node] per horizon.

    Three horizons (h=1, H/2, H) by default. Color = absolute error.
    """
    W, N, H, _ = preds.shape
    if horizons is None:
        horizons = _select_horizons(H)
    cols = len(horizons)
    fig, axes = plt.subplots(
        1, cols, figsize=(4.6 * cols, 4.2), sharey=True, squeeze=False
    )
    err_all = np.abs(preds[..., feat_idx] - ys[..., feat_idx])
    vmax = float(np.quantile(err_all, 0.99)) if err_all.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    for c, (h_idx, h_label) in enumerate(horizons):
        ax = axes[0, c]
        err = err_all[:, :, h_idx].T  # [N, W]
        im = ax.imshow(
            err,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{feat_name}  |  {h_label}", fontsize=10)
        ax.set_xlabel("test window index")
        if c == 0:
            ax.set_ylabel("node index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="|error|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_vs_distance(
    preds: np.ndarray,
    ys: np.ndarray,
    feat_idx: int,
    feat_name: str,
    A_hat: np.ndarray,
    out_path: Path,
):
    """Mean RMSE per hop distance from the per-window event origin.

    For each test window, the "origin" is the node with the largest
    |truth| in the horizon for the chosen feature. We then bin each
    (window, node) pair by hop distance and report mean RMSE per bin
    across the whole split, with a 1-sigma band.
    """
    W, N, H, _ = preds.shape
    hops = _graph_hop_distances(A_hat)  # [N, N], -1 for unreachable

    # Per-window origin = argmax over nodes of max |truth| across horizon.
    truth_abs_max = np.abs(ys[..., feat_idx]).max(axis=2)  # [W, N]
    origin = np.argmax(truth_abs_max, axis=1)  # [W]

    # Per (w, n) RMSE across horizon.
    sq = (preds[..., feat_idx] - ys[..., feat_idx]) ** 2  # [W, N, H]
    rmse_wn = np.sqrt(sq.mean(axis=2))  # [W, N]

    # Distance per (w, n) = hops[origin[w], n].
    dist_wn = hops[origin]  # [W, N] via fancy indexing
    valid = dist_wn >= 0

    max_hop = int(dist_wn[valid].max()) if valid.any() else 0
    bins = np.arange(0, max_hop + 1)
    means = np.zeros_like(bins, dtype=np.float64)
    stds = np.zeros_like(bins, dtype=np.float64)
    counts = np.zeros_like(bins, dtype=np.int64)
    for i, b in enumerate(bins):
        m = (dist_wn == b) & valid
        if m.any():
            vals = rmse_wn[m]
            means[i] = vals.mean()
            stds[i] = vals.std()
            counts[i] = vals.size

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(bins, means, "o-", color="C0", lw=1.6, label="mean RMSE")
    ax.fill_between(
        bins, means - stds, means + stds, color="C0", alpha=0.2, label=r"$\pm 1\sigma$"
    )
    for x, y, n in zip(bins, means, counts):
        ax.text(x, y, f"  n={n}", fontsize=7, va="center")
    ax.set_xlabel("graph distance from event origin (hops)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{feat_name}: error vs hop distance from per-window origin")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Spatiotemporal diagnostics for the FNET T-GCN example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--in_dir",
        type=str,
        default="outputs_fnet_temporal",
        help="Directory containing preds_*.npy / ys_*.npy / A_hat.npy / sites.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for generated PNGs (default: <in_dir>/diagnostics)",
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["val", "test"], help="Which split"
    )
    p.add_argument(
        "--n_overlay_nodes",
        type=int,
        default=4,
        help="How many nodes to overlay in the worst-residual window plot",
    )
    return p


def main():
    args = build_argparser().parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, ys, A_hat, sites_df = _load_artifacts(in_dir, args.split)
    if preds.ndim != 4:
        raise SystemExit(
            f"Expected preds shape [W, N, H, F_out]; got {preds.shape}. "
            f"Re-run training so that the trainer writes 4-D arrays."
        )
    W, N, H, F_out = preds.shape
    print(f"[load] split={args.split}  preds={preds.shape}  ys={ys.shape}")
    print(f"[load] A_hat={A_hat.shape}  sites={len(sites_df)}")

    if F_out != len(FEATURE_NAMES):
        print(
            f"[warn] F_out={F_out} != len(FEATURE_NAMES)={len(FEATURE_NAMES)}; "
            f"using generic feature names."
        )
        feat_names = [f"feat_{i}" for i in range(F_out)]
    else:
        feat_names = FEATURE_NAMES

    for fi, fname in enumerate(feat_names):
        print(f"[plot] {fname} ...")
        plot_overlay(
            preds,
            ys,
            fi,
            fname,
            sites_df,
            out_dir / f"{args.split}_{fname}_overlay.png",
            n_nodes=args.n_overlay_nodes,
        )
        plot_error_heatmap(
            preds, ys, fi, fname, out_dir / f"{args.split}_{fname}_heatmap.png"
        )
        plot_error_vs_distance(
            preds,
            ys,
            fi,
            fname,
            A_hat,
            out_dir / f"{args.split}_{fname}_error_vs_distance.png",
        )

    print(f"[done] figures written to {out_dir}/")


if __name__ == "__main__":
    main()
