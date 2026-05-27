#!/usr/bin/env python3
"""
detector_diagnostics.py
=======================

Reframes the saved forecast outputs as an anomaly detector and visualizes
how candidate events separate from chronic sensor faults.

Two key diagnostics:

  1. Per-window total residual norm (sum over nodes and channels). Peaks
     in this 1D signal are candidate grid events.

  2. Per-node mean residual norm (mean over windows). Bars in this 1D
     signal are chronic bad sensors.

  3. Spatial-coherence score per window: for the worst node in that window,
     ratio of its residual to the mean residual of its k-NN neighbors.
     Low ratio (<~1.5) = coherent → likely real event.
     High ratio (>~5)  = isolated → likely sensor fault.

Plus printed lists of the top-10 candidate events and top-10 chronic faults.

Usage
-----
    python detector_diagnostics.py \\
        --in_dir outputs_fnet_full_h1 \\
        --out_dir outputs_fnet_full_h1/detector \\
        --split test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_NAMES = ["freq_dev", "angle_delta", "volt_dev"]


def _load(in_dir: Path, split: str):
    preds = np.load(in_dir / f"preds_{split}.npy")  # [W, N, H, F]
    ys = np.load(in_dir / f"ys_{split}.npy")
    A_hat = np.load(in_dir / "A_hat.npy")
    feat_mean = np.load(in_dir / "feat_mean.npy")  # (N, F_dyn) or (F_dyn,)
    feat_std = np.load(in_dir / "feat_std.npy")
    out_idx = np.load(in_dir / "out_idx.npy")
    sites_df = pd.read_csv(in_dir / "sites.csv")
    # Bring back to physical units. Saved arrays are in standardized space.
    if feat_mean.ndim == 2:
        m = feat_mean[:, out_idx][None, :, None, :]
        s = feat_std[:, out_idx][None, :, None, :]
    else:
        m = feat_mean[out_idx][None, None, None, :]
        s = feat_std[out_idx][None, None, None, :]
    preds = preds * s + m
    ys = ys * s + m
    return preds, ys, A_hat, sites_df


def _neighbors(A_hat: np.ndarray) -> list:
    """Return list of arrays: neighbors[i] = indices of i's neighbors (no self)."""
    A = A_hat.copy()
    np.fill_diagonal(A, 0.0)
    return [np.where(A[i] > 0)[0] for i in range(A.shape[0])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top events / chronic faults to print",
    )
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, ys, A_hat, sites_df = _load(in_dir, args.split)
    W, N, H, F = preds.shape
    print(f"[load] preds={preds.shape} ys={ys.shape}")

    # Residual in physical units, L2 over horizons and channels per (window, node).
    resid = preds - ys  # [W, N, H, F]
    r_wn = np.sqrt(np.mean(resid**2, axis=(2, 3)))  # [W, N]

    # ---- (1) Per-window score: sum across nodes (event candidate strength) ----
    s_window = r_wn.sum(axis=1)  # [W]

    # ---- (2) Per-node score: mean across windows (chronic-fault strength) ----
    s_node = r_wn.mean(axis=0)  # [N]

    # ---- (3) Spatial coherence per window ---------------------------------
    nbrs = _neighbors(A_hat)
    coh = np.zeros(W, dtype=np.float32)
    worst_node = np.zeros(W, dtype=np.int64)
    for w in range(W):
        i = int(np.argmax(r_wn[w]))
        worst_node[w] = i
        n_i = nbrs[i]
        if len(n_i) == 0:
            coh[w] = np.inf
        else:
            mean_nb = float(r_wn[w, n_i].mean())
            coh[w] = (r_wn[w, i] / mean_nb) if mean_nb > 1e-12 else np.inf

    # ============================ Plots ============================
    # Figure 1: per-window score with coherence overlay (color-coded)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(s_window, lw=0.8, color="black")
    ax[0].set_ylabel("Σ_node ‖residual‖ (physical units)")
    ax[0].set_title(
        f"Per-window total residual ({args.split} split, W={W}, N={N})\n"
        "Peaks = candidate events"
    )
    ax[0].grid(alpha=0.3)

    # Color coherence: low (≤1.5) = green (event), high (≥5) = red (fault)
    colors = np.where(coh <= 1.5, "tab:green", np.where(coh >= 5.0, "tab:red", "gray"))
    ax[1].scatter(np.arange(W), coh, c=colors, s=5)
    ax[1].axhline(1.5, color="tab:green", ls=":", lw=0.8, label="≤1.5 likely real event")
    ax[1].axhline(5.0, color="tab:red", ls=":", lw=0.8, label="≥5 likely sensor fault")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("worst-node ÷ neighbor-mean")
    ax[1].set_xlabel(f"{args.split} window index")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.split}_per_window_score.png", dpi=120)
    plt.close(fig)

    # Figure 2: per-node mean residual (chronic faults)
    fig, ax = plt.subplots(figsize=(11, 4))
    order = np.argsort(s_node)[::-1]
    bars = ax.bar(range(N), s_node[order], color="steelblue")
    # Highlight top-k chronic faults
    for j in range(min(args.top_k, N)):
        bars[j].set_color("tab:red")
    ax.set_xticks(range(N))
    ax.set_xticklabels(
        [str(sites_df.iloc[i]["fdr_id"]) for i in order], rotation=90, fontsize=6
    )
    ax.set_ylabel("mean residual norm (physical units)")
    ax.set_title(
        f"Per-node mean residual ({args.split}); top {args.top_k} (red) = "
        "chronic faults"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.split}_per_node_score.png", dpi=120)
    plt.close(fig)

    # Figure 3: scatter of (window peak strength, coherence) — clusters reveal regimes
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(s_window, coh, c=colors, s=8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.5, color="tab:green", ls=":", lw=0.8)
    ax.axhline(5.0, color="tab:red", ls=":", lw=0.8)
    ax.set_xlabel("per-window residual sum")
    ax.set_ylabel("worst-node ÷ neighbor-mean")
    ax.set_title("Event-candidate map: bottom-right cluster = real events, top = faults")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.split}_event_map.png", dpi=120)
    plt.close(fig)

    # ============================ Reports ============================
    print(f"\n=== Top {args.top_k} candidate EVENT windows (high score, low coherence) ===")
    # Score by s_window weighted toward low coherence (multiply by 1/coh).
    event_score = s_window / np.clip(coh, 1.0, None)
    top_ev = np.argsort(event_score)[::-1][: args.top_k]
    rows_ev = []
    for w in top_ev:
        i = int(worst_node[w])
        rows_ev.append(
            {
                "window": int(w),
                "score": float(s_window[w]),
                "coherence": float(coh[w]),
                "worst_node": int(i),
                "worst_fdr_id": str(sites_df.iloc[i]["fdr_id"]),
                "worst_residual": float(r_wn[w, i]),
            }
        )
    print(pd.DataFrame(rows_ev).to_string(index=False))

    print(f"\n=== Top {args.top_k} chronic FAULT sensors (high mean residual) ===")
    rows_f = []
    for j in range(min(args.top_k, N)):
        i = int(order[j])
        rows_f.append(
            {
                "node": i,
                "fdr_id": str(sites_df.iloc[i]["fdr_id"]),
                "mean_residual": float(s_node[i]),
                "max_residual": float(r_wn[:, i].max()),
                "p99_residual": float(np.percentile(r_wn[:, i], 99)),
            }
        )
    print(pd.DataFrame(rows_f).to_string(index=False))

    # Save the per-(window, node) residual matrix for downstream analysis.
    np.save(out_dir / f"{args.split}_residual_norms.npy", r_wn)
    np.save(out_dir / f"{args.split}_coherence.npy", coh)
    print(f"\n[done] figures + arrays written to {out_dir}/")


if __name__ == "__main__":
    main()
