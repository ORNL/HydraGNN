"""make_mtl_parity_plots.py - per-QoI + combined parity plots for the MTL run.

For the chosen MTL cell (default I1_a5_MTL_full_SAGE):
  - One PNG per QoI: parity scatter with R^2/Pearson/Spearman annotated
  - One combined 3x3 grid: FIP per-grain (n=28958), FIP per-graph head, 7 aux QoIs

Output dir: results/<tag>_seed<seed>/parity_mtl/
Each per-QoI plot also shows points colored by SVE size bucket where applicable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

BUCKET_TO_LSVE = {0: 125, 1: 177, 2: 250, 3: 354, 4: 500}
BUCKET_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728", 4: "#9467bd"}


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def parity_panel(
    ax,
    y_true,
    y_pred,
    color_by=None,
    title="",
    color_map=None,
    unit="",
    show_legend=False,
    sample=None,
):
    """Draw a parity panel on the given axes with R^2/Pearson/Spearman."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if sample is not None and yt.size > sample:
        rs = np.random.RandomState(0)
        idx = rs.choice(yt.size, sample, replace=False)
        yt = yt[idx]
        yp = yp[idx]
        if color_by is not None:
            color_by = np.asarray(color_by)[idx]

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    pad = 0.05 * (hi - lo if hi > lo else max(abs(lo), 1.0))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.0)

    if color_by is not None and color_map is not None:
        for b in sorted(np.unique(color_by)):
            m = color_by == b
            ax.scatter(
                yt[m],
                yp[m],
                s=8,
                alpha=0.6,
                color=color_map[b],
                edgecolor="k" if yt.size < 200 else "none",
                linewidth=0.4 if yt.size < 200 else 0,
                label=f"$L={BUCKET_TO_LSVE.get(int(b), int(b))}\\,\\mu$m"
                if show_legend
                else None,
            )
    else:
        ax.scatter(
            yt,
            yp,
            s=8,
            alpha=0.5,
            color="#333",
            edgecolor="k" if yt.size < 200 else "none",
            linewidth=0.4 if yt.size < 200 else 0,
        )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", "box")
    r2 = r2_score(yt, yp)
    rho_p = pearsonr(yt, yp)[0] if yt.size > 1 else float("nan")
    rho_s = spearmanr(yt, yp).statistic if yt.size > 1 else float("nan")
    ax.text(
        0.04,
        0.96,
        f"$R^2 = {r2:+.3f}$\n"
        f"$\\rho_P = {rho_p:+.3f}$\n"
        f"$\\rho_S = {rho_s:+.3f}$\n"
        f"$n = {yt.size}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"true {unit}".strip())
    ax.set_ylabel(f"pred {unit}".strip())
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="I1_a5_MTL_full_SAGE")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = (
        Path(__file__).resolve().parent.parent
        / "results"
        / f"{args.tag}_seed{args.seed}"
    )
    out = base / "parity_mtl"
    out.mkdir(exist_ok=True)

    rm = json.load(open(base / "run_metrics.json"))
    test = rm["test"]

    # FIP per-grain — read from CSV (true, pred, bucket all there).
    pg_csv = base / "test_pergrain_predictions.csv"
    fip_pn_df = pd.read_csv(pg_csv)
    yt_pn = fip_pn_df["y_true_raw"].values
    yp_pn = fip_pn_df["y_pred_raw"].values
    bk_pn = fip_pn_df["size_bucket"].values

    # FIP per-graph head — from test_predictions.csv (per-graph row).
    head_csv = base / "test_predictions.csv"
    fip_pg_df = pd.read_csv(head_csv)
    yt_pg = fip_pg_df["y_true_raw"].values
    yp_pg = fip_pg_df["y_pred_raw"].values
    bk_pg = (
        fip_pg_df["size_bucket"].values if "size_bucket" in fip_pg_df.columns else None
    )

    # Aux per-graph: pred + true arrays from run_metrics.json.
    aux_data = {}  # name -> (yt, yp)
    for name, info in (test.get("aux_pergraph") or {}).items():
        aux_data[name] = (np.array(info["true"]), np.array(info["pred"]))

    # Per-graph bucket labels (50 SVEs in test) — pull from FIP head CSV
    pg_bucket = bk_pg

    # ----- per-QoI individual plots -----
    # 1. FIP per-grain
    fig, ax = plt.subplots(figsize=(6, 6))
    parity_panel(
        ax,
        yt_pn,
        yp_pn,
        color_by=bk_pn,
        color_map=BUCKET_COLORS,
        title="FIP per-grain MAX (per-grain head, n=28,958)",
        unit="(raw FS-FIP)",
        show_legend=True,
        sample=8000,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "parity_FIP_pergrain.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'parity_FIP_pergrain.png'}")

    # 2. FIP per-graph head
    fig, ax = plt.subplots(figsize=(6, 6))
    parity_panel(
        ax,
        yt_pg,
        yp_pg,
        color_by=pg_bucket,
        color_map=BUCKET_COLORS,
        title="FIP per-graph MAX (per-graph head)",
        unit="(raw FS-FIP)",
        show_legend=True,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "parity_FIP_pergraph.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'parity_FIP_pergraph.png'}")

    # 3. Each aux QoI
    QOI_LABELS = {
        "yield_stress": ("yield stress $\\sigma_y$", "(MPa-equiv)"),
        "cyc_plastic_strain": (
            "cyclic plastic strain amp ($\\Delta\\varepsilon=0.0055$)",
            "",
        ),
        "cyc_stress_amp": (
            "cyclic stress amp ($\\Delta\\varepsilon=0.0055$)",
            "(MPa-equiv)",
        ),
        "aniso_E": ("Young's modulus anisotropy index", ""),
        "aniso_G": ("shear modulus anisotropy index", ""),
        "cyc_low_stress_amp": (
            "cyclic stress amp ($\\Delta\\varepsilon=0.001$)",
            "(MPa-equiv)",
        ),
        "cyc_low_plastic_strain": (
            "cyclic plastic strain amp ($\\Delta\\varepsilon=0.001$)",
            "",
        ),
    }
    for name, (yt, yp) in aux_data.items():
        label, unit = QOI_LABELS.get(name, (name, ""))
        fig, ax = plt.subplots(figsize=(6, 6))
        parity_panel(
            ax,
            yt,
            yp,
            color_by=pg_bucket,
            color_map=BUCKET_COLORS,
            title=label,
            unit=unit,
            show_legend=True,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"parity_{name}.png", dpi=160)
        plt.close(fig)
        print(f"wrote {out / f'parity_{name}.png'}")

    # ----- combined parity grid (auto-sized) -----
    panels = [
        ("FIP per-grain MAX (n=28,958)", yt_pn, yp_pn, bk_pn, "(raw FS-FIP)", 8000),
        ("FIP per-graph MAX", yt_pg, yp_pg, pg_bucket, "(raw FS-FIP)", None),
    ]
    for name, (yt, yp) in aux_data.items():
        label, unit = QOI_LABELS.get(name, (name, ""))
        panels.append((label, yt, yp, pg_bucket, unit, None))
    # Grid sizing: 3 cols, ceil(n/3) rows, plus 1 row for legend if n<=9.
    n_panels = len(panels)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    # Reserve one extra slot at the end for the legend (always keep room).
    if n_panels >= n_rows * n_cols:
        n_rows += 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, (title, yt, yp, bk, unit, sample) in enumerate(panels):
        ax = axes[i]
        parity_panel(
            ax,
            yt,
            yp,
            color_by=bk,
            color_map=BUCKET_COLORS,
            title=title,
            unit=unit,
            show_legend=False,
            sample=sample,
        )
    # Hide unused panels
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    # Add one shared legend in lower-right unused panel
    legend_ax = axes[-1] if len(panels) < len(axes) else None
    if legend_ax is not None:
        legend_ax.axis("off")
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor=BUCKET_COLORS[b],
                edgecolor="k",
                label=f"$L_{{\\rm SVE}}={BUCKET_TO_LSVE[b]}\\,\\mu$m",
            )
            for b in sorted(BUCKET_COLORS)
        ]
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            fontsize=11,
            title="SVE size bucket",
            title_fontsize=12,
        )

    fig.suptitle(
        f"MTL parity: FIP + 7 mech QoIs predicted jointly  |  cell={args.tag}, seed={args.seed}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "parity_combined.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'parity_combined.png'}")


if __name__ == "__main__":
    main()
