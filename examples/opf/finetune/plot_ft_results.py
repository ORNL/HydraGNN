"""Generate comparison plots for FT1 and FT3 fine-tuning experiments.

Reads the summary JSON produced by collect_results.py and writes a set of
publication-quality figures to results/figures/.

Figures produced
----------------
FT1 (feasibility classification):
  ft1_metrics_bar.pdf       Grouped bar chart: Accuracy, F1, AUC-ROC by regime × arch
  ft1_roc_curves.pdf        ROC curves for every run on the same axes
  ft1_learning_curves.pdf   Train/val loss curves per run (grid layout)

FT3 (N-1 contingency regression):
  ft3_mse_bar.pdf           Grouped bar chart: Va MSE, Vm MSE by regime × arch
  ft3_r2_bar.pdf            Grouped bar chart: Va R², Vm R² by regime × arch
  ft3_learning_curves.pdf   Train/val loss curves per run (grid layout)
  ft3_scatter_best.pdf      Pred vs actual scatter for best model (Va and Vm)

Usage::

    # From examples/opf/finetune/
    python plot_ft_results.py
    python plot_ft_results.py --summary results/ft1_ft3_summary.json \\
                               --out_dir results/figures
"""

import argparse
import json
import os
from itertools import groupby

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — one colour per regime + baseline
REGIME_COLORS = {
    "head_only":         "#4E79A7",
    "partial":           "#F28E2B",
    "full":              "#59A14F",
    "full_baseline":     "#E15759",   # full + no_pretrained
    "baseline":          "#E15759",
}
ARCH_HATCHES = {"HeteroSAGE": "", "HeteroHEAT": "//"}

LABEL_MAP = {
    "head_only":      "Head-only (FT)",
    "partial":        "Partial (FT)",
    "full":           "Full (FT)",
    "baseline":       "Scratch (baseline)",
    "full_baseline":  "Scratch (baseline)",
}

plt.rcParams.update({
    "figure.dpi":         150,
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "savefig.bbox":       "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_summary(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def _split(runs: list[dict], strategy_prefix: str) -> list[dict]:
    return [r for r in runs if r["meta"].get("ft_strategy", "").startswith(strategy_prefix)]


def _label(run: dict) -> str:
    meta = run["meta"]
    arch   = meta.get("arch", "?")
    regime = meta.get("regime", "full")
    pretrained = meta.get("pretrained", True)
    if not pretrained:
        return f"{arch}_baseline"
    return f"{arch}_{regime}"


def _regime_key(run: dict) -> str:
    meta = run["meta"]
    pretrained = meta.get("pretrained", True)
    regime     = meta.get("regime", "full")
    return regime if pretrained else "baseline"


def _color(run: dict) -> str:
    return REGIME_COLORS.get(_regime_key(run), "#888888")


def _hatch(run: dict) -> str:
    arch = run["meta"].get("arch", "")
    return ARCH_HATCHES.get(arch, "")


# ─────────────────────────────────────────────────────────────────────────────
# FT1 plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_ft1_metrics_bar(runs: list[dict], out_dir: str) -> None:
    """Grouped bar chart comparing Accuracy, F1, AUC-ROC for every FT1 run."""
    metrics_labels = [("accuracy", "Accuracy"), ("f1", "F1 Score"), ("auc_roc", "AUC-ROC")]
    n_metrics = len(metrics_labels)
    n_runs    = len(runs)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_key, metric_title) in zip(axes, metrics_labels):
        xs      = np.arange(n_runs)
        heights = [
            r["test_metrics"].get(metric_key) or 0.0
            for r in runs
        ]
        bar_labels = [_label(r) for r in runs]
        colors     = [_color(r) for r in runs]
        hatches    = [_hatch(r) for r in runs]

        bars = ax.bar(xs, heights, color=colors, edgecolor="black", linewidth=0.7)
        for bar, h_pat in zip(bars, hatches):
            bar.set_hatch(h_pat)

        ax.set_xticks(xs)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right")
        ax.set_title(metric_title)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")

    fig.suptitle("FT1 – Feasibility Classification: Test Metrics", fontsize=14, y=1.02)

    # Legend for colours (regime) and hatches (arch)
    legend_patches = [
        mpatches.Patch(facecolor=v, edgecolor="black", label=LABEL_MAP.get(k, k))
        for k, v in REGIME_COLORS.items()
    ]
    legend_patches += [
        mpatches.Patch(facecolor="white", hatch=h, edgecolor="black",
                       label=arch)
        for arch, h in ARCH_HATCHES.items()
    ]
    axes[-1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left",
                    borderaxespad=0, frameon=True)

    _save(fig, out_dir, "ft1_metrics_bar")


def plot_ft1_roc_curves(runs: list[dict], out_dir: str) -> None:
    """ROC curves for all FT1 runs on the same axes."""
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print("[plot] sklearn not available — skipping ROC plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    for run in runs:
        probs  = run.get("probs", [])
        labels = run.get("labels", [])
        if not probs or len(set(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc     = auc(fpr, tpr)
        lbl = _label(run)
        ax.plot(fpr, tpr, label=f"{lbl}  (AUC={roc_auc:.3f})",
                color=_color(run),
                linestyle="--" if "HeteroHEAT" in lbl else "-",
                linewidth=1.8)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("FT1 – Feasibility Classification: ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    _save(fig, out_dir, "ft1_roc_curves")


def plot_ft1_learning_curves(runs: list[dict], out_dir: str) -> None:
    _plot_learning_curves(runs, out_dir, prefix="ft1",
                          title="FT1 – Feasibility Classification: Training Curves")


# ─────────────────────────────────────────────────────────────────────────────
# FT3 plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_ft3_mse_bar(runs: list[dict], out_dir: str) -> None:
    """Grouped bar chart: Va MSE and Vm MSE for every FT3 run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    _metric_bar(axes[0], runs, "Va_mse", "Va Voltage Angle MSE")
    _metric_bar(axes[1], runs, "Vm_mse", "Vm Voltage Magnitude MSE")
    fig.suptitle("FT3 – N-1 Contingency Regression: MSE Comparison", fontsize=14, y=1.02)
    _add_legend(axes[-1])
    _save(fig, out_dir, "ft3_mse_bar")


def plot_ft3_r2_bar(runs: list[dict], out_dir: str) -> None:
    """Grouped bar chart: Va R² and Vm R² for every FT3 run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    _metric_bar(axes[0], runs, "Va_r2", "Va Voltage Angle R²", lower_better=False)
    _metric_bar(axes[1], runs, "Vm_r2", "Vm Voltage Magnitude R²", lower_better=False)
    fig.suptitle("FT3 – N-1 Contingency Regression: R² Comparison", fontsize=14, y=1.02)
    _add_legend(axes[-1])
    _save(fig, out_dir, "ft3_r2_bar")


def plot_ft3_learning_curves(runs: list[dict], out_dir: str) -> None:
    _plot_learning_curves(runs, out_dir, prefix="ft3",
                          title="FT3 – N-1 Contingency Regression: Training Curves")


def plot_ft3_scatter_best(runs: list[dict], out_dir: str) -> None:
    """Pred vs actual scatter for the best (lowest Va_mse) FT3 run."""
    valid = [r for r in runs if r["test_metrics"].get("Va_mse") is not None
             and r.get("preds_sample")]
    if not valid:
        print("[plot] No FT3 runs with scatter data — skipping scatter plot.")
        return

    best = min(valid, key=lambda r: r["test_metrics"]["Va_mse"])
    preds   = np.array(best["preds_sample"])    # [N, 2]
    targets = np.array(best["targets_sample"])

    if preds.ndim == 1 or preds.shape[1] < 2:
        print("[plot] Unexpected preds_sample shape — skipping scatter.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dim, name in zip(axes, [0, 1], ["Va (voltage angle)", "Vm (voltage magnitude)"]):
        p, t = preds[:, dim], targets[:, dim]
        ax.scatter(t, p, alpha=0.3, s=8, color="#4E79A7")
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "r--", linewidth=1, label="y = x")
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}")
        ax.legend(fontsize=9)

    fig.suptitle(f"FT3 – Best model ({_label(best)}): Pred vs Actual",
                 fontsize=13, y=1.02)
    _save(fig, out_dir, "ft3_scatter_best")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_bar(ax, runs: list[dict], key: str, title: str,
                lower_better: bool = True) -> None:
    n_runs = len(runs)
    xs     = np.arange(n_runs)
    heights = [r["test_metrics"].get(key) for r in runs]
    bar_labels = [_label(r) for r in runs]
    colors  = [_color(r) for r in runs]
    hatches = [_hatch(r) for r in runs]

    bars = ax.bar(xs, [h or 0.0 for h in heights],
                  color=colors, edgecolor="black", linewidth=0.7)
    for bar, h_pat in zip(bars, hatches):
        bar.set_hatch(h_pat)

    # Annotate missing values
    for x, h in zip(xs, heights):
        if h is None:
            ax.text(x, 0.01, "N/A", ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=30, ha="right")
    ax.set_title(title)
    if lower_better:
        ax.set_ylabel("MSE ↓")
    else:
        ax.set_ylabel("R² ↑")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")


def _add_legend(ax) -> None:
    patches = [
        mpatches.Patch(facecolor=v, edgecolor="black", label=LABEL_MAP.get(k, k))
        for k, v in REGIME_COLORS.items()
    ]
    patches += [
        mpatches.Patch(facecolor="white", hatch=h, edgecolor="black", label=arch)
        for arch, h in ARCH_HATCHES.items()
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left",
              borderaxespad=0, frameon=True)


def _plot_learning_curves(runs: list[dict], out_dir: str,
                           prefix: str, title: str) -> None:
    n = len(runs)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                              squeeze=False)

    for idx, run in enumerate(runs):
        ax    = axes[idx // ncols][idx % ncols]
        curve = run.get("training_curve", {})

        train_vals = curve.get("train error", [])
        val_vals   = curve.get("validate error", [])

        if train_vals:
            epochs = [ep for ep, _ in train_vals]
            vals   = [v  for _, v  in train_vals]
            ax.plot(epochs, vals, label="train", color="#4E79A7", linewidth=1.5)
        if val_vals:
            epochs = [ep for ep, _ in val_vals]
            vals   = [v  for _, v  in val_vals]
            ax.plot(epochs, vals, label="val", color="#F28E2B",
                    linewidth=1.5, linestyle="--")

        ax.set_title(_label(run), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_learning_curves")


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"[plot]  Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Combined FT1 vs FT3 comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_summary(ft1_runs: list[dict], ft3_runs: list[dict],
                           out_dir: str) -> None:
    """Single-page summary: FT1 F1/AUC + FT3 Va R²/Vm R² — 4 panels."""
    if not ft1_runs and not ft3_runs:
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    if ft1_runs:
        _metric_bar(axes[0], ft1_runs, "f1",      "FT1 — F1 Score")
        _metric_bar(axes[1], ft1_runs, "auc_roc", "FT1 — AUC-ROC",
                    lower_better=False)
    if ft3_runs:
        _metric_bar(axes[2], ft3_runs, "Va_r2", "FT3 — Va R²",
                    lower_better=False)
        _metric_bar(axes[3], ft3_runs, "Vm_r2", "FT3 — Vm R²",
                    lower_better=False)

    fig.suptitle("Fine-tuning vs Baseline — FT1 & FT3 Summary", fontsize=14)
    _add_legend(axes[-1])
    _save(fig, out_dir, "combined_summary")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for FT1 and FT3 experiments."
    )
    _ft_dir   = os.path.dirname(os.path.abspath(__file__))
    _res_dir  = os.path.join(_ft_dir, "results")
    parser.add_argument(
        "--summary",
        default=os.path.join(_res_dir, "ft1_ft3_summary.json"),
        help="Path to ft1_ft3_summary.json (produced by collect_results.py)",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(_res_dir, "figures"),
        help="Output directory for figures (default: results/figures/)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.summary):
        print(f"[plot] Summary file not found: {args.summary}")
        print("  Run collect_results.py first.")
        return

    all_runs = _load_summary(args.summary)
    ft1_runs = _split(all_runs, "FT1")
    ft3_runs = _split(all_runs, "FT3")

    print(f"[plot] FT1 runs: {len(ft1_runs)}   FT3 runs: {len(ft3_runs)}")

    # ── FT1 figures ────────────────────────────────────────────────────────
    if ft1_runs:
        plot_ft1_metrics_bar(ft1_runs, args.out_dir)
        plot_ft1_roc_curves(ft1_runs, args.out_dir)
        plot_ft1_learning_curves(ft1_runs, args.out_dir)

    # ── FT3 figures ────────────────────────────────────────────────────────
    if ft3_runs:
        plot_ft3_mse_bar(ft3_runs, args.out_dir)
        plot_ft3_r2_bar(ft3_runs, args.out_dir)
        plot_ft3_learning_curves(ft3_runs, args.out_dir)
        plot_ft3_scatter_best(ft3_runs, args.out_dir)

    # ── Combined summary ───────────────────────────────────────────────────
    plot_combined_summary(ft1_runs, ft3_runs, args.out_dir)

    print(f"\n[plot] All figures written to: {args.out_dir}")


if __name__ == "__main__":
    main()
