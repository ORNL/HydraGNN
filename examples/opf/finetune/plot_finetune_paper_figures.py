"""
Generate all fine-tuning paper figures.

Figures produced (saved to figures/results/ in the LaTeX project):
  fig09_ft1_sample_efficiency.pdf   FT1 accuracy/F1/AUC vs n  (2-arch × 3-metric)
  fig10_ft1_convergence.pdf         FT1 val-loss vs epoch for HeteroSAGE at 4 n values
  fig11_ft3_sample_efficiency.pdf   FT3 Va_r2 / Vm_r2 vs n   (2-arch × 2-metric)
  fig12_ft3_convergence.pdf         FT3 val-loss vs epoch for HeteroSAGE at 4 n values

Usage (from examples/opf/finetune/):
    python3.11 plot_finetune_paper_figures.py
"""

import os, re, json, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── output directory ─────────────────────────────────────────────────────────
LATEX_DIR = "/lustre/orion/lrn078/proj-shared/hydragnn_opf_tsg_project_with_layer_references"
OUT_DIR   = os.path.join(LATEX_DIR, "figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

LOGS_DIR  = "logs"

# ── style ────────────────────────────────────────────────────────────────────
REGIME_ORDER  = ["head_only", "partial", "full", "scratch"]
REGIME_LABELS = {
    "head_only": "Head-only FT",
    "partial":   "Partial FT",
    "full":      "Full FT",
    "scratch":   "Scratch",
}
REGIME_COLORS = {
    "head_only": "#4E79A7",   # blue
    "partial":   "#F28E2B",   # orange
    "full":      "#59A14F",   # green
    "scratch":   "#E15759",   # red
}
REGIME_MARKERS = {
    "head_only": "o",
    "partial":   "s",
    "full":      "^",
    "scratch":   "D",
}
REGIME_DASHES = {
    "head_only": (None, None),    # solid
    "partial":   (4, 2),
    "full":      (None, None),
    "scratch":   (2, 2),
}
ARCH_ORDER  = ["HeteroSAGE", "HeteroHEAT"]
ARCH_LABELS = {"HeteroSAGE": "HeteroSAGE", "HeteroHEAT": "HeteroHEAT"}

plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "lines.linewidth":  1.6,
    "lines.markersize": 5,
    "savefig.bbox":     "tight",
    "savefig.dpi":      300,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_ft1_dirname(name):
    """Return (arch, regime, n) or None."""
    m = re.fullmatch(
        r"FT1_feasibility_(HeteroSAGE|HeteroHEAT)_"
        r"(full|partial|head_only|full_scratch)_n(\d+)", name)
    if not m:
        return None
    arch, regime_raw, n = m.group(1), m.group(2), int(m.group(3))
    regime = "scratch" if regime_raw == "full_scratch" else regime_raw
    return arch, regime, n


def _parse_ft3_dirname(name):
    """Return (arch, regime, n) or None."""
    m = re.fullmatch(
        r"finetune_FT3_contingency_(HeteroSAGE|HeteroHEAT)_"
        r"(full|partial|head_only|full_scratch)_n(\d+)", name)
    if not m:
        return None
    arch, regime_raw, n = m.group(1), m.group(2), int(m.group(3))
    regime = "scratch" if regime_raw == "full_scratch" else regime_raw
    return arch, regime, n


def _load_results(log_dir):
    """Load results.json; return (meta, test_metrics) or (None, None)."""
    path = os.path.join(log_dir, "results.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    return d.get("meta", {}), d.get("test_metrics", {})


def _load_curve(log_dir, tag="validate error"):
    """Load training_curve.csv; return (epochs, values) arrays or (None, None)."""
    path = os.path.join(log_dir, "training_curve.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    sub = df[df["tag"] == tag].copy()
    if sub.empty:
        return None, None
    sub = sub.sort_values("step")
    return sub["step"].values, sub["value"].values


def _collect_ft1():
    """Collect all FT1 final metrics. Returns DataFrame."""
    rows = []
    for name in sorted(os.listdir(LOGS_DIR)):
        parsed = _parse_ft1_dirname(name)
        if parsed is None:
            continue
        arch, regime, n = parsed
        log_dir = os.path.join(LOGS_DIR, name)
        meta, tm = _load_results(log_dir)
        if tm is None:
            continue
        rows.append({
            "arch": arch, "regime": regime, "n": n,
            "accuracy": tm.get("accuracy", np.nan),
            "f1":       tm.get("f1", np.nan),
            "auc_roc":  tm.get("auc_roc", np.nan),
            "bce":      tm.get("bce", np.nan),
            "log_dir":  log_dir,
        })
    return pd.DataFrame(rows)


def _collect_ft3():
    """Collect all FT3 final metrics. Returns DataFrame."""
    rows = []
    for name in sorted(os.listdir(LOGS_DIR)):
        parsed = _parse_ft3_dirname(name)
        if parsed is None:
            continue
        arch, regime, n = parsed
        log_dir = os.path.join(LOGS_DIR, name)
        meta, tm = _load_results(log_dir)
        if tm is None:
            continue
        rows.append({
            "arch": arch, "regime": regime, "n": n,
            "Va_mse": tm.get("Va_mse", np.nan),
            "Va_mae": tm.get("Va_mae", np.nan),
            "Va_r2":  tm.get("Va_r2",  np.nan),
            "Vm_mse": tm.get("Vm_mse", np.nan),
            "Vm_mae": tm.get("Vm_mae", np.nan),
            "Vm_r2":  tm.get("Vm_r2",  np.nan),
            "log_dir": log_dir,
        })
    return pd.DataFrame(rows)


def _regime_line(ax, df, metric, n_col="n", regimes=None, ylog=False):
    """Plot one line per regime on ax."""
    if regimes is None:
        regimes = REGIME_ORDER
    for regime in regimes:
        sub = df[df["regime"] == regime].sort_values(n_col)
        if sub.empty:
            continue
        vals = sub[metric].values
        ns   = sub[n_col].values
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            continue
        dashes = REGIME_DASHES[regime]
        ls = "--" if dashes[0] else "-"
        ax.plot(ns[mask], vals[mask],
                marker=REGIME_MARKERS[regime],
                color=REGIME_COLORS[regime],
                linestyle=ls,
                label=REGIME_LABELS[regime],
                zorder=3)
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)


# ── Fig 09 — FT1 sample efficiency ───────────────────────────────────────────

def plot_ft1_sample_efficiency(df):
    metrics = [
        ("accuracy", "Accuracy",  False, (0, 1.05)),
        ("f1",       "F1 score",  False, (0, 1.05)),
        ("auc_roc",  "AUC-ROC",   False, (0, 1.05)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for r_idx, arch in enumerate(ARCH_ORDER):
        sub_arch = df[df["arch"] == arch]
        for c_idx, (met, label, ylog, ylim) in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            _regime_line(ax, sub_arch, met, ylog=ylog)
            ax.set_ylim(ylim)
            if r_idx == 1:
                ax.set_xlabel("Training samples $n$")
            if c_idx == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel(label)
            ax.set_title(f"{ARCH_LABELS[arch]}", fontweight="bold")
            if r_idx == 0 and c_idx == 2:
                ax.legend(loc="lower right", framealpha=0.85)

    # column titles
    for c_idx, (_, label, _, _) in enumerate(metrics):
        axes[0, c_idx].set_title(
            f"{label}\n{ARCH_LABELS[ARCH_ORDER[0]]}", fontweight="bold")
        axes[1, c_idx].set_title(
            f"{ARCH_LABELS[ARCH_ORDER[1]]}", fontweight="bold")

    fig.suptitle(
        "FT1 — Feasibility Classification: Sample Efficiency",
        fontsize=10, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "fig09_ft1_sample_efficiency.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 10 — FT1 convergence (HeteroSAGE) ────────────────────────────────────

def plot_ft1_convergence(df_ft1):
    arch = "HeteroSAGE"
    n_vals = [100, 1000, 10000, 50000]
    n_labels = {100: "$n=100$", 1000: "$n=1{,}000$",
                10000: "$n=10{,}000$", 50000: "$n=50{,}000$"}

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=False)
    fig.subplots_adjust(wspace=0.38)

    for col, n in enumerate(n_vals):
        ax = axes[col]
        sub = df_ft1[(df_ft1["arch"] == arch) & (df_ft1["n"] == n)]
        plotted = False
        for regime in REGIME_ORDER:
            row = sub[sub["regime"] == regime]
            if row.empty:
                continue
            log_dir = row.iloc[0]["log_dir"]
            epochs, vals = _load_curve(log_dir, tag="validate error")
            if epochs is None:
                continue
            dashes = REGIME_DASHES[regime]
            ls = "--" if dashes[0] else "-"
            ax.plot(epochs + 1, vals,
                    color=REGIME_COLORS[regime],
                    linestyle=ls,
                    label=REGIME_LABELS[regime],
                    zorder=3)
            plotted = True

        ax.set_title(n_labels[n], fontsize=9)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Val. BCE loss")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle=":", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        if col == 3 and plotted:
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "FT1 — Training Convergence (HeteroSAGE, validation loss)",
        fontsize=10, fontweight="bold", y=1.03)

    path = os.path.join(OUT_DIR, "fig10_ft1_convergence.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 11 — FT3 sample efficiency ───────────────────────────────────────────

def plot_ft3_sample_efficiency(df):
    metrics = [
        ("Va_r2",  r"$R^2$ (voltage angle $V_a$)",  False, (-0.1, 1.05)),
        ("Vm_r2",  r"$R^2$ (voltage magnitude $V_m$)", False, (-0.1, 1.05)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.5), sharex=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.40)

    for r_idx, arch in enumerate(ARCH_ORDER):
        sub_arch = df[df["arch"] == arch]
        for c_idx, (met, label, ylog, ylim) in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            _regime_line(ax, sub_arch, met, ylog=ylog)
            ax.set_ylim(ylim)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=0)
            ax.axhline(1, color="gray", linewidth=0.8, linestyle=":", zorder=0)
            if r_idx == 1:
                ax.set_xlabel("Training samples $n$")
            ax.set_ylabel(label)
            title_str = label.split("(")[0].strip() + f"\n{ARCH_LABELS[arch]}"
            ax.set_title(title_str, fontweight="bold")
            if r_idx == 1 and c_idx == 1:
                ax.legend(loc="lower right", framealpha=0.85)

    fig.suptitle(
        r"FT3 — N-1 Contingency OPF Regression: Sample Efficiency ($R^2$)",
        fontsize=10, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "fig11_ft3_sample_efficiency.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 11b — FT3 sample efficiency MSE (supplemental) ───────────────────────

def plot_ft3_sample_efficiency_mse(df):
    metrics = [
        ("Va_mse", r"MSE (voltage angle $V_a$)",       True),
        ("Vm_mse", r"MSE (voltage magnitude $V_m$)",   True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.5), sharex=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.40)

    for r_idx, arch in enumerate(ARCH_ORDER):
        sub_arch = df[df["arch"] == arch]
        for c_idx, (met, label, ylog) in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            _regime_line(ax, sub_arch, met, ylog=ylog)
            if r_idx == 1:
                ax.set_xlabel("Training samples $n$")
            ax.set_ylabel(label)
            title_str = label.split("(")[0].strip() + f"\n{ARCH_LABELS[arch]}"
            ax.set_title(title_str, fontweight="bold")
            if r_idx == 1 and c_idx == 1:
                ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle(
        r"FT3 — N-1 Contingency OPF Regression: Sample Efficiency (MSE)",
        fontsize=10, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "fig11b_ft3_sample_efficiency_mse.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 12 — FT3 convergence (HeteroSAGE) ────────────────────────────────────

def plot_ft3_convergence(df_ft3):
    arch = "HeteroSAGE"
    n_vals  = [100, 1000, 10000, 50000]
    n_labels = {100: "$n=100$", 1000: "$n=1{,}000$",
                10000: "$n=10{,}000$", 50000: "$n=50{,}000$"}

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=False)
    fig.subplots_adjust(wspace=0.38)

    for col, n in enumerate(n_vals):
        ax = axes[col]
        sub = df_ft3[(df_ft3["arch"] == arch) & (df_ft3["n"] == n)]
        plotted = False
        for regime in REGIME_ORDER:
            row = sub[sub["regime"] == regime]
            if row.empty:
                continue
            log_dir = row.iloc[0]["log_dir"]
            epochs, vals = _load_curve(log_dir, tag="validate error")
            if epochs is None:
                continue
            dashes = REGIME_DASHES[regime]
            ls = "--" if dashes[0] else "-"
            ax.plot(epochs + 1, vals,
                    color=REGIME_COLORS[regime],
                    linestyle=ls,
                    label=REGIME_LABELS[regime],
                    zorder=3)
            plotted = True

        ax.set_title(n_labels[n], fontsize=9)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Val. MSE loss")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle=":", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        if col == 3 and plotted:
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "FT3 — Training Convergence (HeteroSAGE, validation loss)",
        fontsize=10, fontweight="bold", y=1.03)

    path = os.path.join(OUT_DIR, "fig12_ft3_convergence.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 13 — FT3 convergence HeteroHEAT (side-by-side with SAGE) ─────────────

def plot_ft3_convergence_both_archs(df_ft3):
    """2-row × 4-col: top=HeteroSAGE, bottom=HeteroHEAT; cols=n values."""
    n_vals  = [100, 1000, 10000, 50000]
    n_labels = {100: "$n=100$", 1000: "$n=1{,}000$",
                10000: "$n=10{,}000$", 50000: "$n=50{,}000$"}

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.5), sharey=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.38)

    for r_idx, arch in enumerate(ARCH_ORDER):
        for col, n in enumerate(n_vals):
            ax = axes[r_idx, col]
            sub = df_ft3[(df_ft3["arch"] == arch) & (df_ft3["n"] == n)]
            plotted = False
            for regime in REGIME_ORDER:
                row = sub[sub["regime"] == regime]
                if row.empty:
                    continue
                log_dir = row.iloc[0]["log_dir"]
                epochs, vals = _load_curve(log_dir, tag="validate error")
                if epochs is None:
                    continue
                dashes = REGIME_DASHES[regime]
                ls = "--" if dashes[0] else "-"
                ax.plot(epochs + 1, vals,
                        color=REGIME_COLORS[regime],
                        linestyle=ls,
                        label=REGIME_LABELS[regime],
                        zorder=3)
                plotted = True

            if r_idx == 0:
                ax.set_title(n_labels[n], fontsize=9)
            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(f"{ARCH_LABELS[arch]}\nVal. MSE loss")
            ax.set_yscale("log")
            ax.grid(True, which="major", linestyle=":", alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", alpha=0.2)
            if r_idx == 1 and col == 3 and plotted:
                ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "FT3 — Training Convergence by Architecture and Dataset Size",
        fontsize=10, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "fig13_ft3_convergence_both_archs.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 14 — FT1 convergence both archs ──────────────────────────────────────

def plot_ft1_convergence_both_archs(df_ft1):
    n_vals  = [100, 1000, 10000, 50000]
    n_labels = {100: "$n=100$", 1000: "$n=1{,}000$",
                10000: "$n=10{,}000$", 50000: "$n=50{,}000$"}

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.5), sharey=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.38)

    for r_idx, arch in enumerate(ARCH_ORDER):
        for col, n in enumerate(n_vals):
            ax = axes[r_idx, col]
            sub = df_ft1[(df_ft1["arch"] == arch) & (df_ft1["n"] == n)]
            plotted = False
            for regime in REGIME_ORDER:
                row = sub[sub["regime"] == regime]
                if row.empty:
                    continue
                log_dir = row.iloc[0]["log_dir"]
                epochs, vals = _load_curve(log_dir, tag="validate error")
                if epochs is None:
                    continue
                dashes = REGIME_DASHES[regime]
                ls = "--" if dashes[0] else "-"
                ax.plot(epochs + 1, vals,
                        color=REGIME_COLORS[regime],
                        linestyle=ls,
                        label=REGIME_LABELS[regime],
                        zorder=3)
                plotted = True

            if r_idx == 0:
                ax.set_title(n_labels[n], fontsize=9)
            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(f"{ARCH_LABELS[arch]}\nVal. BCE loss")
            ax.set_yscale("log")
            ax.grid(True, which="major", linestyle=":", alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", alpha=0.2)
            if r_idx == 1 and col == 3 and plotted:
                ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "FT1 — Training Convergence by Architecture and Dataset Size",
        fontsize=10, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "fig14_ft1_convergence_both_archs.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading FT1 results …")
    df_ft1 = _collect_ft1()
    print(f"  {len(df_ft1)} runs found")
    print(df_ft1.groupby(["arch", "regime"]).size().to_string())

    print("\nLoading FT3 results …")
    df_ft3 = _collect_ft3()
    print(f"  {len(df_ft3)} runs found")
    print(df_ft3.groupby(["arch", "regime"]).size().to_string())

    print("\nGenerating figures …")
    plot_ft1_sample_efficiency(df_ft1)
    plot_ft1_convergence(df_ft1)
    plot_ft1_convergence_both_archs(df_ft1)
    plot_ft3_sample_efficiency(df_ft3)
    plot_ft3_sample_efficiency_mse(df_ft3)
    plot_ft3_convergence(df_ft3)
    plot_ft3_convergence_both_archs(df_ft3)

    print("\nAll figures written to:", OUT_DIR)
