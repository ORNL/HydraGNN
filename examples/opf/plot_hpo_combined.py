#!/usr/bin/env python3
"""
Generate combined HPO validation loss plot from multiple HPO runs,
correctly pairing each CSV source with its DeepHyper output directory.
"""

import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

TYPE_COLORS = {
    "HeteroHEAT": "#d62728",
    "HeteroHGT": "#1f77b4",
    "HeteroPNA": "#2ca02c",
    "HeteroSAGE": "#ff7f0e",
    "HeteroGAT": "#9467bd",
    "HeteroRGAT": "#8c564b",
    "HeteroGIN": "#e377c2",
}
TYPE_MARKERS = {
    "HeteroHEAT": "s",
    "HeteroHGT": "o",
    "HeteroPNA": "D",
    "HeteroSAGE": "^",
    "HeteroGAT": "v",
    "HeteroRGAT": "X",
    "HeteroGIN": "P",
}


def extract_epoch_losses(dh_dir, trial_id):
    path = os.path.join(dh_dir, "output-0.{}.txt".format(trial_id))
    if not os.path.exists(path):
        return []
    losses = []
    with open(path) as f:
        for line in f:
            m = re.match(r"Val loss:\s+([\d.eE+-]+)", line.strip())
            if m:
                val = float(m.group(1))
                if np.isfinite(val):
                    losses.append(val)
    return losses


def load_run(csv_path, dh_dir):
    """Load one HPO run, correctly pairing CSV rows with their dh_dir."""
    trials = []
    n_failed = 0
    failed_types = defaultdict(int)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["objective"] == "F":
                n_failed += 1
                failed_types[r["p:mpnn_type"]] += 1
                continue
            trial_id = r["job_id"]
            mpnn = r["p:mpnn_type"]
            losses = extract_epoch_losses(dh_dir, trial_id)
            if not losses:
                continue
            hparams = {
                "hidden_dim": int(r["p:hidden_dim"]),
                "num_conv_layers": int(r["p:num_conv_layers"]),
                "learning_rate": float(r["p:learning_rate"]),
            }
            trials.append((mpnn, trial_id, hparams, losses))
    return trials, n_failed, dict(failed_types)


def main():
    # Define the two HPO runs: (csv_path, dh_dir)
    runs = [
        ("opf_hpo-4249563/results.csv", "deephyper-opf-hpo-4249563"),
        ("opf_hpo_HeteroPNA-4252004/results.csv", "deephyper-opf-hpo-4252004"),
    ]

    by_type = defaultdict(list)
    total_failed = 0
    all_failed_types = defaultdict(int)

    for csv_path, dh_dir in runs:
        trials, n_failed, failed_types = load_run(csv_path, dh_dir)
        total_failed += n_failed
        for ft, fc in failed_types.items():
            all_failed_types[ft] += fc
        for mpnn, trial_id, hparams, losses in trials:
            by_type[mpnn].append((trial_id, hparams, losses))
        print("Loaded {} successful trials from {} (dh_dir={})".format(
            len(trials), csv_path, dh_dir))

    total = sum(len(v) for v in by_type.values())
    print("\nTotal: {} successful, {} failed".format(total, total_failed))

    # Print summary
    print("\n{:>4} {:>14} {:>6} {:>6} {:>10} {:>13}".format(
        "Rank", "Model Type", "Hidden", "Layers", "LR", "Best Val Loss"))
    print("-" * 60)
    all_trials = []
    for mpnn, trials in by_type.items():
        for trial_id, hparams, losses in trials:
            all_trials.append({
                "mpnn": mpnn, "trial": trial_id,
                "hidden": hparams["hidden_dim"],
                "layers": hparams["num_conv_layers"],
                "lr": hparams["learning_rate"],
                "best_loss": min(losses),
            })
    all_trials.sort(key=lambda x: x["best_loss"])
    for i, t in enumerate(all_trials):
        print("{:>4} {:>14} {:>6} {:>6} {:>10.6f} {:>13.6f}".format(
            i+1, t["mpnn"], t["hidden"], t["layers"], t["lr"], t["best_loss"]))

    # Per-type summary
    print("\n{:>14} {:>6} {:>10} {:>10} {:>10}".format(
        "Model Type", "Trials", "Best", "Mean", "Worst"))
    print("-" * 54)
    type_stats = {}
    for mpnn, trials in by_type.items():
        losses_all = [min(l) for _, _, l in trials]
        type_stats[mpnn] = {
            "count": len(losses_all),
            "best": min(losses_all),
            "worst": max(losses_all),
            "mean": sum(losses_all) / len(losses_all),
        }
    for mpnn in sorted(type_stats, key=lambda t: type_stats[t]["best"]):
        s = type_stats[mpnn]
        print("{:>14} {:>6} {:>10.6f} {:>10.6f} {:>10.6f}".format(
            mpnn, s["count"], s["best"], s["mean"], s["worst"]))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    type_order = sorted(
        by_type.keys(), key=lambda t: min(min(l) for _, _, l in by_type[t]))

    legend_handles = []
    legend_labels = []
    max_epochs = 0
    overall_best_loss = float("inf")
    overall_best_info = {}

    for mpnn in type_order:
        trials = by_type[mpnn]
        color = TYPE_COLORS.get(mpnn, "#333333")
        marker = TYPE_MARKERS.get(mpnn, "o")
        trials.sort(key=lambda t: min(t[2]))

        for idx, (trial_id, hparams, losses) in enumerate(trials):
            epochs = list(range(1, len(losses) + 1))
            max_epochs = max(max_epochs, len(losses))
            alpha = 0.9 if idx == 0 else 0.35
            lw = 2.5 if idx == 0 else 1.0
            line = ax.semilogy(
                epochs, losses,
                color=color, marker=marker,
                markersize=6 if idx == 0 else 4,
                linewidth=lw, alpha=alpha,
                markeredgewidth=0.5, markeredgecolor="white")
            if idx == 0:
                best_loss = min(losses)
                legend_handles.append(line[0])
                legend_labels.append("{} (best: {:.5f})".format(mpnn, best_loss))
                if best_loss < overall_best_loss:
                    overall_best_loss = best_loss
                    best_epoch = losses.index(best_loss) + 1
                    overall_best_info = {
                        "mpnn": mpnn, "trial": trial_id,
                        "loss": best_loss, "epoch": best_epoch}

    ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation Loss", fontsize=13, fontweight="bold")
    ax.set_title(
        "HPO Validation Loss Curves by Model Architecture\n"
        "({} trials, Jobs 4249563 + 4252004, 128 Frontier nodes)".format(total),
        fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, max_epochs + 1))
    ax.set_xlim(0.5, max_epochs + 0.5)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.legend(legend_handles, legend_labels,
              loc="upper right", fontsize=10, framealpha=0.9,
              title="Model Type (best trial val loss)", title_fontsize=10)

    if overall_best_info:
        info = overall_best_info
        ax.annotate(
            "Best overall:\n{}, trial {}\nVal loss = {:.5f}".format(
                info["mpnn"], info["trial"], info["loss"]),
            xy=(info["epoch"], info["loss"]),
            xytext=(max(1, info["epoch"] - 3), info["loss"] * 2.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    plot_path = "hpo_validation_loss_curves.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print("\nPlot saved to {}".format(plot_path))


if __name__ == "__main__":
    main()
