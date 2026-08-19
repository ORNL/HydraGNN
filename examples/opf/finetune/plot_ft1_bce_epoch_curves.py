#!/usr/bin/env python3
"""Plot FT1 BCE loss curves vs epoch for HeteroSAGE and HeteroHEAT.

This script scans FT1 run directories under logs/, parses per-epoch losses from
run.log, and aggregates runs by fine-tuning strategy (full, partial, head_only,
scratch). It generates one figure per architecture with mean curve and +/- 1 std
band across all N-regimes.

Outputs are written as both PNG and PDF for manuscript use.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FT1_RE = re.compile(
    r"^FT1_feasibility_(?P<arch>[A-Za-z0-9]+)_(?P<regime>full|partial|head_only)"
    r"(?P<scratch>_scratch)?_n(?P<n>\d+)$"
)

EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*"
    r"Train Loss:\s*(?P<train>[-+0-9.eE]+),\s*"
    r"Val Loss:\s*(?P<val>[-+0-9.eE]+),\s*"
    r"Test Loss:\s*(?P<test>[-+0-9.eE]+)"
)

METHOD_ORDER = ["full", "partial", "head_only", "scratch"]
METHOD_LABEL = {
    "full": "Full",
    "partial": "Partial",
    "head_only": "Head-only",
    "scratch": "Scratch",
}
METHOD_STYLE = {
    "full": {"color": "#1f77b4"},
    "partial": {"color": "#ff7f0e"},
    "head_only": {"color": "#2ca02c"},
    "scratch": {"color": "#d62728"},
}


def parse_run_log(run_log: Path, split: str) -> Dict[int, float]:
    """Return epoch->loss parsed from run.log for the requested split."""
    loss_by_epoch = {}
    with run_log.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            loss = float(m.group(split))
            loss_by_epoch[epoch] = loss
    return loss_by_epoch


def collect(logs_dir: Path, split: str) -> Dict[str, Dict[str, List[Dict[int, float]]]]:
    """Collect per-run epoch curves organized as arch -> method -> list(curves)."""
    curves = defaultdict(
        lambda: defaultdict(list)
    )

    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = FT1_RE.match(d.name)
        if not m:
            continue

        arch = m.group("arch")
        method = "scratch" if m.group("scratch") else m.group("regime")

        run_log = d / "run.log"
        if not run_log.is_file():
            continue

        curve = parse_run_log(run_log, split=split)
        if curve:
            curves[arch][method].append(curve)

    return curves


def summarize_curves(curves: List[Dict[int, float]]) -> Tuple[List[int], List[float], List[float]]:
    """Return epochs, mean_loss, std_loss from a list of epoch->loss curves."""
    by_epoch = defaultdict(list)
    for c in curves:
        for e, v in c.items():
            by_epoch[e].append(v)

    epochs = sorted(by_epoch)
    means = [mean(by_epoch[e]) for e in epochs]
    stds = []
    for e in epochs:
        vals = by_epoch[e]
        mu = mean(vals)
        var = sum((x - mu) ** 2 for x in vals) / len(vals)
        stds.append(var ** 0.5)
    return epochs, means, stds


def configure_pub_style() -> None:
    """Set plotting defaults suitable for a two-column IEEE manuscript."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "legend.title_fontsize": 15,
            "lines.linewidth": 3.0,
        }
    )


def plot_architecture(
    arch: str,
    method_curves: Dict[str, List[Dict[int, float]]],
    split_label: str,
    out_base: Path,
    dpi: int,
) -> None:
    """Plot one architecture figure and save PNG+PDF."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    plotted_any = False
    for method in METHOD_ORDER:
        curves = method_curves.get(method, [])
        if not curves:
            continue

        epochs, means, stds = summarize_curves(curves)
        color = METHOD_STYLE[method]["color"]
        label = f"{METHOD_LABEL[method]} (n={len(curves)})"

        ax.plot(epochs, means, color=color, label=label)
        lower = [max(m - s, 1e-12) for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.18)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{split_label} BCE Loss")
    ax.set_title(f"FT1 {arch}: {split_label} BCE vs Epoch")
    ax.grid(True, which="both", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.legend(title="Fine-tuning Strategy", loc="best", frameon=True)

    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] {png_path}")
    print(f"[plot] {pdf_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--logs",
        type=Path,
        default=Path(__file__).parent / "logs",
        help="Path to FT run logs directory",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="Directory to save figure files",
    )
    ap.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Which BCE split from run.log to plot",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG export DPI (use >=300 for publication)",
    )
    args = ap.parse_args()

    configure_pub_style()
    args.out.mkdir(parents=True, exist_ok=True)

    curves = collect(args.logs, split=args.split)
    split_label = {"train": "Train", "val": "Validation", "test": "Test"}[args.split]

    for arch in ["HeteroSAGE", "HeteroHEAT"]:
        if arch not in curves:
            print(f"[warn] no FT1 curves found for {arch}")
            continue
        out_base = args.out / f"FT1_{arch}_{args.split}_bce_vs_epoch"
        plot_architecture(arch, curves[arch], split_label, out_base, args.dpi)


if __name__ == "__main__":
    main()
