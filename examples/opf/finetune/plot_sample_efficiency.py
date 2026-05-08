#!/usr/bin/env python3
"""Aggregate OPF FT1 + FT3 single-method, single-N runs into sample-efficiency plots.

Scans ``examples/opf/finetune/logs/`` for directories produced by
``train_opf_ft1_classify.py`` and ``train_opf_finetune.py``, pulls the test
metric from each run's ``results.json``, and produces one PNG per
(task, architecture, metric) showing the metric versus number of training
samples for each fine-tuning method (full, partial, head_only, scratch).

Outputs land in ``examples/opf/finetune/plots/``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# log directory naming conventions used by the training scripts
FT1_RE = re.compile(
    r"^FT1_feasibility_(?P<arch>[A-Za-z0-9]+)_(?P<regime>full|partial|head_only)"
    r"(?P<scratch>_scratch)?_n(?P<total>\d+)$"
)
FT3_RE = re.compile(
    r"^finetune_FT3_contingency_(?P<arch>[A-Za-z0-9]+)_(?P<regime>full|partial|head_only)"
    r"(?P<scratch>_scratch)?_n(?P<n>\d+)$"
)

# which metric to plot per task (key inside results.json["test_metrics"])
FT1_METRICS = ["bce", "accuracy", "f1", "auc_roc"]
FT3_METRICS = ["overall_mse"]

METHOD_ORDER = ["full", "partial", "head_only", "scratch"]
METHOD_STYLE = {
    "full":      {"color": "#1f77b4", "marker": "o"},
    "partial":   {"color": "#ff7f0e", "marker": "s"},
    "head_only": {"color": "#2ca02c", "marker": "^"},
    "scratch":   {"color": "#d62728", "marker": "x"},
}


def collect(logs_dir: Path) -> dict:
    """Return nested dict: results[task][arch][method][n] = metrics dict."""
    results: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir():
            continue
        m1 = FT1_RE.match(d.name)
        m3 = FT3_RE.match(d.name)
        if m1:
            task = "FT1"
            arch = m1["arch"]
            method = "scratch" if m1["scratch"] else m1["regime"]
            # LOG_NAME is now keyed on requested N directly.
            n = int(m1["total"])
        elif m3:
            task = "FT3"
            arch = m3["arch"]
            method = "scratch" if m3["scratch"] else m3["regime"]
            n = int(m3["n"])
        else:
            continue

        results_file = d / "results.json"
        if not results_file.is_file():
            continue
        try:
            payload = json.loads(results_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("test_metrics", {})
        if not metrics:
            continue
        results[task][arch][method][n] = metrics
    return results


def plot_one(task: str, arch: str, metric: str,
             per_method: dict, out_path: Path) -> None:
    """Draw a single sample-efficiency curve and write to ``out_path``."""
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_any = False
    for method in METHOD_ORDER:
        run_map = per_method.get(method, {})
        if not run_map:
            continue
        ns = sorted(run_map)
        ys = [run_map[n].get(metric) for n in ns]
        pairs = [(n, y) for n, y in zip(ns, ys) if y is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        style = METHOD_STYLE[method]
        ax.plot(xs, ys, label=method, linewidth=1.6,
                marker=style["marker"], color=style["color"])
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    if metric in {"bce", "overall_mse"}:
        ax.set_yscale("log")
    ax.set_xlabel("Number of fine-tuning samples (N)")
    ax.set_ylabel(f"Test {metric}")
    ax.set_title(f"{task} {arch} — sample efficiency ({metric})")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(title="Fine-tuning regime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path,
                    default=Path(__file__).parent / "logs",
                    help="Path to logs directory")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "plots",
                    help="Output directory for PNGs and aggregated CSV")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = collect(args.logs)

    # write a flat CSV summary for downstream use in the manuscript
    summary_csv = args.out / "summary.csv"
    rows = ["task,arch,method,n,metric,value"]
    for task, archs in results.items():
        metric_keys = FT1_METRICS if task == "FT1" else FT3_METRICS
        for arch, methods in archs.items():
            for metric in metric_keys:
                per_method = {m: methods.get(m, {}) for m in METHOD_ORDER}
                out_path = args.out / f"{task}_{arch}_{metric}.png"
                plot_one(task, arch, metric, per_method, out_path)
            for method, ns in methods.items():
                for n, mvals in sorted(ns.items()):
                    for k, v in mvals.items():
                        if isinstance(v, (int, float)):
                            rows.append(f"{task},{arch},{method},{n},{k},{v}")
    summary_csv.write_text("\n".join(rows) + "\n")
    print(f"[csv] {summary_csv}  ({len(rows) - 1} rows)")


if __name__ == "__main__":
    main()
