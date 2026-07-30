#!/usr/bin/env python3
"""Compare validation-set violations for HeteroHEAT with and without edge attributes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib-hydragnn"),
)

import matplotlib.pyplot as plt
import pandas as pd

from plot_heat_static_loss_curves import _parse_breakdowns, _save


MODEL_WITH = "HeteroHEAT with Edge Attributes"
MODEL_WITHOUT = "HeteroHEAT without Edge Attributes"
MODELS = (MODEL_WITH, MODEL_WITHOUT)
COLORS = {
    MODEL_WITH: "#2a9d8f",
    MODEL_WITHOUT: "#e76f51",
}
PANELS = (
    ("raw_voltage_bound", "Voltage-Bound Violation MSE", "voltage_bound"),
    ("raw_ac_angle_diff", "AC-Line Angle Violation MSE", "ac_line_angle"),
    ("raw_ac_line_flow", "DC AC-Line Flow Violation MSE", "dc_ac_line_flow"),
    ("raw_ac_apparent_flow", "AC Apparent-Flow Violation MSE", "ac_apparent_flow"),
)


def _load_validation_curves(with_log: Path, without_log: Path) -> pd.DataFrame:
    rows = []
    for model, path in ((MODEL_WITH, with_log), (MODEL_WITHOUT, without_log)):
        parsed = _parse_breakdowns(path)
        validation = [row for row in parsed if row.get("split") == "val"]
        if not validation:
            raise RuntimeError(f"No validation LossBreakdown records found in {path}")
        rows.extend({"model": model, **row} for row in validation)

    frame = pd.DataFrame(rows).sort_values(["model", "epoch"])
    missing_metrics = [
        metric for metric, _, _ in PANELS if metric not in frame.columns
    ]
    if missing_metrics:
        raise RuntimeError(
            "Missing violation metrics: " + ", ".join(missing_metrics)
        )
    return frame


def _plot_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    for model in MODELS:
        group = frame[frame["model"].eq(model)]
        ax.plot(
            group["epoch"],
            group[metric],
            label=model,
            color=COLORS[model],
            linewidth=2.2,
        )
    # The violation terms span many orders of magnitude and can be exactly zero.
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Violation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def _plot_violations(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    for ax, (metric, title, _) in zip(axes.flat, PANELS):
        _plot_metric(ax, frame, metric, title)
    fig.suptitle(
        "HeteroHEAT Physical Violations: Effect of Edge Attributes",
        fontsize=16,
    )
    fig.tight_layout()
    _save(fig, out_dir, "heteroheat_edge_attributes_physics_violations")

    for metric, title, filename in PANELS:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        _plot_metric(ax, frame, metric, title)
        ax.set_title(f"HeteroHEAT Edge-Attribute Comparison: {title}")
        _save(
            fig,
            out_dir,
            f"heteroheat_edge_attributes_{filename}_violation",
        )


def _write_best_epoch_summary(frame: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for model in MODELS:
        group = frame[frame["model"].eq(model)]
        best = group.loc[group["data_driven_mse"].idxmin()]
        rows.append(
            {
                "model": model,
                "best_validation_epoch": int(best["epoch"]),
                "best_validation_mse": best["data_driven_mse"],
                **{metric: best[metric] for metric, _, _ in PANELS},
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "heteroheat_edge_attributes_best_epoch_violations.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with_edge_log", type=Path, required=True)
    parser.add_argument("--without_edge_log", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    frame = _load_validation_curves(args.with_edge_log, args.without_edge_log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.out_dir / "heteroheat_edge_attributes_validation_violations.csv",
        index=False,
    )
    _write_best_epoch_summary(frame, args.out_dir)
    _plot_violations(frame, args.out_dir)
    print(f"Wrote edge-attribute violation comparison to {args.out_dir}")


if __name__ == "__main__":
    main()
