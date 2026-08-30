#!/usr/bin/env python3
"""Plot HEAT baseline, AL-DC, and AL-AC MSE and violation curves."""

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

from plot_heat_static_loss_curves import (
    _has_nonzero_multiplier,
    _parse_breakdowns,
    _save,
)

COLORS = {
    "Basic MSE": "#2a9d8f",
    "AL DC": "#3a86ff",
    "AL AC": "#e76f51",
}


def _load_curves(baseline_log: Path, physics_job_output: Path) -> pd.DataFrame:
    rows = [{"model": "Basic MSE", **row} for row in _parse_breakdowns(baseline_log)]

    for row in _parse_breakdowns(physics_job_output):
        if not _has_nonzero_multiplier(row):
            continue
        if "raw_ac_apparent_flow" in row:
            model = "AL AC"
        elif "raw_ac_line_flow" in row:
            model = "AL DC"
        else:
            continue
        rows.append({"model": model, **row})

    frame = pd.DataFrame(rows)
    frame = frame[frame["split"].eq("val")].copy()
    expected = {"Basic MSE", "AL DC", "AL AC"}
    missing = expected - set(frame["model"].unique())
    if missing:
        raise RuntimeError(
            "Missing validation curves for: " + ", ".join(sorted(missing))
        )
    return frame.sort_values(["model", "epoch"])


def _plot_mse(frame: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for model in ("Basic MSE", "AL DC", "AL AC"):
        group = frame[frame["model"].eq(model)]
        ax.plot(
            group["epoch"],
            group["data_driven_mse"],
            label=model,
            color=COLORS[model],
            linewidth=2.2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (log scale)")
    ax.set_title("HeteroHEAT: Effect of Augmented-Lagrangian Physics Loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "heteroheat_al_validation_mse")


def _plot_violations(frame: pd.DataFrame, out_dir: Path):
    panels = [
        (
            "raw_voltage_bound",
            ("Basic MSE", "AL DC", "AL AC"),
            "Voltage-Bound Violation MSE",
            "voltage_bound",
        ),
        (
            "raw_ac_angle_diff",
            ("Basic MSE", "AL DC", "AL AC"),
            "AC-Line Angle Violation MSE",
            "ac_line_angle",
        ),
        (
            "raw_ac_line_flow",
            ("Basic MSE", "AL DC"),
            "DC AC-Line Flow Violation MSE",
            "dc_ac_line_flow",
        ),
        (
            "raw_ac_apparent_flow",
            ("Basic MSE", "AL AC"),
            "AC Apparent-Flow Violation MSE",
            "ac_apparent_flow",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    for ax, (metric, models, title, _) in zip(axes.flat, panels):
        for model in models:
            group = frame[frame["model"].eq(model)]
            if metric not in group or group[metric].isna().all():
                continue
            ax.plot(
                group["epoch"],
                group[metric],
                label=model,
                color=COLORS[model],
                linewidth=2.0,
            )
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Violation")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "HeteroHEAT Augmented-Lagrangian Violations on the Validation Set",
        fontsize=16,
    )
    fig.tight_layout()
    _save(fig, out_dir, "heteroheat_al_physics_violations")

    for metric, models, title, filename in panels:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for model in models:
            group = frame[frame["model"].eq(model)]
            if metric not in group or group[metric].isna().all():
                continue
            ax.plot(
                group["epoch"],
                group[metric],
                label=model,
                color=COLORS[model],
                linewidth=2.2,
            )
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_title(f"HeteroHEAT Augmented Lagrangian: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Violation")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, out_dir, f"heteroheat_al_{filename}_violation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_log", type=Path, required=True)
    parser.add_argument("--physics_job_output", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    frame = _load_curves(args.baseline_log, args.physics_job_output)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "heteroheat_al_validation_metrics.csv", index=False)
    _plot_mse(frame, args.out_dir)
    _plot_violations(frame, args.out_dir)
    print(f"Wrote AL MSE and violation figures to {args.out_dir}")


if __name__ == "__main__":
    main()
