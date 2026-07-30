#!/usr/bin/env python3
"""Plot HEAT baseline, Static, and Static-AC MSE and violation curves."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib-hydragnn"),
)

import matplotlib.pyplot as plt
import pandas as pd


KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[0-9eE+\-.]+)")
SPLIT_RE = re.compile(r"\bsplit=(?P<split>train|val|test)\b")
COLORS = {
    "Basic MSE": "#2a9d8f",
    "Static DC": "#3a86ff",
    "Static AC": "#e76f51",
}


def _parse_breakdowns(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "LossBreakdown" not in line:
                continue
            split_match = SPLIT_RE.search(line)
            if split_match is None:
                continue
            row = {"split": split_match.group("split")}
            for match in KV_RE.finditer(line):
                key = match.group("key")
                value = float(match.group("value"))
                row[key] = int(value) if key == "epoch" else value
            rows.append(row)
    return rows


def _has_nonzero_multiplier(row: dict) -> bool:
    return any(
        abs(float(value)) > 1e-12
        for key, value in row.items()
        if key.startswith("mu_")
    )


def _load_curves(baseline_log: Path, physics_job_output: Path) -> pd.DataFrame:
    rows = []
    for row in _parse_breakdowns(baseline_log):
        rows.append({"model": "Basic MSE", **row})

    for row in _parse_breakdowns(physics_job_output):
        # The shared four-model output also contains AL and AL-AC. Static runs
        # have zero multipliers; their flow field identifies DC versus AC.
        if _has_nonzero_multiplier(row):
            continue
        if "raw_ac_apparent_flow" in row:
            model = "Static AC"
        elif "raw_ac_line_flow" in row:
            model = "Static DC"
        else:
            continue
        rows.append({"model": model, **row})

    frame = pd.DataFrame(rows)
    frame = frame[frame["split"].eq("val")].copy()
    expected = {"Basic MSE", "Static DC", "Static AC"}
    missing = expected - set(frame["model"].unique())
    if missing:
        raise RuntimeError("Missing validation curves for: " + ", ".join(sorted(missing)))
    return frame.sort_values(["model", "epoch"])


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _plot_mse(frame: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for model in ("Basic MSE", "Static DC", "Static AC"):
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
    ax.set_title("HeteroHEAT: Effect of Static Physics Penalties on Accuracy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "heteroheat_static_validation_mse")


def _plot_violations(frame: pd.DataFrame, out_dir: Path):
    panels = [
        (
            "raw_voltage_bound",
            ("Basic MSE", "Static DC", "Static AC"),
            "Voltage-Bound Violation MSE",
            "voltage_bound",
        ),
        (
            "raw_ac_angle_diff",
            ("Basic MSE", "Static DC", "Static AC"),
            "AC-Line Angle Violation MSE",
            "ac_line_angle",
        ),
        (
            "raw_ac_line_flow",
            ("Basic MSE", "Static DC"),
            "DC AC-Line Flow Violation MSE",
            "dc_ac_line_flow",
        ),
        (
            "raw_ac_apparent_flow",
            ("Basic MSE", "Static AC"),
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
        # symlog preserves exact zeros while retaining visibility across the
        # many orders of magnitude present in the violation measurements.
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Violation")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("HeteroHEAT Physical Violations on the Validation Set", fontsize=16)
    fig.tight_layout()
    _save(fig, out_dir, "heteroheat_static_physics_violations")

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
        ax.set_title(f"HeteroHEAT Static Penalties: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Violation")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, out_dir, f"heteroheat_static_{filename}_violation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_log",
        type=Path,
        required=True,
        help="Monitor-only HeteroHEAT run.log.",
    )
    parser.add_argument(
        "--physics_job_output",
        type=Path,
        required=True,
        help="Shared Slurm output containing Static and Static-AC breakdowns.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    frame = _load_curves(args.baseline_log, args.physics_job_output)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "heteroheat_static_validation_metrics.csv", index=False)
    _plot_mse(frame, args.out_dir)
    _plot_violations(frame, args.out_dir)
    print(f"Wrote MSE and violation figures to {args.out_dir}")


if __name__ == "__main__":
    main()
