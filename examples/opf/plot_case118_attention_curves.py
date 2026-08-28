#!/usr/bin/env python3
"""Plot full TensorBoard curves for the case118 attention comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "case118_heterosage_gps_all_attention": {
        "label": "All-node attention",
        "color": "#D55E00",
    },
    "case118_heterosage_gps_bus_attention": {
        "label": "Bus-only attention",
        "color": "#0072B2",
    },
}

PANELS = (
    ("train error", "Train MSE"),
    ("validate error", "Validation MSE"),
    ("test error", "Test MSE"),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=script_dir / "logs",
        help="Directory containing the two TensorBoard run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "case118_attention_full_curves.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_run(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")

    accumulator = EventAccumulator(
        str(run_dir), size_guidance={"scalars": 0}
    )
    accumulator.Reload()

    curves = {}
    for tag, _ in PANELS:
        events = accumulator.Scalars(tag)
        if not events:
            raise RuntimeError(f"Scalar tag {tag!r} is missing from {run_dir}")
        curves[tag] = (
            [event.step for event in events],
            [event.value for event in events],
        )
    return curves


def value_at_step(
    curve: tuple[list[int], list[float]], step: int
) -> float:
    steps, values = curve
    return values[steps.index(step)]


def main() -> None:
    args = parse_args()
    histories = {
        run: load_run(args.log_root / run)
        for run in RUNS
    }
    best_steps = {
        run: min(
            zip(
                histories[run]["validate error"][1],
                histories[run]["validate error"][0],
            )
        )[1]
        for run in RUNS
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)

    for axis, (tag, title) in zip(axes, PANELS):
        for run, style in RUNS.items():
            steps, values = histories[run][tag]
            axis.plot(
                steps,
                values,
                color=style["color"],
                label=style["label"],
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                markevery=5,
            )

            selected_step = best_steps[run]
            selected_value = value_at_step(histories[run][tag], selected_step)
            axis.scatter(
                selected_step,
                selected_value,
                marker="*",
                s=150,
                color=style["color"],
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

            if tag == "validate error":
                axis.annotate(
                    f"epoch {selected_step}\n{selected_value:.2e}",
                    xy=(selected_step, selected_value),
                    xytext=(-7, 13),
                    textcoords="offset points",
                    color=style["color"],
                    fontsize=8.5,
                    ha="right",
                )

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.12)
        axis.set_xlim(0, 49)
        axis.set_xticks([0, 10, 20, 30, 40, 49])

    axes[0].set_ylabel("Mean squared error (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle(
        "case118 HeteroSAGE + GPS: attention-scope comparison",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
