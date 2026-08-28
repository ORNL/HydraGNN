#!/usr/bin/env python3
"""Plot full TensorBoard curves for the case4661 PE experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "case4661_bus_attention_control": {
        "label": "Control",
        "color": "#555555",
        "linestyle": "--",
    },
    "case4661_effective_resistance": {
        "label": "ER",
        "color": "#0072B2",
        "linestyle": "-",
    },
    "case4661_lpe": {
        "label": "L",
        "color": "#E69F00",
        "linestyle": "-",
    },
    "case4661_lpe_effective_resistance": {
        "label": "L + ER",
        "color": "#009E73",
        "linestyle": "-",
    },
    "case4661_lpe_signflip": {
        "label": "L + SF",
        "color": "#CC79A7",
        "linestyle": "-",
    },
    "case4661_lpe_signflip_effective_resistance": {
        "label": "L + SF + ER",
        "color": "#D55E00",
        "linestyle": "-",
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
        help="Directory containing the TensorBoard run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "case4661_pe_full_curves.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_run(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    values_by_tag_and_step = {tag: {} for tag, _ in PANELS}
    for event_file in event_files:
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        available_tags = set(accumulator.Tags()["scalars"])
        for tag, _ in PANELS:
            if tag not in available_tags:
                continue
            for event in accumulator.Scalars(tag):
                # A resumed segment can repeat its checkpoint boundary. As in
                # TensorBoard, the later event replaces the earlier value.
                values_by_tag_and_step[tag][event.step] = event.value

    curves = {}
    for tag, _ in PANELS:
        values_by_step = values_by_tag_and_step[tag]
        if not values_by_step:
            raise RuntimeError(f"Scalar tag {tag!r} is missing from {run_dir}")
        steps = sorted(values_by_step)
        curves[tag] = (steps, [values_by_step[step] for step in steps])
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
            "legend.fontsize": 9.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True)

    for axis, (tag, title) in zip(axes, PANELS):
        for run, style in RUNS.items():
            steps, values = histories[run][tag]
            axis.plot(
                steps,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
                linewidth=1.9,
                marker="o",
                markersize=2.8,
                markevery=3,
                alpha=0.95,
            )

            selected_step = best_steps[run]
            selected_value = value_at_step(histories[run][tag], selected_step)
            axis.scatter(
                selected_step,
                selected_value,
                marker="*",
                s=105,
                color=style["color"],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.11)
        axis.set_xlim(0, 31)
        axis.set_xticks([0, 5, 10, 15, 20, 25, 30])

    axes[0].set_ylabel("Mean squared error (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        "case4661 positional-encoding comparison",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.84))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(args.output.resolve())
    for run, style in RUNS.items():
        best_step = best_steps[run]
        best_value = value_at_step(histories[run]["validate error"], best_step)
        print(
            f"{style['label']}: epochs={len(histories[run]['validate error'][0])}, "
            f"best_step={best_step}, best_val={best_value:.6e}"
        )


if __name__ == "__main__":
    main()
