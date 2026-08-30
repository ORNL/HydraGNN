#!/usr/bin/env python3
"""Create case500 OPF ablation figures for the four research questions."""

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

ARCH_ORDER = ["HeteroHEAT", "HeteroSAGE", "HeteroGAT", "HeteroGIN", "HeteroRGAT"]
PHYSICS_ORDER = [
    "heat_attr_case500_no_physics",
    "heat_attr_case500",
    "heat_attr_case500_AC_fixed",
    "heat_attr_case500_AL",
    "heat_attr_case500_AL_AC_fixed",
]
PHYSICS_LABELS = {
    "heat_attr_case500_no_physics": "No Physics",
    "heat_attr_case500": "Static",
    "heat_attr_case500_AC_fixed": "Static AC",
    "heat_attr_case500_AL": "AL",
    "heat_attr_case500_AL_AC_fixed": "AL AC",
}
COLORS = {
    "attr": "#2a9d8f",
    "no_attr": "#e76f51",
    "val": "#3a86ff",
    "test": "#ffbe0b",
    "runtime": "#6d597a",
}


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["edge_attr_enabled"] = df["edge_attr_enabled"].astype(str).str.lower().eq("true")
    return df


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _clean_arch(row) -> str:
    arch = row["mpnn_type"]
    if arch == "HeteroHEAT":
        return "HEAT"
    return arch.replace("Hetero", "")


def _q1_edge_architecture(df: pd.DataFrame, out_dir: Path):
    q1 = df[df["domain_loss_mode"].eq("none")].copy()
    q1["arch_label"] = q1.apply(_clean_arch, axis=1)
    q1["arch_order"] = q1["mpnn_type"].map(
        {arch: i for i, arch in enumerate(ARCH_ORDER)}
    )
    q1 = q1.sort_values(["arch_order", "edge_attr_enabled"])

    pivot = q1.pivot(
        index="arch_label", columns="edge_attr_enabled", values="best_test_loss"
    )
    pivot = pivot.reindex(
        [a.replace("Hetero", "") if a != "HeteroHEAT" else "HEAT" for a in ARCH_ORDER]
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = range(len(pivot.index))
    width = 0.36
    no_attr = pivot.get(False)
    attr = pivot.get(True)
    ax.bar(
        [i - width / 2 for i in x],
        no_attr,
        width,
        label="No Edge Attr",
        color=COLORS["no_attr"],
    )
    ax.bar(
        [i + width / 2 for i in x], attr, width, label="Edge Attr", color=COLORS["attr"]
    )
    ax.set_yscale("log")
    ax.set_ylabel("Best Test Loss (log scale)")
    ax.set_title("Q1: Architecture and Edge-Attribute Effect")
    ax.set_xticks(list(x), pivot.index)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "q1_architecture_edge_attr_best_test")

    # Edge-attribute gain within each architecture.
    rows = []
    for arch, group in q1.groupby("arch_label"):
        vals = group.set_index("edge_attr_enabled")["best_test_loss"]
        if True in vals.index and False in vals.index:
            rows.append(
                {
                    "arch_label": arch,
                    "attr_improvement_percent": 100.0
                    * (vals.loc[False] - vals.loc[True])
                    / vals.loc[False],
                }
            )
    gain = pd.DataFrame(rows)
    gain["arch_order"] = gain["arch_label"].map(
        {"HEAT": 0, "SAGE": 1, "GAT": 2, "GIN": 3, "RGAT": 4}
    )
    gain = gain.sort_values("arch_order")

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.axhline(0, color="#444444", linewidth=1)
    ax.bar(gain["arch_label"], gain["attr_improvement_percent"], color=COLORS["attr"])
    ax.set_ylabel("Test Loss Improvement from Edge Attr (%)")
    ax.set_title("Q1: How Much Do Edge Attributes Help?")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "q1_edge_attr_improvement_percent")


def _physics_subset(df: pd.DataFrame) -> pd.DataFrame:
    phys = df[df["log_name"].isin(PHYSICS_ORDER)].copy()
    phys["label"] = phys["log_name"].map(PHYSICS_LABELS)
    phys["order"] = phys["log_name"].map(
        {name: i for i, name in enumerate(PHYSICS_ORDER)}
    )
    return phys.sort_values("order")


def _q2_q3_q4_physics(df: pd.DataFrame, out_dir: Path):
    phys = _physics_subset(df)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    x = range(len(phys))
    width = 0.36
    ax1.bar(
        [i - width / 2 for i in x],
        phys["best_val_loss"],
        width,
        label="Best Val",
        color=COLORS["val"],
    )
    ax1.bar(
        [i + width / 2 for i in x],
        phys["best_test_loss"],
        width,
        label="Best Test",
        color=COLORS["test"],
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Q2-Q4: Physics Loss Variants vs Accuracy")
    ax1.set_xticks(list(x), phys["label"], rotation=20, ha="right")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "q2_q3_q4_physics_accuracy")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(phys["label"], phys["total_train_seconds"] / 60.0, color=COLORS["runtime"])
    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Q2-Q4: Compute Cost by Physics Variant")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "q2_q3_q4_physics_runtime")

    baseline = phys.loc[phys["log_name"].eq("heat_attr_case500_no_physics")].iloc[0]
    compare = phys[~phys["log_name"].eq("heat_attr_case500_no_physics")].copy()
    compare["test_loss_change_percent"] = (
        100.0
        * (compare["best_test_loss"] - baseline["best_test_loss"])
        / baseline["best_test_loss"]
    )
    compare["runtime_change_percent"] = (
        100.0
        * (compare["total_train_seconds"] - baseline["total_train_seconds"])
        / baseline["total_train_seconds"]
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvline(0, color="#777777", linewidth=1)
    ax.axhline(0, color="#777777", linewidth=1)
    ax.scatter(
        compare["runtime_change_percent"],
        compare["test_loss_change_percent"],
        s=90,
        color="#3a86ff",
    )
    for _, row in compare.iterrows():
        ax.annotate(
            row["label"],
            (row["runtime_change_percent"], row["test_loss_change_percent"]),
            xytext=(6, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Runtime Change vs No Physics (%)")
    ax.set_ylabel("Best Test Loss Change vs No Physics (%)")
    ax.set_title("Q2-Q4: Accuracy-Cost Tradeoff")
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "q2_q3_q4_accuracy_runtime_tradeoff")

    compare.to_csv(out_dir / "physics_vs_no_physics_percent_changes.csv", index=False)


def _learning_curves(epochs: pd.DataFrame, out_dir: Path):
    selected = PHYSICS_ORDER + [
        "heat_no_attr_case500_no_physics",
        "heterosage_attr_case500_no_physics",
    ]
    labels = {
        **PHYSICS_LABELS,
        "heat_no_attr_case500_no_physics": "HEAT No Attr",
        "heterosage_attr_case500_no_physics": "SAGE Attr",
    }
    curve = epochs[epochs["log_name"].isin(selected)].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for log_name, group in curve.groupby("log_name"):
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["val_loss"],
            label=labels.get(log_name, log_name),
            linewidth=1.8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (log scale)")
    ax.set_title("Case500 Validation Curves")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "case500_validation_curves")


def _heat_edge_attribute_mse(epochs: pd.DataFrame, out_dir: Path):
    """Plot the controlled HEAT edge-attribute ablation and nothing else."""
    runs = {
        "heat_attr_case500_no_physics": "HeteroHEAT with Edge Attributes",
        "heat_no_attr_case500_no_physics": "HeteroHEAT without Edge Attributes",
    }
    curve = epochs[epochs["log_name"].isin(runs)].copy()
    missing = set(runs) - set(curve["log_name"].unique())
    if missing:
        raise RuntimeError(
            "Missing HEAT edge-ablation epoch data for: " + ", ".join(sorted(missing))
        )

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for log_name, label in runs.items():
        group = curve[curve["log_name"].eq(log_name)].sort_values("epoch")
        color = COLORS["attr"] if "no_attr" not in log_name else COLORS["no_attr"]
        ax.plot(
            group["epoch"],
            group["val_loss"],
            label=label,
            color=color,
            linewidth=2.2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (log scale)")
    ax.set_title("HeteroHEAT: Effect of Physical Edge Attributes")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "heteroheat_edge_attributes_validation_mse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_case500_final",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures_case500",
    )
    parser.add_argument(
        "--only",
        choices=["all", "heat-edge-mse"],
        default="all",
        help="Generate all case500 figures or only the HEAT edge-attribute MSE chart.",
    )
    args = parser.parse_args()

    summary = _read_csv(args.results_dir / "opf_ablation_summary.csv")
    epochs = pd.read_csv(args.results_dir / "opf_ablation_epochs.csv")

    if args.only == "heat-edge-mse":
        _heat_edge_attribute_mse(epochs, args.out_dir)
        print(f"Wrote HEAT edge-attribute MSE figure to {args.out_dir}")
        return

    _q1_edge_architecture(summary, args.out_dir)
    _q2_q3_q4_physics(summary, args.out_dir)
    _learning_curves(epochs, args.out_dir)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
