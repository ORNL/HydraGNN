#!/usr/bin/env python3
"""Compare two HydraGNN training logs (baseline vs physics-informed).

Usage:
    python compare_heat_runs.py <baseline_run.log> <physics_run.log> [--json]
"""
import json
import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"Epoch:\s*(\d+),\s*Train Loss:\s*([0-9eE+\-.]+),\s*Val Loss:\s*([0-9eE+\-.]+),\s*Test Loss:\s*([0-9eE+\-.]+)"
)


def extract_stats(log_path: Path):
    """Return first, best, last epoch rows plus total epoch count, or None."""
    rows = []
    with log_path.open() as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            epoch, train_loss, val_loss, test_loss = m.groups()
            rows.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "test_loss": float(test_loss),
                }
            )
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val_loss"])
    return {
        "first": rows[0],
        "last": rows[-1],
        "best": best,
        "num_epochs": len(rows),
    }


def _fmt(stats: dict) -> str:
    b = stats["best"]
    return (
        f"  epochs logged : {stats['num_epochs']}\n"
        f"  first epoch   : {stats['first']['epoch']}  "
        f"train={stats['first']['train_loss']:.6f}  "
        f"val={stats['first']['val_loss']:.6f}  "
        f"test={stats['first']['test_loss']:.6f}\n"
        f"  last epoch    : {stats['last']['epoch']}  "
        f"train={stats['last']['train_loss']:.6f}  "
        f"val={stats['last']['val_loss']:.6f}  "
        f"test={stats['last']['test_loss']:.6f}\n"
        f"  best val epoch: {b['epoch']}  "
        f"train={b['train_loss']:.6f}  "
        f"val={b['val_loss']:.6f}  "
        f"test={b['test_loss']:.6f}"
    )


def main(argv):
    use_json = "--json" in argv
    paths = [a for a in argv[1:] if not a.startswith("--")]
    if len(paths) != 2:
        raise SystemExit(
            "Usage: compare_heat_runs.py <baseline_run.log> <physics_run.log> [--json]"
        )

    baseline_path = Path(paths[0])
    physics_path = Path(paths[1])

    for p in (baseline_path, physics_path):
        if not p.exists():
            raise SystemExit(f"Log file not found: {p}")

    baseline = extract_stats(baseline_path)
    physics = extract_stats(physics_path)

    if baseline is None:
        raise SystemExit(f"No epoch metrics found in baseline log: {baseline_path}")
    if physics is None:
        raise SystemExit(f"No epoch metrics found in physics log: {physics_path}")

    delta_val = physics["best"]["val_loss"] - baseline["best"]["val_loss"]
    delta_test = physics["best"]["test_loss"] - baseline["best"]["test_loss"]

    if use_json:
        print(
            json.dumps(
                {
                    "baseline": baseline,
                    "physics": physics,
                    "delta_best_val_loss": delta_val,
                    "delta_best_test_loss": delta_test,
                },
                indent=2,
            )
        )
        return

    print(f"Baseline ({baseline_path}):")
    print(_fmt(baseline))
    print()
    print(f"Physics-informed ({physics_path}):")
    print(_fmt(physics))
    print()
    sign_val = "+" if delta_val >= 0 else ""
    sign_test = "+" if delta_test >= 0 else ""
    print(f"Delta best val_loss  (physics - baseline): {sign_val}{delta_val:.6f}")
    print(f"Delta best test_loss (physics - baseline): {sign_test}{delta_test:.6f}")
    if delta_val < 0:
        print("=> Physics-informed loss IMPROVED best validation loss.")
    else:
        print("=> Physics-informed loss did NOT improve best validation loss.")


if __name__ == "__main__":
    main(sys.argv)
