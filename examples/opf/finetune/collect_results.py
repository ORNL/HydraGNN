"""Aggregate results.json files from all FT1 and FT3 experiment runs.

Scans the logs/ directory for completed runs, reads their results.json and
training_curve.csv files, and writes two summary artefacts to results/:
  results/ft1_ft3_summary.csv   — one row per run, flat columns
  results/ft1_ft3_summary.json  — same data as structured JSON (for plots)

Usage::

    # From examples/opf/finetune/
    python collect_results.py
    python collect_results.py --logs_root ../../logs --out_dir results

The output files are intentionally human-readable and can be reloaded by
plot_ft_results.py to regenerate all figures without re-running experiments.
"""

import argparse
import csv
import glob
import json
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_results_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_training_curve(csv_path: str) -> list[dict]:
    """Return list of {step, tag, value} rows."""
    if not os.path.isfile(csv_path):
        return []
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({
                    "step":  int(row["step"]) if row["step"] else None,
                    "tag":   row["tag"],
                    "value": float(row["value"]) if row["value"] else None,
                })
            except (KeyError, ValueError):
                continue
    return rows


def _pivot_training_curve(rows: list[dict]) -> dict[str, list]:
    """Convert flat rows to tag → list of (epoch, value) pairs."""
    from collections import defaultdict
    result = defaultdict(list)
    for r in rows:
        if r["step"] is not None and r["value"] is not None:
            result[r["tag"]].append((r["step"], r["value"]))
    return dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# Core collection logic
# ─────────────────────────────────────────────────────────────────────────────

def collect(logs_root: str, out_dir: str) -> None:
    """Scan logs_root for FT1/FT3 results and write summary files."""
    # Discover all results.json files for FT1 and FT3 runs.
    # We match any log name that starts with FT1_ or FT3_.
    patterns = [
        os.path.join(logs_root, "FT1_*", "results.json"),
        os.path.join(logs_root, "FT3_*", "results.json"),
    ]
    found = []
    for pat in patterns:
        found.extend(sorted(glob.glob(pat)))

    if not found:
        print(f"[collect] No results.json files found under {logs_root}/FT1_* or FT3_*.")
        print("  Have all experiments finished?  Check that save_run_results() completed.")
        sys.exit(0)

    print(f"[collect] Found {len(found)} results file(s).")

    records = []   # flat dicts for CSV
    detailed = []  # full JSON payloads (with training curves) for JSON

    for results_path in found:
        log_dir  = os.path.dirname(results_path)
        log_name = os.path.basename(log_dir)

        payload = _load_results_json(results_path)
        meta    = payload.get("meta", {})
        metrics = payload.get("test_metrics", {})

        # Load training curve
        curve_path = os.path.join(log_dir, "training_curve.csv")
        curve_rows = _load_training_curve(curve_path)
        curve      = _pivot_training_curve(curve_rows)

        # ── Determine friendly run label ──────────────────────────────────
        strategy  = meta.get("ft_strategy", "unknown")
        arch      = meta.get("arch", "unknown")
        regime    = meta.get("regime", "full")
        pretrained = meta.get("pretrained", True)
        label     = f"{arch}_{regime}" + ("" if pretrained else "_baseline")

        # ── Flat record (one row per run) ─────────────────────────────────
        flat = {
            "log_name":  log_name,
            "strategy":  strategy,
            "arch":      arch,
            "regime":    regime,
            "pretrained": pretrained,
            "label":     label,
            # FT1 classification metrics
            "bce":        metrics.get("bce"),
            "accuracy":   metrics.get("accuracy"),
            "precision":  metrics.get("precision"),
            "recall":     metrics.get("recall"),
            "f1":         metrics.get("f1"),
            "auc_roc":    metrics.get("auc_roc"),
            "n_samples":  metrics.get("n_samples"),
            # FT3 regression metrics
            "overall_mse": metrics.get("overall_mse"),
            "Va_mse":      metrics.get("Va_mse"),
            "Va_mae":      metrics.get("Va_mae"),
            "Va_r2":       metrics.get("Va_r2"),
            "Vm_mse":      metrics.get("Vm_mse"),
            "Vm_mae":      metrics.get("Vm_mae"),
            "Vm_r2":       metrics.get("Vm_r2"),
            "n_nodes":     metrics.get("n_nodes"),
            # Training hyperparameters
            "num_epoch":    meta.get("num_epoch"),
            "learning_rate": meta.get("learning_rate"),
            "config_file":  meta.get("config_file"),
            # Training curve extremes
            "best_val_error": _best_val(curve),
            "final_train_error": _final_train(curve),
            "n_epochs_trained": _n_epochs(curve),
        }
        records.append(flat)

        # ── Detailed entry (full metrics + curve for plotting) ─────────────
        detailed.append({
            "log_name":    log_name,
            "meta":        meta,
            "test_metrics": {
                # Omit large arrays from the JSON summary; keep just scalars.
                k: v for k, v in metrics.items()
                if k not in ("probs", "labels", "preds_sample", "targets_sample")
            },
            "probs":          metrics.get("probs", []),
            "labels":         metrics.get("labels", []),
            "preds_sample":   metrics.get("preds_sample", []),
            "targets_sample": metrics.get("targets_sample", []),
            "training_curve": curve,
        })

    # ── Write outputs ──────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "ft1_ft3_summary.csv")
    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    print(f"[collect] CSV  → {csv_path}  ({len(records)} rows)")

    # JSON
    json_path = os.path.join(out_dir, "ft1_ft3_summary.json")
    with open(json_path, "w") as fh:
        json.dump(detailed, fh, indent=2, default=_json_default)
    print(f"[collect] JSON → {json_path}")

    # ── Print quick summary table ──────────────────────────────────────────
    _print_table(records)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_val(curve: dict) -> float | None:
    vals = curve.get("validate error", [])
    return min(v for _, v in vals) if vals else None


def _final_train(curve: dict) -> float | None:
    vals = curve.get("train error", [])
    return vals[-1][1] if vals else None


def _n_epochs(curve: dict) -> int | None:
    vals = curve.get("train error", [])
    return len(vals) if vals else None


def _json_default(obj):
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _print_table(records: list[dict]) -> None:
    """Print a compact ASCII summary table."""
    # Group by strategy
    from itertools import groupby
    for strategy, grp in groupby(
        sorted(records, key=lambda r: (r["strategy"], r["arch"], r["regime"])),
        key=lambda r: r["strategy"],
    ):
        rows = list(grp)
        print(f"\n{'─'*72}")
        print(f"  {strategy}")
        print(f"{'─'*72}")
        is_ft1 = "FT1" in strategy
        if is_ft1:
            hdr = f"  {'Label':<30} {'BCE':>7} {'Acc':>7} {'F1':>7} {'AUC':>7}"
        else:
            hdr = f"  {'Label':<30} {'Va_MSE':>9} {'Vm_MSE':>9} {'Va_R2':>7} {'Vm_R2':>7}"
        print(hdr)
        for r in rows:
            lbl = r["label"]
            if is_ft1:
                print(
                    f"  {lbl:<30} "
                    f"{_fmt(r['bce']):>7} {_fmt(r['accuracy']):>7} "
                    f"{_fmt(r['f1']):>7} {_fmt(r['auc_roc']):>7}"
                )
            else:
                print(
                    f"  {lbl:<30} "
                    f"{_fmt(r['Va_mse']):>9} {_fmt(r['Vm_mse']):>9} "
                    f"{_fmt(r['Va_r2']):>7} {_fmt(r['Vm_r2']):>7}"
                )
    print(f"\n{'─'*72}\n")


def _fmt(v) -> str:
    if v is None:
        return "   —"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate FT1 and FT3 results from the logs/ directory."
    )
    _ft_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.normpath(os.path.join(_ft_dir, "..", "..", ".."))
    parser.add_argument(
        "--logs_root",
        default=os.path.join(_repo_root, "logs"),
        help="Root directory containing log subdirectories (default: <repo>/logs)",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(_ft_dir, "results"),
        help="Directory to write summary files (default: examples/opf/finetune/results/)",
    )
    args = parser.parse_args()
    collect(args.logs_root, args.out_dir)


if __name__ == "__main__":
    main()
