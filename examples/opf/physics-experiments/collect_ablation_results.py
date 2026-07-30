#!/usr/bin/env python3
"""Collect OPF ablation results from HydraGNN log directories.

This script scans training ``run.log`` files, optional ``config.json`` files,
and optional inference ``test_metrics.json`` files, then writes one flat CSV
and one structured JSON summary.

Examples:
    python physics-experiments/collect_ablation_results.py
    python physics-experiments/collect_ablation_results.py --logs_root logs --out_dir physics-experiments/results
    python physics-experiments/collect_ablation_results.py --run logs/heat_attr --run logs/heat_attr_AL
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*"
    r"Train Loss:\s*(?P<train>[0-9eE+\-.]+),\s*"
    r"Val Loss:\s*(?P<val>[0-9eE+\-.]+),\s*"
    r"Test Loss:\s*(?P<test>[0-9eE+\-.]+)"
)

KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[0-9eE+\-.]+)")
SPLIT_RE = re.compile(r"\bsplit=(?P<split>train|val|test)\b")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_run_log(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    epochs: list[dict[str, Any]] = []
    breakdowns: list[dict[str, Any]] = []
    epoch_times: list[dict[str, Any]] = []
    training_time: dict[str, Any] = {}

    if not path.is_file():
        return epochs, breakdowns, epoch_times, training_time

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(
                    {
                        "epoch": int(m.group("epoch")),
                        "train_loss": float(m.group("train")),
                        "val_loss": float(m.group("val")),
                        "test_loss": float(m.group("test")),
                    }
                )
                continue

            if "LossBreakdown" in line:
                row: dict[str, Any] = {}
                split_match = SPLIT_RE.search(line)
                if split_match:
                    row["split"] = split_match.group("split")
                for kv in KV_RE.finditer(line):
                    key = kv.group("key")
                    value = float(kv.group("value"))
                    row[key] = int(value) if key == "epoch" else value
                if row:
                    breakdowns.append(row)
                continue

            if "EpochTime" in line:
                row = {}
                for kv in KV_RE.finditer(line):
                    key = kv.group("key")
                    value = float(kv.group("value"))
                    row[key] = int(value) if key == "epoch" else value
                if row:
                    epoch_times.append(row)
                continue

            if "TrainingTime" in line:
                row = {}
                for kv in KV_RE.finditer(line):
                    key = kv.group("key")
                    value = float(kv.group("value"))
                    row[key] = int(value) if key == "epochs" else value
                if row:
                    training_time = row

    return epochs, breakdowns, epoch_times, training_time


def _best_epoch(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    if not epochs:
        return {}
    return min(epochs, key=lambda row: row["val_loss"])


def _last_epoch(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    return epochs[-1] if epochs else {}


def _nearest_breakdown(
    breakdowns: list[dict[str, Any]], epoch: int | None, split: str = "val"
) -> dict[str, Any]:
    if epoch is None or not breakdowns:
        return {}
    split_rows = [row for row in breakdowns if row.get("split") == split]
    # Older logs did not include a split, so retain backward compatibility.
    candidates = split_rows or breakdowns
    exact = [row for row in candidates if row.get("epoch") == epoch]
    if exact:
        return exact[-1]
    return min(
        candidates, key=lambda row: abs(int(row.get("epoch", -10**9)) - epoch)
    )


def _final_breakdown(
    breakdowns: list[dict[str, Any]], split: str = "val"
) -> dict[str, Any]:
    split_rows = [row for row in breakdowns if row.get("split") == split]
    candidates = split_rows or breakdowns
    return candidates[-1] if candidates else {}


def _nearest_epoch_time(
    epoch_times: list[dict[str, Any]], epoch: int | None
) -> dict[str, Any]:
    if epoch is None or not epoch_times:
        return {}
    exact = [row for row in epoch_times if row.get("epoch") == epoch]
    if exact:
        return exact[-1]
    return min(epoch_times, key=lambda row: abs(int(row.get("epoch", -10**9)) - epoch))


def _config_meta(config: dict[str, Any]) -> dict[str, Any]:
    nn = config.get("NeuralNetwork", {})
    arch = nn.get("Architecture", {})
    training = nn.get("Training", {})
    domain = training.get("DomainLoss", {})
    edge_dim = arch.get("edge_dim")

    if isinstance(edge_dim, dict):
        edge_attr_enabled = bool(edge_dim)
        edge_dim_repr = json.dumps(edge_dim, sort_keys=True)
    else:
        edge_attr_enabled = edge_dim is not None
        edge_dim_repr = str(edge_dim) if edge_dim is not None else None

    return {
        "mpnn_type": arch.get("mpnn_type"),
        "hidden_dim": arch.get("hidden_dim"),
        "num_conv_layers": arch.get("num_conv_layers"),
        "edge_attr_enabled": edge_attr_enabled,
        "edge_dim": edge_dim_repr,
        "domain_loss_enabled": bool(domain.get("enabled", False)),
        "domain_loss_mode": domain.get("mode", "none" if not domain else "static"),
        "al_rho": domain.get("al_rho"),
        "al_mu_max": domain.get("al_mu_max"),
        "voltage_bound_weight": domain.get("voltage_bound_weight"),
        "angle_diff_weight": domain.get("angle_diff_weight"),
        "line_flow_weight": domain.get("line_flow_weight"),
        "num_epoch_config": training.get("num_epoch"),
        "batch_size": training.get("batch_size"),
        "learning_rate": (
            training.get("Optimizer", {}).get("learning_rate")
            if isinstance(training.get("Optimizer"), dict)
            else None
        ),
    }


def _infer_meta_from_name(log_name: str) -> dict[str, Any]:
    name = log_name.lower()
    meta: dict[str, Any] = {}
    for mpnn in ("heteroheat", "heterosage", "heterogat", "heterogin"):
        if mpnn in name or mpnn.replace("hetero", "") in name:
            meta["mpnn_type"] = {
                "heteroheat": "HeteroHEAT",
                "heterosage": "HeteroSAGE",
                "heterogat": "HeteroGAT",
                "heterogin": "HeteroGIN",
            }[mpnn]
            break
    if "no_attr" in name:
        meta["edge_attr_enabled"] = False
    elif "attr" in name:
        meta["edge_attr_enabled"] = True
    if "al" in name:
        meta["domain_loss_mode"] = "augmented_lagrangian"
        meta["domain_loss_enabled"] = True
    elif "static" in name or "physics" in name:
        meta["domain_loss_mode"] = "static"
        meta["domain_loss_enabled"] = True
    elif "no_physics" in name:
        meta["domain_loss_mode"] = "none"
        meta["domain_loss_enabled"] = False
    return meta


def _flatten_inference_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "inference_test_error": _parse_float(metrics.get("test_error")),
    }

    task_errors = metrics.get("task_errors")
    if isinstance(task_errors, list):
        for idx, value in enumerate(task_errors):
            flat[f"inference_task_{idx}_error"] = _parse_float(value)

    for mae in metrics.get("mae", []) or []:
        if not isinstance(mae, dict):
            continue
        quantity = str(mae.get("quantity", "output"))
        flat[f"{quantity}_mae_overall"] = _parse_float(mae.get("mae_overall"))
        per_dim = mae.get("mae_per_dim")
        if isinstance(per_dim, list):
            for idx, value in enumerate(per_dim):
                flat[f"{quantity}_dim{idx}_mae"] = _parse_float(value)

    for diag in metrics.get("diagnostics", []) or []:
        if not isinstance(diag, dict):
            continue
        quantity = str(diag.get("quantity", "output"))
        for key in (
            "bias_per_dim",
            "abs_error_p50_per_dim",
            "abs_error_p90_per_dim",
            "abs_error_p99_per_dim",
            "high_true_bias_per_dim",
        ):
            vals = diag.get(key)
            if isinstance(vals, list):
                stem = key.removesuffix("_per_dim")
                for idx, value in enumerate(vals):
                    flat[f"{quantity}_dim{idx}_{stem}"] = _parse_float(value)

    return flat


def _read_optional_metrics(run_dir: Path) -> dict[str, Any]:
    candidates = [
        run_dir / "test_metrics.json",
        run_dir / "inference" / "test_metrics.json",
        run_dir / "results.json",
    ]
    for path in candidates:
        if path.is_file():
            payload = _load_json(path)
            if "test_metrics" in payload and isinstance(payload["test_metrics"], dict):
                return payload["test_metrics"]
            return payload
    return {}


def _collect_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    log_name = run_dir.name
    run_log = run_dir / "run.log"
    epochs, breakdowns, epoch_times, training_time = _parse_run_log(run_log)
    best = _best_epoch(epochs)
    last = _last_epoch(epochs)
    best_breakdown = _nearest_breakdown(breakdowns, best.get("epoch"))
    final_breakdown = _final_breakdown(breakdowns)
    best_epoch_time = _nearest_epoch_time(epoch_times, best.get("epoch"))
    final_epoch_time = _nearest_epoch_time(epoch_times, last.get("epoch"))

    config = _load_json(run_dir / "config.json")
    meta = _config_meta(config) if config else {}
    inferred = _infer_meta_from_name(log_name)
    for key, value in inferred.items():
        meta.setdefault(key, value)

    metrics = _read_optional_metrics(run_dir)

    row: dict[str, Any] = {
        "log_name": log_name,
        "log_dir": str(run_dir),
        "has_run_log": run_log.is_file(),
        "num_epochs_logged": len(epochs),
        "best_epoch": best.get("epoch"),
        "best_train_loss": best.get("train_loss"),
        "best_val_loss": best.get("val_loss"),
        "best_test_loss": best.get("test_loss"),
        "final_epoch": last.get("epoch"),
        "final_train_loss": last.get("train_loss"),
        "final_val_loss": last.get("val_loss"),
        "final_test_loss": last.get("test_loss"),
        "best_epoch_seconds": best_epoch_time.get("seconds"),
        "final_epoch_seconds": final_epoch_time.get("seconds"),
        "total_train_seconds": training_time.get("total_seconds"),
        "avg_epoch_seconds": training_time.get("avg_epoch_seconds"),
        "timed_epochs": training_time.get("epochs"),
    }
    row.update(meta)

    for prefix, source in (
        ("best", best_breakdown),
        ("final", final_breakdown),
    ):
        for key, value in source.items():
            if key == "epoch":
                continue
            row[f"{prefix}_{key}"] = value

    row.update(_flatten_inference_metrics(metrics))

    detail = {
        "log_name": log_name,
        "log_dir": str(run_dir),
        "meta": meta,
        "epochs": epochs,
        "loss_breakdowns": breakdowns,
        "epoch_times": epoch_times,
        "training_time": training_time,
        "inference_metrics": metrics,
        "summary": row,
    }
    return row, detail


def _discover_runs(logs_root: Path) -> list[Path]:
    run_logs = sorted(glob.glob(str(logs_root / "**" / "run.log"), recursive=True))
    return [Path(path).parent for path in run_logs]


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "log_name",
        "log_dir",
        "mpnn_type",
        "edge_attr_enabled",
        "domain_loss_enabled",
        "domain_loss_mode",
        "al_rho",
        "hidden_dim",
        "num_conv_layers",
        "batch_size",
        "learning_rate",
        "num_epochs_logged",
        "best_epoch",
        "best_val_loss",
        "best_test_loss",
        "final_val_loss",
        "final_test_loss",
        "best_epoch_seconds",
        "final_epoch_seconds",
        "avg_epoch_seconds",
        "total_train_seconds",
        "best_data_driven_mse",
        "best_physics_penalty_total",
        "best_raw_voltage_bound",
        "best_raw_ac_angle_diff",
        "best_raw_tr_angle_diff",
        "best_raw_ac_line_flow",
        "best_raw_tr_line_flow",
        "best_raw_ac_apparent_flow",
        "best_raw_tr_apparent_flow",
        "best_mu_voltage_bound",
        "best_mu_ac_angle_diff",
        "best_mu_tr_angle_diff",
        "best_mu_ac_line_flow",
        "best_mu_tr_line_flow",
        "best_mu_ac_apparent_flow",
        "best_mu_tr_apparent_flow",
        "inference_test_error",
        "bus_solution_dim0_mae",
        "bus_solution_dim1_mae",
    ]
    all_keys = set().union(*(row.keys() for row in rows)) if rows else set()
    ordered = [key for key in preferred if key in all_keys]
    ordered.extend(sorted(all_keys - set(ordered)))
    return ordered


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    if rows:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text("", encoding="utf-8")


def _write_outputs(rows: list[dict[str, Any]], details: list[dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "opf_ablation_summary.csv"
    json_path = out_dir / "opf_ablation_summary.json"
    epochs_path = out_dir / "opf_ablation_epochs.csv"
    epoch_times_path = out_dir / "opf_ablation_epoch_times.csv"
    breakdowns_path = out_dir / "opf_ablation_loss_breakdowns.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(details, fh, indent=2)

    epoch_rows = []
    epoch_time_rows = []
    breakdown_rows = []
    for detail in details:
        log_name = detail["log_name"]
        log_dir = detail["log_dir"]
        for row in detail["epochs"]:
            epoch_rows.append({"log_name": log_name, "log_dir": log_dir, **row})
        for row in detail["epoch_times"]:
            epoch_time_rows.append({"log_name": log_name, "log_dir": log_dir, **row})
        for row in detail["loss_breakdowns"]:
            breakdown_rows.append({"log_name": log_name, "log_dir": log_dir, **row})

    _write_csv(epochs_path, epoch_rows)
    _write_csv(epoch_times_path, epoch_time_rows)
    _write_csv(breakdowns_path, breakdown_rows)

    print(f"[collect] CSV  -> {csv_path} ({len(rows)} rows)")
    print(f"[collect] JSON -> {json_path}")
    print(f"[collect] epochs CSV      -> {epochs_path} ({len(epoch_rows)} rows)")
    print(f"[collect] epoch times CSV -> {epoch_times_path} ({len(epoch_time_rows)} rows)")
    print(f"[collect] breakdowns CSV  -> {breakdowns_path} ({len(breakdown_rows)} rows)")


def _fmt(value: Any, width: int = 11) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    try:
        return f"{float(value):>{width}.6g}"
    except (TypeError, ValueError):
        text = str(value)
        return text[:width].rjust(width)


def _print_table(rows: list[dict[str, Any]]):
    if not rows:
        return
    print()
    print(
        f"{'run':<34} {'arch':<11} {'edge':<5} {'loss':<22} "
        f"{'best_val':>11} {'best_test':>11} {'phys':>11} {'sec/ep':>11}"
    )
    print("-" * 124)
    for row in sorted(
        rows,
        key=lambda r: (
            str(r.get("mpnn_type")),
            str(r.get("edge_attr_enabled")),
            str(r.get("domain_loss_mode")),
            str(r.get("log_name")),
        ),
    ):
        loss = row.get("domain_loss_mode", "unknown")
        edge = "yes" if row.get("edge_attr_enabled") else "no"
        print(
            f"{str(row.get('log_name')):<34.34} "
            f"{str(row.get('mpnn_type', '-')):<11.11} "
            f"{edge:<5} "
            f"{str(loss):<22.22} "
            f"{_fmt(row.get('best_val_loss'))} "
            f"{_fmt(row.get('best_test_loss'))} "
            f"{_fmt(row.get('best_physics_penalty_total'))} "
            f"{_fmt(row.get('avg_epoch_seconds'))}"
        )
    print()


def collect(run_dirs: list[Path], logs_root: Path, out_dir: Path):
    if not run_dirs:
        run_dirs = _discover_runs(logs_root)
    run_dirs = sorted({path.resolve() for path in run_dirs})
    resolved_logs_root = logs_root.resolve()

    if not run_dirs:
        print(f"[collect] No run.log files found under {logs_root}")
        return

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row, detail = _collect_run(run_dir)
        try:
            relative_run_dir = run_dir.relative_to(resolved_logs_root)
            display_dir = Path(resolved_logs_root.name) / relative_run_dir
        except ValueError:
            # Avoid embedding machine- or user-specific absolute paths when
            # collecting an explicitly supplied run outside --logs_root.
            display_dir = Path(run_dir.name)
        row["log_dir"] = str(display_dir)
        detail["log_dir"] = str(display_dir)
        detail["summary"]["log_dir"] = str(display_dir)
        rows.append(row)
        details.append(detail)

    _write_outputs(rows, details, out_dir)
    _print_table(rows)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Collect OPF ablation results from HydraGNN logs."
    )
    script_dir = Path(__file__).resolve().parent
    opf_dir = script_dir.parent
    parser.add_argument(
        "--logs_root",
        type=Path,
        default=opf_dir / "logs",
        help="Directory to recursively scan for run.log files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=script_dir / "results",
        help="Directory for opf_ablation_summary.csv/json.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        default=[],
        help="Specific run directory to collect. May be repeated.",
    )
    args = parser.parse_args(argv)

    collect(args.run, args.logs_root, args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
