#!/usr/bin/env python3
"""Generate base-case AC-OPF labels for discovered Challenge 1 scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go_challenge1.discovery import discover_scenarios
from go_challenge1.opf_generation import generate_opf_sample
from go_challenge1.parser_utils import parse_scenario_bundle
from go_challenge1.validation import ValidationTolerances, validate_solution


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--solver", choices=["pandapower", "powermodels"], default="pandapower")
    parser.add_argument("--powermodels-timeout", type=float, default=600.0)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--power-balance-tol", type=float, default=1e-5)
    parser.add_argument("--voltage-tol", type=float, default=1e-6)
    parser.add_argument("--branch-limit-tol", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(Path(args.input_dir))
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    tol = ValidationTolerances(
        power_balance_tolerance=args.power_balance_tol,
        voltage_tolerance=args.voltage_tol,
        branch_limit_tolerance=args.branch_limit_tol,
    )

    summary = []
    for files in scenarios:
        try:
            scenario = parse_scenario_bundle(files)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] failed to parse scenario bundle: {exc}")
            summary.append({
                "scenario_id": files.uid,
                "task_name": "opf",
                "success": False,
                "label_valid": False,
                "validation_messages": [f"parse_error:{type(exc).__name__}:{exc}"],
            })
            continue

        try:
            solution = generate_opf_sample(
                scenario,
                solver=args.solver,
                raw_path=str(files.raw) if files.raw is not None else None,
                timeout=args.powermodels_timeout,
            )
            report = validate_solution(scenario, solution, tol)

            rec = {
                "scenario_id": scenario.scenario_id,
                "sample_uid": files.uid,
                "grid_id": files.network_name,
                "dataset_partition": files.dataset_partition,
                "network_name": scenario.network_name,
                "task_name": "opf",
                "solver_name": solution.get("solver_name", "unknown"),
                "solver_status": solution.get("termination_status", "unknown"),
                "success": bool(solution.get("success", False)),
                "label_valid": report.ok,
                "validation_messages": report.messages,
                "solution": solution,
            }

            with open(out_dir / f"{files.uid}.opf.json", "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
            summary.append({k: v for k, v in rec.items() if k != "solution"})
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {scenario.scenario_id}: OPF generation failed: {exc}")
            summary.append({
                "scenario_id": scenario.scenario_id,
                "sample_uid": files.uid,
                "grid_id": files.network_name,
                "network_name": scenario.network_name,
                "task_name": "opf",
                "success": False,
                "label_valid": False,
                "validation_messages": [f"generation_error:{type(exc).__name__}:{exc}"],
            })
            continue

    with open(out_dir / "summary.opf.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
