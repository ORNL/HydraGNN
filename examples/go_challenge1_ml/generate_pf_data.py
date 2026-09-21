#!/usr/bin/env python3
"""Generate base-case AC-PF labels for discovered Challenge 1 scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go_challenge1.discovery import discover_scenarios
from go_challenge1.parser_utils import parse_scenario_bundle
from go_challenge1.pf_generation import generate_pf_sample
from go_challenge1.validation import ValidationTolerances, validate_solution


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--opf-solution-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-opf-controls", action="store_true")
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--power-balance-tol", type=float, default=1e-5)
    parser.add_argument("--voltage-tol", type=float, default=1e-6)
    parser.add_argument("--branch-limit-tol", type=float, default=1e-5)
    return parser.parse_args()


def _controls_from_opf(opf_record_path: Path, scenario):
    with open(opf_record_path, "r", encoding="utf-8") as fh:
        rec = json.load(fh)
    gen_sol = rec.get("solution", {}).get("generator_solution", {})
    controls = {"generator_setpoints": {}}
    for idx_str, row in gen_sol.items():
        idx = int(idx_str)
        if idx >= len(scenario.generators):
            continue
        g = scenario.generators[idx]
        gen_key = f"{g.bus_id}:{g.generator_id}"
        controls["generator_setpoints"][gen_key] = {
            "p_mw": row.get("p_mw", 0.0),
            "vm_pu": row.get("vm_pu", 1.0),
        }
    return controls


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
                "task_name": "pf",
                "success": False,
                "label_valid": False,
                "validation_messages": [f"parse_error:{type(exc).__name__}:{exc}"],
            })
            continue

        try:
            controls = None
            opf_output_controls = None
            opf_output_state = None
            if args.use_opf_controls:
                if args.opf_solution_dir is None:
                    raise ValueError("--opf-solution-dir is required when --use-opf-controls is enabled")
                opf_path = Path(args.opf_solution_dir) / f"{files.uid}.opf.json"
                if opf_path.exists():
                    with open(opf_path, "r", encoding="utf-8") as fh:
                        opf_rec = json.load(fh)
                    controls = _controls_from_opf(opf_path, scenario)
                    opf_output_controls = controls
                    opf_output_state = opf_rec.get("solution", {})

            solution = generate_pf_sample(scenario, controls=controls)
            report = validate_solution(scenario, solution, tol)

            rec = {
                "scenario_id": scenario.scenario_id,
                "sample_uid": files.uid,
                "grid_id": files.network_name,
                "dataset_partition": files.dataset_partition,
                "network_name": scenario.network_name,
                "task_name": "pf",
                "is_paired_pf_opf": bool(controls),
                "solver_name": solution.get("solver_name", "unknown"),
                "solver_status": solution.get("termination_status", "unknown"),
                "success": bool(solution.get("success", False)),
                "label_valid": report.ok,
                "validation_messages": report.messages,
                "used_opf_controls": bool(controls),
                "opf_output_controls": opf_output_controls,
                "opf_output_state": opf_output_state,
                "pf_input_using_opf_controls": controls,
                "pf_verified_state": solution,
                "solution": solution,
            }

            with open(out_dir / f"{files.uid}.pf.json", "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
            summary.append({k: v for k, v in rec.items() if k != "solution"})
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {scenario.scenario_id}: PF generation failed: {exc}")
            summary.append({
                "scenario_id": scenario.scenario_id,
                "sample_uid": files.uid,
                "grid_id": files.network_name,
                "network_name": scenario.network_name,
                "task_name": "pf",
                "success": False,
                "label_valid": False,
                "validation_messages": [f"generation_error:{type(exc).__name__}:{exc}"],
            })
            continue

    with open(out_dir / "summary.pf.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
