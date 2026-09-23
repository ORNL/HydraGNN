#!/usr/bin/env python3
"""Convert solved PF/OPF records into PyG HeteroData and tabular metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from go_challenge1.discovery import discover_scenarios
from go_challenge1.graph_conversion import scenario_solution_to_heterodata
from go_challenge1.parser_utils import parse_scenario_bundle


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--solution-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=["pf", "opf", "both"], default="both")
    parser.add_argument("--format", choices=["pyg"], default="pyg")
    return parser.parse_args()


def _load_solution(solution_root: Path, scenario_id: str, task: str) -> dict | None:
    path = solution_root / task / f"{scenario_id}.{task}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["pf", "opf"] if args.task == "both" else [args.task]
    scenarios = discover_scenarios(Path(args.scenario_dir))

    metadata_rows = []

    for task in tasks:
        graphs_dir = out_dir / task / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        records_dir = Path(args.solution_dir) / task

        for files in scenarios:
            try:
                scenario = parse_scenario_bundle(files)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] failed to parse scenario bundle: {exc}")
                continue
            rec = _load_solution(records_dir.parent, files.uid, task)
            if rec is None:
                continue

            # Keep processed dataset strictly to validated labels.
            if not bool(rec.get("success", False)):
                continue
            if not bool(rec.get("label_valid", False)):
                continue

            solution = rec.get("solution", {})
            try:
                data = scenario_solution_to_heterodata(
                    scenario,
                    solution,
                    task_name=task,
                    sample_uid=files.uid,
                    grid_id=files.network_name,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {files.uid}: graph conversion failed: {exc}")
                continue

            graph_path = graphs_dir / f"{files.uid}.{task}.pt"
            torch.save(data, graph_path)

            metadata_rows.append(
                {
                    "task_name": task,
                    "task_id": int(data.task_id.item()),
                    "scenario_id": data.scenario_id,
                    "grid_id": data.grid_id,
                    "sample_id": data.sample_id,
                    "is_contingency": int(data.is_contingency.item()),
                    "contingency_id": data.contingency_id,
                    "solver_name": data.solver_name,
                    "solver_status": data.solver_status,
                    "graph_path": str(graph_path),
                }
            )

        if metadata_rows:
            task_rows = [r for r in metadata_rows if r["task_name"] == task]
            if task_rows:
                pd.DataFrame(task_rows).to_parquet(
                    out_dir / task / "metadata.parquet", index=False
                )
            with open(
                out_dir / task / "normalization.json", "w", encoding="utf-8"
            ) as fh:
                json.dump(
                    {"note": "Compute normalization on training split only."},
                    fh,
                    indent=2,
                )

    if metadata_rows:
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata_rows, fh, indent=2)


if __name__ == "__main__":
    main()
