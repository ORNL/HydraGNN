#!/usr/bin/env python3
"""Inspect discovered Challenge 1 scenarios and summarize parsed counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go_challenge1.discovery import discover_scenarios
from go_challenge1.parser_utils import parse_scenario_bundle


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--scenario", default=None, help="Optional scenario name filter"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of table"
    )
    return parser.parse_args()


def _row(files, scenario):
    return {
        "scenario_identifier": files.scenario_id,
        "network_name": files.network_name,
        "path": str(files.path),
        "has_raw": files.raw is not None,
        "has_rop": files.rop is not None,
        "has_inl": files.inl is not None,
        "has_con": files.con is not None,
        "file_sizes": {
            "raw": files.raw.stat().st_size if files.raw else 0,
            "rop": files.rop.stat().st_size if files.rop else 0,
            "inl": files.inl.stat().st_size if files.inl else 0,
            "con": files.con.stat().st_size if files.con else 0,
        },
        "num_buses": len(scenario.buses),
        "num_loads": len(scenario.loads),
        "num_generators": len(scenario.generators),
        "num_branches": len(scenario.branches),
        "num_transformers": len(scenario.transformers),
        "num_shunts": len(scenario.fixed_shunts) + len(scenario.switched_shunts),
        "num_contingencies": len(scenario.contingencies),
    }


def main():
    args = parse_args()
    scenarios = discover_scenarios(Path(args.input_dir))
    if args.scenario:
        scenarios = [
            s
            for s in scenarios
            if s.scenario_id == args.scenario or s.network_name == args.scenario
        ]

    rows = []
    for files in scenarios:
        try:
            scenario = parse_scenario_bundle(files)
            rows.append(_row(files, scenario))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "scenario_identifier": files.scenario_id,
                    "network_name": files.network_name,
                    "path": str(files.path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
