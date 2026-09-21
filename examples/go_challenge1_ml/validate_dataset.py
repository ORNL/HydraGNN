#!/usr/bin/env python3
"""Validate processed dataset metadata and serialized graph files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args()


def _validate_task_dir(task_dir: Path):
    metadata_path = task_dir / "metadata.parquet"
    if not metadata_path.exists():
        return []

    df = pd.read_parquet(metadata_path)
    errors = []
    for _, row in df.iterrows():
        graph_path = Path(row["graph_path"])
        if not graph_path.exists():
            errors.append(f"Missing graph file: {graph_path}")
            continue
        # PyTorch 2.6 defaults to weights_only=True, which blocks HeteroData unpickling.
        data = torch.load(graph_path, weights_only=False)
        if int(data.task_id.item()) != int(row["task_id"]):
            errors.append(f"task_id mismatch for {graph_path}")
        if str(data.scenario_id) != str(row["scenario_id"]):
            errors.append(f"scenario_id mismatch for {graph_path}")
    return errors


def main():
    args = parse_args()
    root = Path(args.input_dir)

    errors = []
    for task_name in ["pf", "opf", "contingency_pf", "scopf"]:
        task_dir = root / task_name
        if task_dir.exists():
            errors.extend(_validate_task_dir(task_dir))

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)

    print("Dataset validation passed")


if __name__ == "__main__":
    main()
