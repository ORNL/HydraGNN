##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Run comparable preliminary PubChem training across selected MPNNs."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .pubchem_gaussian_hpo import configure_trial, validation_losses
except ImportError:
    from pubchem_gaussian_hpo import configure_trial, validation_losses


EXAMPLE_DIR = Path(__file__).resolve().parent
BASE_CONFIG_PATH = EXAMPLE_DIR / "pubchem_gaussian.json"
DEFAULT_MODELS = ("PAINN", "MACE", "SchNet", "DimeNet", "UMA", "AllScAIP")
TRANSIENT_IO_ERRORS = ("Input/output error", "cannot read file data")


def model_parameters(mpnn_type):
    return {
        "mpnn_type": mpnn_type,
        "num_conv_layers": 4,
        "hidden_dim": 128,
        "equivariant_attn_num_hidden_layers": 1,
        "equivariant_attn_feedforward_multiplier": 2,
        "energy_weight": 1.0,
        "force_weight": 1.0,
        "hessian_weight": 1.0,
        "use_equivariant_graph_transformer": "off",
        "global_attn_heads": 1,
    }


def trial_command(args, config_path, log_name):
    command = []
    if os.environ.get("SLURM_JOB_ID"):
        command.extend(
            [
                "srun",
                "--exclusive",
                "--exact",
                "-N1",
                "-n1",
                "--ntasks-per-node=1",
                "--cpus-per-task=7",
                "--gpus-per-node=1",
                "--gpus-per-task=1",
                "--gpu-bind=closest",
                "--kill-on-bad-exit=1",
            ]
        )
    command.extend(
        [
            sys.executable,
            "-u",
            str(EXAMPLE_DIR / "train.py"),
            f"--inputfile={config_path}",
            f"--dataset-path={args.dataset_path}",
            "--adios",
            f"--num-epoch={args.epochs}",
            f"--num-train-samples={args.train_samples}",
            f"--num-val-samples={args.val_samples}",
            f"--num-test-samples={args.test_samples}",
            f"--subset-seed={args.subset_seed}",
            f"--log={log_name}",
        ]
    )
    return command


def run_model(mpnn_type, args, base_config, output_dir):
    trial_dir = output_dir / mpnn_type.lower()
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = (trial_dir / "config.json").resolve()
    log_path = (trial_dir / "run.log").resolve()
    result_path = trial_dir / "result.json"

    parameters = model_parameters(mpnn_type)
    config = configure_trial(base_config, parameters)
    config_path.write_text(json.dumps(config, indent=4) + "\n")
    command = trial_command(args, config_path, f"mpnn_benchmark_{mpnn_type.lower()}")

    start = time.monotonic()
    return_code = -1
    attempts = 0
    for attempts in range(1, 3):
        with log_path.open("a") as stream:
            stream.write(f"\n===== attempt {attempts} =====\n")
            stream.flush()
            return_code = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        if return_code == 0 or not any(
            message in log_path.read_text(errors="replace")
            for message in TRANSIENT_IO_ERRORS
        ):
            break
    elapsed_seconds = time.monotonic() - start
    losses = validation_losses(log_path) if return_code == 0 else None
    result = {
        "model": mpnn_type,
        "return_code": return_code,
        "attempts": attempts,
        "elapsed_seconds": elapsed_seconds,
        "losses": losses,
        "config": str(config_path),
        "log": str(log_path),
    }
    result_path.write_text(json.dumps(result, indent=4) + "\n")
    return result


def write_summary(output_dir, results):
    (output_dir / "results.json").write_text(json.dumps(results, indent=4) + "\n")
    metric_names = sorted(
        {name for result in results for name in (result.get("losses") or {})}
    )
    with (output_dir / "results.csv").open("w", newline="") as stream:
        fields = ["model", "return_code", "elapsed_seconds", *metric_names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result["model"],
                    "return_code": result["return_code"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    **(result.get("losses") or {}),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(models) - set(DEFAULT_MODELS)
    if unknown:
        parser.error("unsupported models: " + ", ".join(sorted(unknown)))
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(BASE_CONFIG_PATH.read_text())
    results = []
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(models))) as pool:
        futures = {
            pool.submit(run_model, model, args, base_config, args.output_dir): model
            for model in models
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['model']}: return_code={result['return_code']} "
                f"elapsed_seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )

    results.sort(key=lambda result: models.index(result["model"]))
    write_summary(args.output_dir, results)
    if any(result["return_code"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()