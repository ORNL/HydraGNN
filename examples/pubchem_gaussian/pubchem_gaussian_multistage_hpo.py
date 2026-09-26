##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Resumable, balanced successive-halving HPO for PubChem Gaussian."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

try:
    from .pubchem_gaussian_hpo import (
        BASE_CONFIG_PATH,
        EXAMPLE_DIR,
        configure_trial,
        validation_losses,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pubchem_gaussian_hpo import (
        BASE_CONFIG_PATH,
        EXAMPLE_DIR,
        configure_trial,
        validation_losses,
    )


DEFAULT_MODELS = (
    "EGNN",
    "SchNet",
    "DimeNet",
    "MACE",
    "PAINN",
    "PNAEq",
    "AllScAIP",
    "UMA",
)
DEFAULT_STAGES = (
    {"name": "screen", "train_samples": 50_000, "epochs": 3, "keep": 128},
    {"name": "refine", "train_samples": 300_000, "epochs": 8, "keep": 32},
    {"name": "confirm", "train_samples": 1_000_000, "epochs": 15, "keep": 8},
    {"name": "full", "train_samples": 3_000_000, "epochs": 25, "keep": 3},
)
WEIGHT_CHOICES = (0.01, 0.1, 1.0, 10.0, 100.0)
PRIMARY_METRICS = ("Energy", "Forces", "Hessian")
DERIVED_PRIMARY_METRICS = ("Energy Per Atom",)
AUXILIARY_METRICS = (
    "mulliken_charges",
    "dipole_magnitude",
    "quadrupole_eigenvalues",
    "polarizability_eigenvalues",
    "frontier_orbital_energies",
    "rotational_constants",
    "thermochemistry",
)


def sample_candidates(count, seed, model_types=DEFAULT_MODELS):
    """Sample a model-family-balanced initial population."""
    rng = random.Random(seed)
    candidates = []
    for index in range(count):
        model_type = model_types[index % len(model_types)]
        candidates.append(
            {
                "id": f"candidate-{index:05d}",
                "parameters": {
                    "mpnn_type": model_type,
                    "use_equivariant_graph_transformer": rng.choice(("off", "on")),
                    "energy_weight": rng.choice(WEIGHT_CHOICES),
                    "force_weight": rng.choice(WEIGHT_CHOICES),
                    "hessian_weight": rng.choice(WEIGHT_CHOICES),
                    "num_conv_layers": rng.randint(2, 6),
                    "hidden_dim": rng.choice((64, 128, 256, 512)),
                    "global_attn_heads": rng.choice((1, 2, 4, 8)),
                    "equivariant_attn_num_hidden_layers": rng.randint(1, 4),
                    "equivariant_attn_feedforward_multiplier": rng.choice((1, 2, 4)),
                },
            }
        )
    rng.shuffle(candidates)
    return candidates


def normalized_score(losses, scales):
    """Average dimensionless metrics using scales fixed from the screen stage."""
    if losses is None:
        return math.inf
    values = [losses[name] / scales[name] for name in PRIMARY_METRICS]
    if not all(math.isfinite(value) for value in values):
        return math.inf
    return sum(values) / len(values)


def auxiliary_score(losses, scales):
    """Return a dimensionless auxiliary score, or infinity if labels are missing."""
    if losses is None:
        return math.inf
    names = [name for name in AUXILIARY_METRICS if name in scales]
    if not names or any(name not in losses for name in names):
        return math.inf
    values = [losses[name] / scales[name] for name in names]
    if not values or not all(math.isfinite(value) for value in values):
        return math.inf
    return sum(values) / len(values)


def rank_with_auxiliary_tiebreakers(results, scales, relative_tolerance=0.02):
    """Rank auxiliaries only inside the primary-best model's error envelope.

    A model is comparable only when every primary loss is no more than
    ``relative_tolerance`` above the candidate with the best aggregate primary
    score. This makes energy, force, and Hessian errors hard selection gates;
    auxiliary losses cannot compensate for a primary regression.
    """
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    for result in results:
        result["primary_score"] = normalized_score(result.get("losses"), scales)
        result["auxiliary_score"] = auxiliary_score(result.get("losses"), scales)
        result["score"] = result["primary_score"]
        result["primary_comparable"] = False
    successful = [
        result for result in results if math.isfinite(result["primary_score"])
    ]
    if not successful:
        return results
    anchor = min(successful, key=lambda result: result["primary_score"])
    anchor_losses = anchor["losses"]
    for result in successful:
        result["primary_comparable"] = all(
            result["losses"][name] <= anchor_losses[name] * (1.0 + relative_tolerance)
            for name in PRIMARY_METRICS
        )
    results.sort(
        key=lambda result: (
            0 if result["primary_comparable"] else 1,
            (
                result["auxiliary_score"]
                if result["primary_comparable"]
                else result["primary_score"]
            ),
            result["primary_score"],
            result["id"],
        )
    )
    return results


def _command(
    config_path,
    log_name,
    stage,
    subset_seed,
    nodes_per_trial,
    tasks_per_node,
    dataset_format="adios",
    ddstore=False,
    ddstore_width=None,
    shmem=False,
):
    command = []
    if os.environ.get("SLURM_JOB_ID"):
        command.extend(
            [
                "srun",
                "--exclusive",
                "--exact",
                "-N",
                str(nodes_per_trial),
                "-n",
                str(nodes_per_trial * tasks_per_node),
                f"--ntasks-per-node={tasks_per_node}",
                f"--gpus-per-node={tasks_per_node}",
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
            f"--log={log_name}",
            f"--num-epoch={stage['epochs']}",
            f"--num-train-samples={stage['train_samples']}",
            f"--num-val-samples={stage['val_samples']}",
            f"--num-test-samples={stage['test_samples']}",
            f"--subset-seed={subset_seed}",
            f"--{dataset_format}",
        ]
    )
    if ddstore:
        command.append("--ddstore")
    if ddstore_width is not None:
        command.append(f"--ddstore-width={ddstore_width}")
    if shmem:
        command.append("--shmem")
    return command


def run_candidate(
    candidate,
    base_config,
    stage,
    output_dir,
    subset_seed,
    nodes_per_trial,
    tasks_per_node,
    dataset_format,
    ddstore,
    ddstore_width,
    shmem,
):
    """Run one candidate as an exclusive multi-node DDP Slurm step."""
    trial_dir = output_dir / stage["name"] / candidate["id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = (trial_dir / "config.json").resolve()
    log_path = (trial_dir / "run.log").resolve()
    result_path = trial_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    config = configure_trial(base_config, candidate["parameters"])
    config_path.write_text(json.dumps(config, indent=4) + "\n")
    command = _command(
        config_path,
        f"{stage['name']}_{candidate['id']}",
        stage,
        subset_seed,
        nodes_per_trial,
        tasks_per_node,
        dataset_format,
        ddstore,
        ddstore_width,
        shmem,
    )
    return_code = -1
    try:
        with log_path.open("w") as stream:
            return_code = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        losses = validation_losses(log_path) if return_code == 0 else None
    except OSError:
        losses = None
    result = {
        **candidate,
        "stage": stage["name"],
        "return_code": return_code,
        "losses": losses,
        "config": str(config_path),
        "log": str(log_path),
    }
    result_path.write_text(json.dumps(result, indent=4) + "\n")
    return result


def _write_results(path, results):
    parameter_names = sorted(results[0]["parameters"]) if results else []
    fields = (
        [
            "id",
            "stage",
            "score",
            "primary_score",
            "auxiliary_score",
            "primary_comparable",
        ]
        + list(PRIMARY_METRICS)
        + list(AUXILIARY_METRICS)
        + parameter_names
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            losses = result.get("losses") or {}
            writer.writerow(
                {
                    "id": result["id"],
                    "stage": result["stage"],
                    "score": result["score"],
                    "primary_score": result["primary_score"],
                    "auxiliary_score": result["auxiliary_score"],
                    "primary_comparable": result["primary_comparable"],
                    **{
                        name: losses.get(name)
                        for name in PRIMARY_METRICS + AUXILIARY_METRICS
                    },
                    **result["parameters"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-candidates", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--nodes-per-trial", type=int, default=1)
    parser.add_argument("--tasks-per-node", type=int, default=1)
    parser.add_argument("--ddstore", action="store_true")
    parser.add_argument("--ddstore-width", type=int)
    parser.add_argument("--shmem", action="store_true")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--adios", action="store_const", dest="dataset_format", const="adios"
    )
    format_group.add_argument(
        "--pickle", action="store_const", dest="dataset_format", const="pickle"
    )
    parser.set_defaults(dataset_format="adios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="pubchem-multistage-hpo")
    parser.add_argument("--schedule", help="Optional JSON stage schedule")
    parser.add_argument(
        "--primary-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative regression in every primary metric for tie-breaking",
    )
    args = parser.parse_args()
    if args.nodes_per_trial <= 0 or args.tasks_per_node <= 0:
        parser.error("--nodes-per-trial and --tasks-per-node must be positive")
    if args.shmem and args.ddstore:
        parser.error("--shmem and --ddstore are mutually exclusive")
    if args.primary_tolerance < 0:
        parser.error("--primary-tolerance must be non-negative")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = list(DEFAULT_STAGES)
    if args.schedule:
        stages = json.loads(Path(args.schedule).read_text())
    for stage in stages:
        stage.setdefault("val_samples", min(50_000, stage["train_samples"] // 5))
        stage.setdefault("test_samples", stage["val_samples"])

    base_config = json.loads(BASE_CONFIG_PATH.read_text())
    candidates = sample_candidates(args.initial_candidates, args.seed)
    scales = None
    for stage_index, stage in enumerate(stages):
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    run_candidate,
                    candidate,
                    base_config,
                    stage,
                    output_dir,
                    args.seed,
                    args.nodes_per_trial,
                    args.tasks_per_node,
                    args.dataset_format,
                    args.ddstore,
                    args.ddstore_width,
                    args.shmem,
                )
                for candidate in candidates
            ]
            results = [future.result() for future in as_completed(futures)]

        if scales is None:
            finite_losses = [result["losses"] for result in results if result["losses"]]
            if not finite_losses:
                raise RuntimeError("No screen-stage trial completed with finite losses")
            metric_names = PRIMARY_METRICS + AUXILIARY_METRICS
            scales = {
                name: max(
                    statistics.median(
                        loss[name] for loss in finite_losses if name in loss
                    ),
                    1.0e-12,
                )
                for name in metric_names
                if any(name in loss for loss in finite_losses)
            }
            (output_dir / "objective_scales.json").write_text(
                json.dumps(scales, indent=4) + "\n"
            )
        tolerance = float(stage.get("primary_tolerance", args.primary_tolerance))
        rank_with_auxiliary_tiebreakers(results, scales, tolerance)
        _write_results(output_dir / f"{stage['name']}-results.csv", results)
        successful = [
            result for result in results if math.isfinite(result["primary_score"])
        ]
        if not successful:
            raise RuntimeError(f"No successful candidates in stage {stage['name']}")
        keep = min(int(stage["keep"]), len(successful))
        candidates = [
            {"id": result["id"], "parameters": result["parameters"]}
            for result in successful[:keep]
        ]
        if stage_index + 1 == len(stages):
            (output_dir / "finalists.json").write_text(
                json.dumps(successful[:keep], indent=4) + "\n"
            )


if __name__ == "__main__":
    main()
