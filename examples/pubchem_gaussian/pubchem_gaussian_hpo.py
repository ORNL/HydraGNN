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
"""DeepHyper search for the PubChem Gaussian interatomic potential."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


EXAMPLE_DIR = Path(__file__).resolve().parent
BASE_CONFIG_PATH = EXAMPLE_DIR / "pubchem_gaussian.json"
LOSS_PATTERN = re.compile(
    r"^(?:\d+:\s*)?(Energy|Forces|Hessian) Train Loss: "
    r"[-+\d.eE]+, Val Loss: ([-+\d.eE]+), Test Loss: [-+\d.eE]+$"
)


def configure_trial(base_config, parameters):
    """Return a trial-local config populated from DeepHyper parameters."""
    config = deepcopy(base_config)
    architecture = config["NeuralNetwork"]["Architecture"]

    architecture["mpnn_type"] = parameters["mpnn_type"]
    architecture["num_conv_layers"] = int(parameters["num_conv_layers"])
    architecture["hidden_dim"] = int(parameters["hidden_dim"])
    architecture["equivariant_attn_num_hidden_layers"] = int(
        parameters["equivariant_attn_num_hidden_layers"]
    )
    architecture["equivariant_attn_feedforward_multiplier"] = int(
        parameters["equivariant_attn_feedforward_multiplier"]
    )
    architecture["energy_weight"] = float(parameters["energy_weight"])
    architecture["force_weight"] = float(parameters["force_weight"])
    architecture["hessian_weight"] = float(parameters["hessian_weight"])

    supported_transformer_models = {"SchNet", "DimeNet", "MACE", "PAINN", "PNAEq"}
    use_transformer = (
        parameters["use_equivariant_graph_transformer"] == "on"
        and architecture["mpnn_type"] in supported_transformer_models
    )
    if use_transformer:
        architecture["global_attn_engine"] = "EquivariantTransformer"
        architecture["global_attn_type"] = ""
        architecture["global_attn_heads"] = int(parameters["global_attn_heads"])
        scalar_only = architecture["mpnn_type"] in {"SchNet", "DimeNet"}
        architecture["equivariant_attn_allow_scalar_only"] = scalar_only
        architecture["equivariant_attn_require_tensor_coupling"] = not scalar_only
    else:
        architecture["global_attn_engine"] = ""
        architecture["global_attn_type"] = ""
        architecture["global_attn_heads"] = 1

    if architecture["mpnn_type"] == "AllScAIP":
        architecture["equivariance"] = False
    elif architecture["mpnn_type"] == "UMA":
        architecture["equivariance"] = True
        architecture["max_ell"] = 2
        architecture["uma_mmax"] = 2
    elif architecture["mpnn_type"] == "MACE":
        architecture["max_ell"] = 2
        architecture["node_max_ell"] = 2
        architecture["equivariant_attn_lmax"] = 2

    return config


def validation_objective(log_path):
    """Return the negative mean of the latest energy, force, and Hessian losses."""
    current = {}
    latest = None
    for line in Path(log_path).read_text(errors="replace").splitlines():
        match = LOSS_PATTERN.match(line.strip())
        if match is None:
            continue
        current[match.group(1)] = float(match.group(2))
        if match.group(1) == "Hessian" and set(current) == {
            "Energy",
            "Forces",
            "Hessian",
        }:
            latest = sum(current.values()) / len(current)
            current = {}
    return -latest if latest is not None and math.isfinite(latest) else -math.inf


def _trial_command(config_path, log_name, nodes):
    python_script = EXAMPLE_DIR / "train.py"
    command = []
    if nodes:
        tasks_per_node = int(os.environ.get("TASKS_PER_NODE", "1"))
        command.extend(
            [
                "srun",
                "-N",
                str(len(nodes)),
                "-n",
                str(len(nodes) * tasks_per_node),
                "--ntasks-per-node",
                str(tasks_per_node),
                "--gpus-per-task=1",
                "--gpu-bind=closest",
                f"--nodelist={','.join(nodes)}",
            ]
        )
    command.extend(
        [
            sys.executable,
            "-u",
            str(python_script),
            f"--inputfile={config_path}",
            f"--log={log_name}",
        ]
    )
    return command


def run(trial, dequed=None):
    """Execute one isolated trial and return DeepHyper's maximization objective."""
    log_dir = Path(os.environ.get("DEEPHYPER_LOG_DIR", "pubchem-hpo-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = json.load(stream)
    config = configure_trial(base_config, trial.parameters)

    config_path = (log_dir / f"trial-{trial.id}.json").resolve()
    output_path = (log_dir / f"trial-{trial.id}.log").resolve()
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    command = _trial_command(config_path, f"pubchem_hpo_{trial.id}", dequed)

    try:
        with output_path.open("w", encoding="utf-8") as output:
            subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        objective = validation_objective(output_path)
    except (OSError, subprocess.CalledProcessError):
        objective = -math.inf
    return {
        "objective": objective,
        "metadata": {"config": str(config_path), "log": str(output_path)},
    }


def build_problem(mpnn_types):
    """Build the nine-axis PubChem Gaussian search space."""
    from deephyper.hpo import HpProblem

    problem = HpProblem()
    problem.add_hyperparameter(mpnn_types, "mpnn_type")
    problem.add_hyperparameter(["off", "on"], "use_equivariant_graph_transformer")
    problem.add_hyperparameter([0.1, 1.0, 10.0], "energy_weight")
    problem.add_hyperparameter([0.1, 1.0, 10.0], "force_weight")
    problem.add_hyperparameter([0.1, 1.0, 10.0], "hessian_weight")
    problem.add_hyperparameter((2, 6), "num_conv_layers")
    problem.add_hyperparameter([64, 128, 256, 512], "hidden_dim")
    problem.add_hyperparameter([1, 2, 4, 8], "global_attn_heads")
    problem.add_hyperparameter((1, 4), "equivariant_attn_num_hidden_layers")
    problem.add_hyperparameter(
        [1, 2, 4], "equivariant_attn_feedforward_multiplier"
    )
    return problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument(
        "--mpnn-types",
        default="EGNN,SchNet,DimeNet,MACE,PAINN,PNAEq,AllScAIP,UMA",
        help="Comma-separated message-passing implementations",
    )
    args = parser.parse_args()

    try:
        from deephyper.evaluator import ProcessPoolEvaluator, queued
        from deephyper.hpo import CBO
        from hydragnn.utils.hpo.deephyper import read_node_list
    except ImportError as error:
        raise ImportError("Install deephyper to run PubChem Gaussian HPO") from error

    mpnn_types = [name.strip() for name in args.mpnn_types.split(",") if name.strip()]
    problem = build_problem(mpnn_types)
    queue, _ = read_node_list()
    nodes_per_trial = int(os.environ.get("NNODES_PER_TRIAL", "1"))
    num_workers = int(os.environ.get("NUM_CONCURRENT_TRIALS", "1"))
    evaluator = queued(ProcessPoolEvaluator)(
        run,
        num_workers=num_workers,
        queue=queue,
        queue_pop_per_task=nodes_per_trial,
    )
    search = CBO(
        problem,
        acq_func="UCB",
        multi_point_strategy="cl_min",
        random_state=42,
        log_dir=os.environ.get("DEEPHYPER_SEARCH_DIR", "pubchem-hpo"),
    )
    print(search.search(evaluator, max_evals=args.max_evals))


if __name__ == "__main__":
    main()
