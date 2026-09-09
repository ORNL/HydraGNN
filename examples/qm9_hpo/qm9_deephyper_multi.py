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
import os, sys

import pandas as pd
import subprocess
import re

try:
    from .workflow import load_base_config, prepare_splits
except ImportError:
    from workflow import load_base_config, prepare_splits

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def _settings():
    names = (
        "NNODES",
        "NTOTGPUS",
        "NNODES_PER_TRIAL",
        "NGPUS_PER_TRIAL",
        "NUM_CONCURRENT_TRIALS",
        "NTOT_DEEPHYPER_RANKS",
        "OMP_NUM_THREADS",
    )
    return {name: int(os.environ[name]) for name in names}


def _parse_results(stdout):
    pattern = r"Validation Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    matches = re.findall(pattern, stdout.decode())
    # By default, DeepHyper maximized the objective function, so we need to flip the sign of the validation loss function
    if matches:
        return -float(matches[-1][0])
    else:
        return "F"


def run(trial, dequed=None):
    settings = _settings()
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "qm9.py")

    # TODO: Launch a subprocess with `srun` to train neural networks
    params = trial.parameters
    log_name = "qm9" + "_" + str(trial.id)
    command = [
        "srun",
        "-N",
        str(settings["NNODES_PER_TRIAL"]),
        "-n",
        str(settings["NGPUS_PER_TRIAL"]),
        "--ntasks-per-node=8",
        "--gpus-per-node=8",
        "--cpus-per-task",
        str(settings["OMP_NUM_THREADS"]),
        "--threads-per-core=1",
        "--cpu-bind=threads",
        "--gpus-per-task=1",
        "--gpu-bind=closest",
        f"--export=ALL,HYDRAGNN_MASTER_ADDR={dequed[0]}",
        python_exe,
        "-u",
        python_script,
        f"--mpnn_type={trial.parameters['mpnn_type']}",
        f"--hidden_dim={trial.parameters['hidden_dim']}",
        f"--num_conv_layers={trial.parameters['num_conv_layers']}",
        f"--num_headlayers={trial.parameters['num_headlayers']}",
        f"--dim_headlayers={trial.parameters['dim_headlayers']}",
        f"--log={log_name}",
    ]
    with open(f"output-{trial.id}.txt", "w", encoding="utf-8") as stream:
        print("Command =", command, file=stream)
        try:
            result = subprocess.check_output(command, stderr=subprocess.STDOUT)
            objective = _parse_results(result)
        except Exception as error:
            print(error, file=stream)
            objective = "F"
        print("Objective =", objective, file=stream)
    metadata = {"some_info": "some_value"}

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":

    log_name = "qm9"
    settings = _settings()

    # Finish the shared raw download and processed-cache build before launching
    # concurrent srun trials. Trial-local rank-zero barriers alone cannot
    # coordinate independent MPI jobs that share the same filesystem.
    prepare_splits(load_base_config())

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from hydragnn.utils.deephyper import read_node_list

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((1, 2), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((50, 52), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((1, 3), "dim_headlayers")  # discrete parameter
    problem.add_hyperparameter(
        ["EGNN", "PNA", "SchNet", "DimeNet"], "mpnn_type"
    )  # categorical parameter

    # Create the node queue
    queue, _ = read_node_list()
    print("The queue:", queue, len(queue))
    print("NNODES_PER_TRIAL", settings["NNODES_PER_TRIAL"])
    print("NUM_CONCURRENT_TRIALS", settings["NUM_CONCURRENT_TRIALS"])
    print("NGPUS_PER_TRIAL", settings["NGPUS_PER_TRIAL"])
    print("NTOTGPUS", settings["NTOTGPUS"])

    # Define the search space for hyperparameters
    # define the evaluator to distribute the computation
    evaluator = queued(ProcessPoolEvaluator)(
        run,
        num_workers=settings["NUM_CONCURRENT_TRIALS"],
        queue=queue,
        queue_pop_per_task=settings["NNODES_PER_TRIAL"],
    )

    # Define the search method and scalarization
    # search = CBO(problem, parallel_evaluator, random_state=42, log_dir=log_name)
    search = CBO(
        problem,
        evaluator,
        acq_func="UCB",
        multi_point_strategy="cl_min",  # Constant liar strategy
        random_state=42,
        # Location where to store the results
        log_dir=log_name,
        # Number of threads used to update surrogate model of BO
        n_jobs=settings["OMP_NUM_THREADS"],
    )

    timeout = 1200
    results = search.search(max_evals=10, timeout=timeout)
    print(results)

    sys.exit(0)
