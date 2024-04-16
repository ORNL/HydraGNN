import os, sys

import torch

torch.backends.cudnn.enabled = False

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import pandas as pd
import subprocess
import re

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

# Retrieve constants
NNODES = int(os.environ["NNODES"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NNODES_PER_TRIAL = int(os.environ["NNODES_PER_TRIAL"])
NGPUS_PER_TRIAL = int(os.environ["NGPUS_PER_TRIAL"])
NUM_CONCURRENT_TRIALS = int(os.environ["NUM_CONCURRENT_TRIALS"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
DEEPHYPER_DB_HOST = os.environ["DEEPHYPER_DB_HOST"]
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]


def _parse_results(stdout):
    pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    matches = re.findall(pattern, stdout.decode())
    if matches:
        return matches[-1][0]
    else:
        return "F"


def run(trial, dequed=None):
    f = open(f"output-{SLURM_JOB_ID}-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "gfm.py")

    # TODO: Launch a subprocess with `srun` to train neural networks
    params = trial.parameters
    log_name = "gfm" + "_" + str(trial.id)
    master_addr = f"HYDRAGNN_MASTER_ADDR={dequed[0]}"
    nodelist = ",".join(dequed)

    # time srun -u -n32 -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest
    prefix = " ".join(
        [
            f"srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            f"--ntasks-per-node=4 --gpus-per-node=4",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            f"--gpus-per-task=1",
            f"--export=ALL,{master_addr},HYDRAGNN_MAX_NUM_BATCH=100,HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1,HYDRAGNN_AGGR_BACKEND=mpi",
            f"--nodelist={nodelist}",
            f"--output {DEEPHYPER_LOG_DIR}/output_{SLURM_JOB_ID}_{trial.id}.txt",
            f"--error {DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt",
        ]
    )

    command = " ".join(
        [
            prefix,
            python_exe,
            "-u",
            python_script,
            f"--model_type={trial.parameters['model_type']}",
            f"--hidden_dim={trial.parameters['hidden_dim']}",
            f"--num_conv_layers={trial.parameters['num_conv_layers']}",
            f"--num_headlayers={trial.parameters['num_headlayers']}",
            f"--dim_headlayers={trial.parameters['dim_headlayers']}",
            f"--multi",
            f"--ddstore",
            # f'--multi_model_list="ANI1x,MPTrj,OC2020-2M,OC2022,qm7x"',
            ## debugging
            f'--multi_model_list="ANI1x,MPTrj,qm7x"',
            f"--num_samples=3200",
            f"--num_epoch=5",
            f"--log={log_name}",
        ]
    )
    print("Command = ", command, flush=True, file=f)

    result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    output = "F"
    try:
        pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        fout = open(f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt", "r")
        while True:
            line = fout.readline()
            matches = re.findall(pattern, line)
            if matches:
                output = -float(matches[-1][0])
            if not line:
                break
        fout.close()

    except Exception as excp:
        print(excp, flush=True, file=f)
        output = "F"

    print("Output:", output, flush=True, file=f)
    objective = output
    print(objective, flush=True, file=f)
    metadata = {"some_info": "some_value"}
    f.close()

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":

    log_name = f"gfm-{SLURM_JOB_ID}"

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from hydragnn.utils.deephyper import read_node_list

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((2, 6), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((100, 2000), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((100, 1000), "dim_headlayers")  # discrete parameter
    problem.add_hyperparameter(
        ["EGNN", "SchNet", "PNA"], "model_type"
    )  # categorical parameter

    # Create the node queue
    queue, _ = read_node_list()
    print("The queue:", queue, len(queue))
    print("NNODES_PER_TRIAL", NNODES_PER_TRIAL)
    print("NUM_CONCURRENT_TRIALS", NUM_CONCURRENT_TRIALS)
    print("NGPUS_PER_TRIAL", NGPUS_PER_TRIAL)
    print("NTOTGPUS", NTOTGPUS)
    print(NTOTGPUS, NGPUS_PER_TRIAL, NTOTGPUS // NGPUS_PER_TRIAL, len(queue))

    # Define the search space for hyperparameters
    # define the evaluator to distribute the computation
    evaluator = queued(ProcessPoolEvaluator)(
        run,
        num_workers=NUM_CONCURRENT_TRIALS,
        queue=queue,
        queue_pop_per_task=NNODES_PER_TRIAL,  # Remove the hard-coded value later
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
        n_jobs=OMP_NUM_THREADS,
    )

    timeout = None
    results = search.search(max_evals=100, timeout=timeout)
    print(results)

    sys.exit(0)
