import os, sys

import torch

torch.backends.cudnn.enabled = False

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import pandas as pd
import subprocess
import re
import time
import random

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
#OMNISTAT_WRAPPER = os.environ["OMNISTAT_WRAPPER"]

def _parse_results(stdout):
    pattern = r"Val Loss: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    matches = re.findall(pattern, stdout.decode())
    if matches:
        return matches[-1][0]
    else:
        return "F"


def run(trial, dequed=None):
    f = open(f"output-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "gfm.py")

    # TODO: Launch a subprocess with `srun` to train neural networks
    params = trial.parameters
    log_name = "gfm" + "_" + str(trial.id)

    # time srun -u -n32 -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest
    prefix = " ".join(
        [
            f"srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL} -u",
            f"--ntasks-per-node=8 --gpus-per-node=8",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            f"--gpus-per-task=1 --gpu-bind=closest",
            f"--export=ALL,HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1,HYDRAGNN_AGGR_BACKEND=mpi",
            # f"--nodelist={nodelist}",
            f"--output {DEEPHYPER_LOG_DIR}/output_{SLURM_JOB_ID}_{trial.id}.txt",
            f"--error {DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt",
        ]
    )

    command = " ".join(
        [
            prefix,
            f"bash -c \"",
            f"touch {DEEPHYPER_LOG_DIR}/trial_map_{trial.id}_\\$SLURM_STEP_ID;",
            #f"{OMNISTAT_WRAPPER} rms;",
            python_exe,
            "-u",
            python_script,
            f"--model_type={trial.parameters['model_type']}",
            f"--hidden_dim={trial.parameters['hidden_dim']}",
            f"--num_conv_layers={trial.parameters['num_conv_layers']}",
            f"--num_headlayers={trial.parameters['num_headlayers']}",
            f"--dim_headlayers={trial.parameters['dim_headlayers']}",
            f"--batch_size={trial.parameters['batch_size']}",
            f"--multi",
            f"--ddstore",
            #f"--multi_model_list=ANI1x-v3,MPTrj-v3,OC2020-20M-v3,OC2022-v3,qm7x-v3",
            f"--multi_model_list=ANI1x-v3,MPTrj-v3",
            ## debugging
            #f"--multi_model_list=ANI1x-v3",
            # f"--num_samples=1000",
            f"--num_epoch=10",
            f"--log={log_name};",
            #f"{OMNISTAT_WRAPPER} rms --nostep;",
            f"\"",
        ]
    )
    print("Command = ", command, flush=True, file=f)
    ## try to avoid burst execution
    random_int = random.randrange(NUM_CONCURRENT_TRIALS*10)
    print(f"Random wait {random_int} sec", flush=True, file=f)
    time.sleep(random_int)

    output = "F"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
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

    log_name = "gfm"

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.hpo import HpProblem, CBO
    from hydragnn.utils.deephyper import read_node_list
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from hydragnn.utils.deephyper import read_node_list

    problem = HpProblem()

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((2, 6), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((100, 2000), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((2, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((300, 1000), "dim_headlayers")  # discrete parameter
    problem.add_hyperparameter((16, 128), "batch_size")  # batch_size
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
    # Note: In newer DeepHyper API, evaluator is passed to search.search(), not CBO constructor
    search = CBO(
        problem,
        acq_func="UCB",
        multi_point_strategy="cl_min",  # Constant liar strategy
        # random_state=42,
        # Location where to store the results
        log_dir=log_name,
    )

    fname = os.path.join("gfm", "preloaded_results.csv")
    if os.path.exists(fname):
        t0 = time.time()
        print("Read existing results:", fname)
        preloaded_results = pd.read_csv(fname, header=0)
        search.fit_surrogate(preloaded_results)
        t1 = time.time()
        print("Fit done:", t1-t0)

    print("Search starts")
    timeout = None
    results = search.search(max_evals=200, timeout=timeout, evaluator=evaluator)
    print(results)

    sys.exit(0)
