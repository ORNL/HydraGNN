import os, sys

import math
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
import argparse
import time

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
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]

# HydraGNN-specific environment variables
MULTI_MODEL_LIST = os.environ["MULTI_MODEL_LIST"]
NUM_EPOCH = int(os.environ["NUM_EPOCH"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
HYDRAGNN_MAX_NUM_BATCH = int(os.environ["HYDRAGNN_MAX_NUM_BATCH"])


def to_float(x):
    x = x.lower()
    if x == "nan":
        return math.nan
    if x in ("inf", "+inf"):
        return math.inf
    if x == "-inf":
        return -math.inf
    return float(x)


def run(trial, dequed=None):
    f = open(f"output-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "gfm_mlip_all_mpnn.py")

    log_name = f"gfm_{SLURM_JOB_ID}_{trial.id}"
    master_addr = f"HYDRAGNN_MASTER_ADDR={dequed[0]}"
    nodelist = ",".join(dequed)

    prefix = " ".join(
        [
            f"srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            f"--ntasks-per-node=8 --gpus-per-node=8",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            f"--gpus-per-task=1 --gpu-bind=closest",
            f"--export=ALL,{master_addr}",
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
            f"--inputfile=gfm_mlip.json",
            f"--multi",
            f"--ddstore",
            f"--precision=fp64",
            f"--multi_model_list={MULTI_MODEL_LIST}",
            f"--num_epoch={NUM_EPOCH}",
            f"--batch_size={BATCH_SIZE}",
            f"--num_samples={BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH}",
            f"--oversampling_num_samples={BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH}",
            f"--log={log_name}",
            # f"--learning_rate={trial.parameters['learning_rate']}",
        ]
        + [
            f"--{param}={trial.parameters[param]}"
            for param in trial.parameters
            if param != "id"
        ]
    )

    print("Command = ", command, flush=True, file=f)

    objective = -math.inf
    num_pattern = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        fout = open(f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt", "r")
        while True:
            line = fout.readline()
            if "Tasks Val Loss" in line:
                nums = re.findall(num_pattern, line, flags=re.IGNORECASE)
                ## Task Val Loss: [a, b, c]. The output must be -(b+c)/2
                output = -0.5 * (to_float(nums[1]) + to_float(nums[2]))
                print(
                    f"Val losses: {-to_float(nums[1])}, {-to_float(nums[2])} Average: {output}",
                    flush=True,
                    file=f,
                )
            if not line:
                break
        fout.close()
    except Exception as excp:
        print(excp, flush=True, file=f)
        output = -math.inf

    print("Output:", output, flush=True, file=f)
    objective = output
    print(objective, flush=True, file=f)
    metadata = {"some_info": "some_value"}
    f.close()

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mpnn_type", type=str, default="EGNN,SchNet,DimeNet,MACE,PAINN,PNAEq"
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=200,
        help="Number of max evaluations for HPO search",
    )
    args = parser.parse_args()
    mpnn_type_list = args.mpnn_type.split(",")

    log_name = f"gfm-{SLURM_JOB_ID}"

    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import ProcessPoolEvaluator, queued
    from hydragnn.utils.hpo.deephyper import read_node_list

    ## We maintain dict of hyperparameters and then problem.add later
    hyperparameters = dict()
    hyperparameters["num_conv_layers"] = (2, 6)
    # keep <=6 conv layers to mitigate oversmoothing/oversquashing
    # ~5B params target (rough):  problem.add_hyperparameter((2, 6), "num_conv_layers")
    # ~10B params target (rough): problem.add_hyperparameter((2, 6), "num_conv_layers")
    # ~100B params target (rough): problem.add_hyperparameter((2, 6), "num_conv_layers")

    hyperparameters["hidden_dim"] = (100, 3000)
    # compensate with channel width (MPNN hidden_dim) instead of extra depth
    # ~5B params target (rough):  problem.add_hyperparameter((3000, 10000), "hidden_dim")
    # ~10B params target (rough): problem.add_hyperparameter((10000, 20000), "hidden_dim")
    # ~100B params target (rough): problem.add_hyperparameter((20000, 30000), "hidden_dim")

    hyperparameters["num_headlayers"] = (2, 4)
    # keep <=4 MLP head layers to mitigate oversmoothing/oversquashing
    # ~5B params target (rough):  problem.add_hyperparameter((2, 4), "num_headlayers")
    # ~10B params target (rough): problem.add_hyperparameter((2, 4), "num_headlayers")
    # ~100B params target (rough): problem.add_hyperparameter((2, 4), "num_headlayers")

    hyperparameters["dim_headlayers"] = (300, 2000)
    # ~5B params target (rough):  problem.add_hyperparameter((2000, 4000), "dim_headlayers")
    # ~10B params target (rough): problem.add_hyperparameter((4000, 600), "dim_headlayers")
    # ~100B params target (rough): problem.add_hyperparameter((6000, 9000), "dim_headlayers")

    hyperparameters["force_weight"] = [10.0, 50.0, 100.0]
    hyperparameters["learning_rate"] = (1e-5, 1e-3)
    hyperparameters["mpnn_type"] = mpnn_type_list

    ## Model specific hyperparameters
    if len(mpnn_type_list) == 1:
        if mpnn_type_list[0] == "EGNN":
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)
        elif mpnn_type_list[0] == "SchNet":
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["num_filters"] = (6, 2000)
            hyperparameters["num_gaussians"] = (3, 24)
        elif mpnn_type_list[0] == "DimeNet":
            hyperparameters["hidden_dim"] = (10, 100)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["basis_emb_size"] = (8, 200)
            hyperparameters["envelope_exponent"] = (4, 6)
            hyperparameters["int_emb_size"] = (16, 500)
            hyperparameters["out_emb_size"] = (24, 4000)
            hyperparameters["num_after_skip"] = (0, 2)
            hyperparameters["num_before_skip"] = (0, 2)
            hyperparameters["num_radial"] = (3, 12)
            hyperparameters["num_spherical"] = (2, 8)
        elif mpnn_type_list[0] == "MACE":
            hyperparameters["hidden_dim"] = (100, 1000)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["radial_type"] = ["bessel", "gaussian", "chebyshev"]
            hyperparameters["distance_transform"] = ["Agnesi", "Soft"]
            hyperparameters["num_radial"] = (3, 12)
            hyperparameters["max_ell"] = (2, 3)
            hyperparameters["node_max_ell"] = (1, 3)
            hyperparameters["correlation"] = (2, 3)
        elif mpnn_type_list[0] == "PAINN":
            hyperparameters["hidden_dim"] = (100, 1000)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["num_radial"] = (3, 12)
        elif mpnn_type_list[0] == "EGNN":
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)
        elif mpnn_type_list[0] == "PNAEq":
            hyperparameters["hidden_dim"] = (100, 1000)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["num_radial"] = (3, 12)
        else:
            raise ValueError(f"Unsupported MPNN type: {mpnn_type_list[0]}")

    ## Create HPO problem with the defined hyperparameters
    problem = HpProblem()
    for k, v in hyperparameters.items():
        problem.add_hyperparameter(v, k)

    # Create the node queue
    queue, _ = read_node_list()
    print("The queue:", queue, len(queue))
    print("NNODES_PER_TRIAL", NNODES_PER_TRIAL)
    print("NUM_CONCURRENT_TRIALS", NUM_CONCURRENT_TRIALS)
    print("NGPUS_PER_TRIAL", NGPUS_PER_TRIAL)
    print("NTOTGPUS", NTOTGPUS)
    print(NTOTGPUS, NGPUS_PER_TRIAL, NTOTGPUS // NGPUS_PER_TRIAL, len(queue))

    evaluator = queued(ProcessPoolEvaluator)(
        run,
        num_workers=NUM_CONCURRENT_TRIALS,
        queue=queue,
        queue_pop_per_task=NNODES_PER_TRIAL,
    )

    search = CBO(
        problem,
        acq_func="UCB",
        multi_point_strategy="cl_min",
        random_state=42,
        log_dir=log_name,
    )

    # Optionally preload results
    fname = os.path.join("gfm", "preloaded_results.csv")
    if os.path.exists(fname):
        t0 = time.time()
        print("Read existing results:", fname)
        preloaded_results = pd.read_csv(fname, header=0)
        search.fit_surrogate(preloaded_results)
        t1 = time.time()
        print("Fit done:", t1 - t0)

    timeout = None
    results = search.search(evaluator, max_evals=args.max_evals, timeout=timeout)
    print(results)

    sys.exit(0)
