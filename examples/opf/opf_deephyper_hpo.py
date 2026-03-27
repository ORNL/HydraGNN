import os, sys

import math
import torch

torch.backends.cudnn.enabled = False

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import pandas as pd
import subprocess
import re
import argparse
import glob

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

# OPF-specific environment variables
NUM_EPOCH = int(os.environ["NUM_EPOCH"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])


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
    os.makedirs(f"{DEEPHYPER_LOG_DIR}", exist_ok=True)
    f = open(f"{DEEPHYPER_LOG_DIR}/output-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(
        os.path.dirname(__file__), "train_opf_solution_heterogeneous.py"
    )

    log_name = f"opf_hpo_{SLURM_JOB_ID}_{trial.id}"
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
            f"--inputfile=opf_solution_heterogeneous.json",
            f"--hdf5",
            f"--num_epoch={NUM_EPOCH}",
            f"--batch_size={BATCH_SIZE}",
            f"--log={log_name}",
            f"--mpnn_type={trial.parameters['mpnn_type']}",
            f"--hidden_dim={trial.parameters['hidden_dim']}",
            f"--num_conv_layers={trial.parameters['num_conv_layers']}",
            f"--learning_rate={trial.parameters['learning_rate']}",
        ]
    )

    print("Command = ", command, flush=True, file=f)

    output = -math.inf
    num_pattern = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        fout = open(f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt", "r")
        while True:
            line = fout.readline()
            if "Tasks Val Loss:" in line:
                nums = re.findall(num_pattern, line, flags=re.IGNORECASE)
                # nums[0] is the rank prefix (e.g. "0:"), nums[1] is the actual loss
                if len(nums) >= 2:
                    val = -to_float(nums[1])
                    print(
                        f"Val loss: {-val}",
                        flush=True,
                        file=f,
                    )
                    # Keep the best (minimum) val loss across epochs
                    if val > output:
                        output = val
            if not line:
                break
        fout.close()
    except subprocess.CalledProcessError as cpe:
        # If the trial was killed by walltime, epochs may have completed.
        # Try to extract the best val loss from whatever was logged.
        print(f"Trial failed with exit code {cpe.returncode}", flush=True, file=f)
        error_file = f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt"
        if os.path.exists(error_file):
            fout = open(error_file, "r")
            for line in fout:
                if "Tasks Val Loss:" in line:
                    nums = re.findall(num_pattern, line, flags=re.IGNORECASE)
                    if len(nums) >= 2:
                        val = -to_float(nums[1])
                        print(
                            f"Val loss (from partial run): {-val}",
                            flush=True,
                            file=f,
                        )
                        if val > output:
                            output = val
            fout.close()
    except Exception as excp:
        print(excp, flush=True, file=f)
        output = -math.inf

    print(f"Best val loss (min across epochs): {-output}", flush=True, file=f)
    print("Output:", output, flush=True, file=f)
    objective = output
    print(objective, flush=True, file=f)
    metadata = {"mpnn_type": trial.parameters["mpnn_type"]}
    f.close()

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mpnn_type",
        type=str,
        default="HeteroPNA,HeteroSAGE,HeteroGAT,HeteroRGAT,HeteroHGT,HeteroHEAT",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=100,
        help="Number of max evaluations for HPO search",
    )
    parser.add_argument(
        "--hidden_dim_range",
        type=str,
        default="32,256",
        help="min,max for hidden_dim (e.g. '32,64')",
    )
    parser.add_argument(
        "--num_conv_layers_range",
        type=str,
        default="2,6",
        help="min,max for num_conv_layers (e.g. '2,4')",
    )
    parser.add_argument(
        "--learning_rate_range",
        type=str,
        default="1e-5,1e-2",
        help="min,max for learning_rate (e.g. '1e-4,1e-2')",
    )
    args = parser.parse_args()
    mpnn_type_list = args.mpnn_type.split(",")

    log_name = f"opf_hpo-{SLURM_JOB_ID}"
    if len(mpnn_type_list) == 1:
        log_name = f"opf_hpo_{mpnn_type_list[0]}-{SLURM_JOB_ID}"

    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import ProcessPoolEvaluator, queued
    from hydragnn.utils.hpo.deephyper import read_node_list

    hd_min, hd_max = (int(x) for x in args.hidden_dim_range.split(","))
    cl_min, cl_max = (int(x) for x in args.num_conv_layers_range.split(","))
    lr_min, lr_max = (float(x) for x in args.learning_rate_range.split(","))

    hyperparameters = dict()
    hyperparameters["mpnn_type"] = mpnn_type_list
    hyperparameters["learning_rate"] = (lr_min, lr_max)
    hyperparameters["hidden_dim"] = (hd_min, hd_max)
    hyperparameters["num_conv_layers"] = (cl_min, cl_max)

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

    ## Preload results from previous runs
    opf_dir = "opf_hpo"
    if len(mpnn_type_list) == 1:
        opf_dir = f"opf_hpo_{mpnn_type_list[0]}"

    df_list = list()
    files = glob.glob(os.path.join(opf_dir, "*.csv"))
    for fname in files:
        try:
            df = pd.read_csv(fname, header=0)
            total_rows = len(df)
            df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
            df = df.dropna(subset=["objective"])
            valid_rows = list()
            for i in range(len(df)):
                try:
                    search.fit_surrogate(df.iloc[i : i + 1])
                    valid_rows.append(i)
                except:
                    continue
            print(f"Checking {fname}: total {total_rows}, valid {len(valid_rows)}")
            df = df.iloc[valid_rows]
            df_list.append(df)
        except Exception as excp:
            print(f"Error loading {fname}:", excp)

    ## Create a clean search object again
    search = CBO(
        problem,
        acq_func="UCB",
        multi_point_strategy="cl_min",
        random_state=42,
        log_dir=log_name,
    )

    if len(df_list) > 0:
        try:
            preloaded_results = pd.concat(df_list, ignore_index=True)
            print(
                f"Loaded {len(preloaded_results)} preloaded results from {len(df_list)} files."
            )
            search.fit_surrogate(preloaded_results)
        except Exception as excp:
            print("Error in loading preloaded results:", excp)

    timeout = None
    results = search.search(evaluator, max_evals=args.max_evals, timeout=timeout)
    print(results)

    sys.exit(0)
