"""
DeepHyper-based scalable hyperparameter optimization for the FNET temporal
anomaly-detection example.

Pattern follows ``examples/multidataset_hpo_sc26/gfm_deephyper_multi_all_mpnn.py``:
each HPO trial launches the existing single-trial training script
``fnet_temporal_anomaly_detection.py`` as a subprocess via ``srun`` and forwards
the sampled hyperparameters as ``--{name}={value}`` CLI flags. This keeps the
HPO orchestrator decoupled from the model code: anything the training script
already understands as a CLI argument can become a search dimension simply by
adding it to the ``HpProblem``.

Required environment variables (set by the accompanying SLURM job scripts):

  NNODES, NTOTGPUS, NNODES_PER_TRIAL, NGPUS_PER_TRIAL,
  NUM_CONCURRENT_TRIALS, NTOT_DEEPHYPER_RANKS, OMP_NUM_THREADS,
  DEEPHYPER_LOG_DIR, SLURM_JOB_ID

  FNET_CACHE_DIR     -- directory with pre-processed pickle/ADIOS splits
  FNET_DATE          -- date subfolder used during pre-processing (e.g. 2024-06-01)
  FNET_FORMAT        -- "pickle" or "adios" (must match the cache)
  NUM_EPOCH          -- training epochs per trial
  BATCH_SIZE         -- training batch size per trial

The objective maximised by DeepHyper is ``-val_loss`` (lower validation loss ->
higher objective).
"""

import argparse
import math
import os
import re
import subprocess
import sys

import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

# ---------------------------------------------------------------------------
# Cluster configuration (set by SLURM job script)
# ---------------------------------------------------------------------------
NNODES = int(os.environ["NNODES"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NNODES_PER_TRIAL = int(os.environ["NNODES_PER_TRIAL"])
NGPUS_PER_TRIAL = int(os.environ["NGPUS_PER_TRIAL"])
NUM_CONCURRENT_TRIALS = int(os.environ["NUM_CONCURRENT_TRIALS"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]

# ---------------------------------------------------------------------------
# FNET-specific fixed configuration (passed to every trial)
# ---------------------------------------------------------------------------
FNET_CACHE_DIR = os.environ["FNET_CACHE_DIR"]
FNET_DATE = os.environ.get("FNET_DATE", "2024-06-01")
FNET_FORMAT = os.environ.get("FNET_FORMAT", "pickle")
NUM_EPOCH = int(os.environ.get("NUM_EPOCH", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))


def _to_float(x: str) -> float:
    x = x.lower()
    if x == "nan":
        return math.nan
    if x in ("inf", "+inf"):
        return math.inf
    if x == "-inf":
        return -math.inf
    return float(x)


def _parse_val_loss_from_file(path: str) -> float:
    """Extract the last `Val Loss: <value>` written by HydraGNN's trainer."""
    pattern = r"Val Loss: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    output = -math.inf
    with open(path, "r", errors="replace") as fout:
        for line in fout:
            matches = re.findall(pattern, line)
            if matches:
                output = -_to_float(matches[-1])
    return output


def run(trial, dequed=None):
    os.makedirs(DEEPHYPER_LOG_DIR, exist_ok=True)
    f = open(f"{DEEPHYPER_LOG_DIR}/output-{trial.id}.txt", "w")

    python_exe = sys.executable
    python_script = os.path.join(
        os.path.dirname(__file__), "fnet_temporal_anomaly_detection.py"
    )
    log_name = f"fnet_{SLURM_JOB_ID}_{trial.id}"
    out_dir = f"outputs_fnet_hpo/{SLURM_JOB_ID}/{trial.id}"

    master_addr = f"HYDRAGNN_MASTER_ADDR={dequed[0]}"
    nodelist = ",".join(dequed)

    err_path = f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt"
    out_path = f"{DEEPHYPER_LOG_DIR}/output_{SLURM_JOB_ID}_{trial.id}.txt"

    prefix = " ".join(
        [
            "srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            "--ntasks-per-node=8 --gpus-per-node=8",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            "--gpus-per-task=1 --gpu-bind=closest",
            f"--export=ALL,{master_addr}",
            f"--nodelist={nodelist}",
            f"--output {out_path}",
            f"--error {err_path}",
        ]
    )

    # Fixed args (data + training-control) come from environment.
    fixed_args = [
        f"--cache_dir={FNET_CACHE_DIR}",
        f"--date={FNET_DATE}",
        f"--format={FNET_FORMAT}",
        f"--num_epoch={NUM_EPOCH}",
        f"--batch_size={BATCH_SIZE}",
        f"--log={log_name}",
        f"--out_dir={out_dir}",
    ]

    # Hyperparameter args from DeepHyper (forwarded verbatim as --{key}={val}).
    param_args = [f"--{k}={v}" for k, v in trial.parameters.items() if k != "id"]

    command = " ".join(
        [prefix, python_exe, "-u", python_script] + fixed_args + param_args
    )

    print(f"Command = {command}", flush=True, file=f)

    output = -math.inf
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        # HydraGNN writes "Val Loss: ..." to stderr (via print_utils); the SLURM
        # `--error` file captures it. Fall back to stdout if needed.
        for path in (err_path, out_path):
            if os.path.exists(path):
                val = _parse_val_loss_from_file(path)
                if val != -math.inf:
                    output = val
                    break
    except subprocess.CalledProcessError as exc:
        tail = exc.output.decode(errors="replace")[-2000:] if exc.output else ""
        print(
            f"Subprocess failed (rc={exc.returncode}); tail of output:\n{tail}",
            flush=True,
            file=f,
        )
        output = -math.inf
    except Exception as exc:
        print(f"Unexpected error: {exc}", flush=True, file=f)
        output = -math.inf

    print(f"Objective: {output}", flush=True, file=f)
    f.close()

    return {"objective": output, "metadata": {"trial_id": str(trial.id)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mpnn_type",
        type=str,
        default="GCN,GIN,GAT,SAGE,PNA",
        help="Comma-separated MPNN backbones to include in the search space.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=200,
        help="Maximum number of HPO evaluations.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-clock search timeout in seconds (None = unlimited).",
    )
    args = parser.parse_args()
    mpnn_type_list = args.mpnn_type.split(",")

    log_name = f"fnet_hpo-{SLURM_JOB_ID}"
    if len(mpnn_type_list) == 1:
        log_name = f"fnet_hpo_{mpnn_type_list[0]}-{SLURM_JOB_ID}"

    # Newer DeepHyper exposes HpProblem/CBO under `deephyper.hpo`; older
    # releases keep them under `deephyper.problem` / `deephyper.search.hps`.
    try:
        from deephyper.hpo import CBO, HpProblem
    except ImportError:
        from deephyper.problem import HpProblem
        from deephyper.search.hps import CBO
    from deephyper.evaluator import ProcessPoolEvaluator, queued

    from hydragnn.utils.hpo.deephyper import read_node_list

    # ---------------------------------------------------------------
    # Search space (hyperparameter dict -> HpProblem)
    # ---------------------------------------------------------------
    hyperparameters = dict()
    hyperparameters["mpnn_type"] = mpnn_type_list
    hyperparameters["backbone"] = ["gru", "lstm"]
    hyperparameters["mode"] = ["post_gcn", "pre_gcn", "interleaved"]
    hyperparameters["hidden_dim"] = (16, 128)
    hyperparameters["num_conv_layers"] = (1, 4)
    hyperparameters["lookback"] = (8, 64)
    hyperparameters["k"] = (2, 10)
    hyperparameters["lag_lambda"] = (0.5, 10.0)
    hyperparameters["learning_rate"] = (1e-5, 1e-2)

    problem = HpProblem()
    for k, v in hyperparameters.items():
        problem.add_hyperparameter(v, k)

    # ---------------------------------------------------------------
    # Node queue (multi-node) -> queued ProcessPoolEvaluator
    # ---------------------------------------------------------------
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
        evaluator,
        acq_func="UCB",
        multi_point_strategy="cl_min",  # Constant liar (async parallelism)
        random_state=42,
        log_dir=log_name,
        n_jobs=OMP_NUM_THREADS,
    )

    results = search.search(max_evals=args.max_evals, timeout=args.timeout)
    print(results)

    sys.exit(0)
