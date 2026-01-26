import os, sys, time
import subprocess
import re
import torch
import pandas as pd

torch.backends.cudnn.enabled = False

random_state = 0
torch.manual_seed(random_state)

try:
    from torch_geometric.loader import DataLoader
except:  # pragma: no cover - compatibility shim
    from torch_geometric.data import DataLoader

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

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

MPNN_TYPE = "PNAEq"
WRAPPED_SCRIPT = os.path.join(os.path.dirname(__file__), "gfm_mlip_pnaeq.py")


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
    python_script = WRAPPED_SCRIPT

    log_name = f"gfm-{MPNN_TYPE.lower()}_{trial.id}"
    master_addr = f"HYDRAGNN_MASTER_ADDR={dequed[0]}"
    nodelist = ",".join(dequed)

    prefix = " ".join(
        [
            "srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            "--ntasks-per-node=8 --gpus-per-node=8",
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            "--gpus-per-task=1 --gpu-bind=closest",
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
            f"--mpnn_type={MPNN_TYPE}",
            f"--hidden_dim={trial.parameters['hidden_dim']}",
            f"--num_conv_layers={trial.parameters['num_conv_layers']}",
            f"--num_headlayers={trial.parameters['num_headlayers']}",
            f"--dim_headlayers={trial.parameters['dim_headlayers']}",
            f"--force_weight={trial.parameters['force_weight']}",
            f"--num_radial={trial.parameters['num_radial']}",
            "--inputfile=gfm_mlip.json",
            "--multi",
            "--ddstore",
            '--multi_model_list="ANI1x,MPTrj,OC2020,OC2022,ODAC23,OMat24,OMol25,OC2025,OPoly2026,nabla2dft,QCML,qm7x,transition1x"',
            "--num_epoch=10",
            f"--log={log_name}",
            f"--learning_rate={trial.parameters['learning_rate']}",
        ]
    )
    print("Command = ", command, flush=True, file=f)

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

    log_name = f"gfm-{MPNN_TYPE.lower()}"

    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from deephyper.hpo import HpProblem, CBO
    from hydragnn.utils.deephyper import read_node_list

    problem = HpProblem()

    problem.add_hyperparameter((2, 6), "num_conv_layers")
    problem.add_hyperparameter((100, 2000), "hidden_dim")
    problem.add_hyperparameter((2, 3), "num_headlayers")
    problem.add_hyperparameter((300, 1000), "dim_headlayers")
    problem.add_hyperparameter((10.0, 1000.0), "force_weight")
    problem.add_hyperparameter((1e-5, 3e-3), "learning_rate", log=True)
    problem.add_hyperparameter((3, 12), "num_radial")

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
        multi_point_strategy="cl_min",
        random_state=42,
        log_dir=log_name,
        n_jobs=OMP_NUM_THREADS,
    )

    fname = os.path.join("gfm", f"preloaded_results_{MPNN_TYPE.lower()}.csv")
    if os.path.exists(fname):
        t0 = time.time()
        print("Read existing results:", fname)
        preloaded_results = pd.read_csv(fname, header=0)
        search.fit_surrogate(preloaded_results)
        t1 = time.time()
        print("Fit done:", t1 - t0)

    timeout = None
    results = search.search(max_evals=200, timeout=timeout)
    print(results)

    sys.exit(0)
