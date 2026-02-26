#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J HydraGNN-SC26-Inf
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/global/cfs/projectdirs/amsc001/cm2us/mlupopa/HydraGNN}
VENV_PATH=${VENV_PATH:-$HYDRAGNN_ROOT/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv}
EXAMPLE_DIR=$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26

# --- Perlmutter module + conda setup ---
module reset
ml nersc-default/1.0 || true
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found."
    exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: VENV_PATH does not exist: $VENV_PATH"
    echo "Set VENV_PATH to your Perlmutter HydraGNN conda env path."
    exit 1
fi

conda activate "$VENV_PATH"

cd "$HYDRAGNN_ROOT" || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

# --- MPI/runtime envs ---
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=8
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

# Dataset ordering matches gfm_deephyper_multi_all_mpnn.py multi_model_list
export datadir0=Alexandria
export datadir1=ANI1x
export datadir2=MPTrj
export datadir3=OC2020
export datadir4=OC2022
export datadir5=OC25
export datadir6=ODAC23
export datadir7=OMat24
export datadir8=OMol25
export datadir9=OMol25-neutral
export datadir10=OMol25-non-neutral
export datadir11=OPoly2026
export datadir12=Nabla2DFT
export datadir13=QCML
export datadir14=QM7X
export datadir15=transition1x

MULTI_MODEL_LIST=$datadir0
# MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15

# Keep key size/precision knobs aligned with training script
export HYDRAGNN_MAX_NUM_BATCH=1000
export NUM_EPOCH=50
export BATCH_SIZE=40
export NUM_SAMPLES=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH))
export INFER_PRECISION=fp64

# Hard-coded training log directory to load for inference.
CHECKPOINT_LOGDIR="$HYDRAGNN_ROOT/logs/multidataset_hpo-4150722-NN16-PM-FSDP1-V2-TP0"

if [ -z "$CHECKPOINT_LOGDIR" ] || [ ! -d "$CHECKPOINT_LOGDIR" ]; then
    echo "ERROR: Could not resolve CHECKPOINT_LOGDIR."
    echo "Expected hard-coded path: $CHECKPOINT_LOGDIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_LOGDIR/config.json" ]; then
    echo "ERROR: config.json not found in $CHECKPOINT_LOGDIR"
    exit 1
fi

# Distributed rendezvous for torch/c10d
MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR=$MASTER_ADDR
export HYDRAGNN_MASTER_PORT=$MASTER_PORT

echo "Using checkpoint log dir: $CHECKPOINT_LOGDIR"
echo "Using datasets:           $MULTI_MODEL_LIST"
echo "Batch size / samples:     $BATCH_SIZE / $NUM_SAMPLES"
echo "Precision:                $INFER_PRECISION"

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -l --kill-on-bad-exit=1 \
    --export=ALL \
    python -u "$EXAMPLE_DIR/inference.py" \
    --logdir "$CHECKPOINT_LOGDIR" \
    --multi_model_list "$MULTI_MODEL_LIST" \
    --dataset_dir "$EXAMPLE_DIR/dataset" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --precision "$INFER_PRECISION"
