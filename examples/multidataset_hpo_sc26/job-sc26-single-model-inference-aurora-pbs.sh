#!/bin/bash
#PBS -A CM2US
#PBS -N HydraGNN-SC26-Inf
#PBS -l select=16
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -l place=scatter
#PBS -q workq
#PBS -j oe

set -euo pipefail

function cmd() {
    echo "$@"
    time "$@"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
VENV_PATH=${VENV_PATH:-$HYDRAGNN_ROOT/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}
EXAMPLE_DIR=$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26

# Aurora proxy (needed on compute nodes for outbound HTTPS in some environments)
export HTTP_PROXY=${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTPS_PROXY=${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export ftp_proxy=$HTTP_PROXY
export no_proxy=${no_proxy:-admin,*.hostmgmt.cm.aurora.alcf.anl.gov,*.alcf.anl.gov,localhost}

module reset
module load frameworks

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: VENV_PATH not found: $VENV_PATH"
    echo "Using frameworks Python environment only."
fi

cd "$HYDRAGNN_ROOT"
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    NODE_COUNT=$(sort -u "${PBS_NODEFILE}" | wc -l)
    MASTER_HOST=$(head -n 1 "${PBS_NODEFILE}")
else
    NODE_COUNT=${NNODES:-16}
    MASTER_HOST=$(hostname)
fi

MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi

export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR=$MASTER_ADDR
export HYDRAGNN_MASTER_PORT=$MASTER_PORT

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

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

export HYDRAGNN_MAX_NUM_BATCH=1000
export NUM_EPOCH=50
export BATCH_SIZE=40
export NUM_SAMPLES=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH))
export INFER_PRECISION=fp64

# For inference, set CHECKPOINT_LOGDIR to the output log directory produced by training.
if [ -z "${CHECKPOINT_LOGDIR:-}" ]; then
    echo "ERROR: CHECKPOINT_LOGDIR is not set."
    echo "Example: export CHECKPOINT_LOGDIR=\"$HYDRAGNN_ROOT/logs/multidataset_hpo-<training-job-id>-NN16-AURORA-FSDP1-V2-TP0\""
    exit 1
fi

if [ ! -d "$CHECKPOINT_LOGDIR" ]; then
    echo "ERROR: CHECKPOINT_LOGDIR does not exist: $CHECKPOINT_LOGDIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_LOGDIR/config.json" ]; then
    echo "ERROR: config.json not found in $CHECKPOINT_LOGDIR"
    exit 1
fi

echo "Using checkpoint log dir: $CHECKPOINT_LOGDIR"
echo "Using datasets:           $MULTI_MODEL_LIST"
echo "Batch size / samples:     $BATCH_SIZE / $NUM_SAMPLES"
echo "Precision:                $INFER_PRECISION"

RANKS_PER_NODE=${RANKS_PER_NODE:-12}
NRANKS=$((NODE_COUNT * RANKS_PER_NODE))

cmd mpiexec -n "$NRANKS" -ppn "$RANKS_PER_NODE" \
    python -u "$EXAMPLE_DIR/inference.py" \
    --logdir "$CHECKPOINT_LOGDIR" \
    --multi_model_list "$MULTI_MODEL_LIST" \
    --dataset_dir "$EXAMPLE_DIR/dataset" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --precision "$INFER_PRECISION"
