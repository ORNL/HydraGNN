#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN-Inf
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:30:00
#SBATCH -N 16
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -C nvme

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

function cmd() {
    echo "$@"
    time "$@"
}

HYDRAGNN_ROOT=/lustre/orion/mat746/world-shared/mlupopa/Supercomputing2026/HydraGNN
EXAMPLE_DIR=$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26

# Load conda environment (same pattern as job-sc26-oom.sh)
source /lustre/orion/mat746/world-shared/mlupopa/module-to-load-frontier-rocm640.sh
source activate /lustre/orion/mat746/world-shared/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm6.4/hydragnn_venv

cd "$HYDRAGNN_ROOT"
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo "===== LD_LIBRARY_PATH ====="
echo $LD_LIBRARY_PATH | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p "$MIOPEN_USER_DB_PATH"

export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

# Multi-node torch/c10d networking
MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR="$MASTER_ADDR"
export HYDRAGNN_MASTER_PORT="$MASTER_PORT"
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0

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

# Keep key size/precision knobs aligned with job-sc26-oom.sh style
export HYDRAGNN_MAX_NUM_BATCH=1000
export NUM_EPOCH=50
export BATCH_SIZE=40
export NUM_SAMPLES=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH))
export INFER_PRECISION=fp64

# Hard-coded training log directory to load for inference.
CHECKPOINT_LOGDIR="$HYDRAGNN_ROOT/logs/multidataset_hpo-4150722-NN16-FSDP1-V2-TP0"

if [ -z "$CHECKPOINT_LOGDIR" ] || [ ! -d "$CHECKPOINT_LOGDIR" ]; then
    echo "ERROR: Could not resolve CHECKPOINT_LOGDIR."
    echo "Expected hard-coded path: $CHECKPOINT_LOGDIR"
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

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest -l --kill-on-bad-exit=1 \
    --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,HYDRAGNN_MASTER_ADDR=$HYDRAGNN_MASTER_ADDR,HYDRAGNN_MASTER_PORT=$HYDRAGNN_MASTER_PORT,GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME,NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    python -u "$EXAMPLE_DIR/inference.py" \
    --logdir "$CHECKPOINT_LOGDIR" \
    --multi_model_list "$MULTI_MODEL_LIST" \
    --dataset_dir "$EXAMPLE_DIR/dataset" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --precision "$INFER_PRECISION"
