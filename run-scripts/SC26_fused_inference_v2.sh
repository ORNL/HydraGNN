#!/bin/bash
# ============================================================================
# HydraGNN Fused Inference — Frontier Golden Script (MI250X)
# ============================================================================
#
# Single self-contained job script for running fused energy-gradient inference
# with encoder reuse on OLCF Frontier (AMD MI250X GPUs).
#
# USAGE:
#   1. Edit the two lines marked *** below (project ID and node count)
#   2. sbatch scripts/frontier_golden_fused_inference.sh
#
# QUICK REFERENCE — Frontier batch partition scheduling bins:
#   Nodes        Max Walltime   Aging Boost
#   1–91         2 h            —
#   92–183       6 h            —
#   184–1,881    12 h           —
#   1,882–5,644  12 h           4 days
#   5,645–9,472  12 h           8 days
#
# QUICK REFERENCE — Batch sizes (v6-tested, MI250X fused+encoder-reuse, 64 GiB HBM):
#   fp64:  50   (default — primary datatype)
#   fp32: 150   (secondary — future runs)
#   bf16: 250   (no planned runs, for reference only)
#
# NVMe BURST BUFFER:
#   Each Frontier node has 2x 1.92 TB NVMe SSDs (5.5 GB/s read, 2 GB/s write).
#   Accessed at /mnt/bb/$USER after requesting -C nvme.
#   This script writes per-GPU JSON to NVMe, stages out to Lustre, then cleans up.
#
# STDOUT-ONLY MODE:
#   To skip JSON output entirely, change INFER_SCRIPT below to inference_fused.py.
#   NVMe allocation (-C nvme) can be removed in that case.
#
# CONDA AT EXTREME SCALE (>1,000 nodes):
#   Consider using sbcast to broadcast the packed conda env to each node's
#   NVMe or /tmp to avoid Lustre contention during Python imports.
#   See: https://docs.olcf.ornl.gov/software/python/sbcast_conda.html
#
# OMNISTAT GPU MONITORING:
#   Uses Frontier's official omnistat-wrapper module.
#   See: https://rocm.github.io/omnistat/site.frontier.html
#   FOM (figure of merit) integration sends per-batch throughput to Omnistat.
# ============================================================================

# ***  EDIT THIS: your OLCF project allocation  ***
#SBATCH -A <YOUR_PROJECT>

# ***  EDIT THIS: number of compute nodes (up to 9,408; see walltime table above)  ***
#SBATCH -N 10

#SBATCH -J hydragnn-fused
#SBATCH -o hydragnn-fused-%j.out
#SBATCH -e hydragnn-fused-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

set -euo pipefail

# ============================================================================
# Configuration — safe defaults; override via environment if needed
# ============================================================================

PRECISION=${PRECISION:-fp64}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-50}
INFER_NUM_STRUCTURES=${INFER_NUM_STRUCTURES:-15000}
INFER_SCRIPT=${INFER_SCRIPT:-inference_fused_write_adios.py}

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
EXAMPLE_DIR=${EXAMPLE_DIR:-$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26}

CHECKPOINT_LOGDIR=${CHECKPOINT_LOGDIR:-$HYDRAGNN_ROOT/logs/multidataset_hpo-BEST6-fp64}
MLP_CHECKPOINT=${MLP_CHECKPOINT:-"${CHECKPOINT_LOGDIR}/mlp_branch_weights.pt"}

# CONDA_ENV=${CONDA_ENV:-/lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN-infer/HydraGNN-Installation-Frontier/hydragnn_venv}
RESULTS_DIR=${RESULTS_DIR:-"${HYDRAGNN_ROOT}/fused_${SLURM_JOB_ID}"}

# ============================================================================
# Module environment
# ============================================================================

#module reset
#module load rocm/7.2.0
#module load craype-accel-amd-gfx90a
#module unload darshan-runtime

# Load conda environment
module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/7.1.1
ml amd-mixed/7.1.1
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
module unload darshan-runtime

module use /sw/frontier/amdsw/modulefiles
module load omnistat-wrapper


# Unset proxies — Omnistat's internal server communication breaks behind proxies
# (see Frontier User Guide / Omnistat section)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY proxy no_proxy all_proxy ftp_proxy

# ============================================================================
# Conda environment
# ============================================================================

# shellcheck source=/dev/null
source activate ${HYDRAGNN_ROOT}/HydraGNN-Installation-Frontier/hydragnn_venv

# export LD_LIBRARY_PATH="${CONDA_ENV}/lib64:${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${HYDRAGNN_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"

PYTHON_BIN=$(which python)

echo ""
echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

# ============================================================================
# Runtime environment
# ============================================================================

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0

unset PYTORCH_ALLOC_CONF

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT=1

# ============================================================================
# Distributed setup (MASTER_ADDR from first node in allocation)
# ============================================================================

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR="$MASTER_ADDR"
export HYDRAGNN_MASTER_PORT="$MASTER_PORT"

# ============================================================================
# Validate paths
# ============================================================================

if [ ! -d "$CHECKPOINT_LOGDIR" ]; then
    echo "ERROR: CHECKPOINT_LOGDIR does not exist: $CHECKPOINT_LOGDIR" >&2
    exit 1
fi
if [ ! -f "$MLP_CHECKPOINT" ]; then
    echo "ERROR: MLP checkpoint not found: $MLP_CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT_LOGDIR/config.json" ]; then
    echo "ERROR: config.json not found in $CHECKPOINT_LOGDIR" >&2
    exit 1
fi
if ! ls "$CHECKPOINT_LOGDIR"/*.pk &>/dev/null; then
    echo "ERROR: No .pk checkpoint found in $CHECKPOINT_LOGDIR" >&2
    exit 1
fi

# ============================================================================
# Results directory (Lustre)
# ============================================================================

mkdir -p "$RESULTS_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"

# ============================================================================
# Omnistat GPU monitoring — start
# ============================================================================

_omnistat_started=0
_omnistat_config="${HYDRAGNN_ROOT}/omnistat.hydragnn-external.config"

if [ -n "${OMNISTAT_WRAPPER:-}" ]; then
    if [ -f "$_omnistat_config" ]; then
        export OMNISTAT_CONFIG="$_omnistat_config"
        ${OMNISTAT_WRAPPER} usermode --start --interval 15.0 --pushinterval 3
        _omnistat_started=1
    else
        echo "WARNING: Omnistat configuration file not found: ${_omnistat_config}"
    fi
else
    echo "WARNING: OMNISTAT_WRAPPER not set — omnistat-wrapper module not loaded?"
fi

# ============================================================================
# Omnistat FOM (figure of merit)
# ============================================================================

_fom_args=""
_fom_port=8001

if [ "$_omnistat_started" = "1" ]; then
    _fom_args="--omnistat_fom --omnistat_fom_port ${_fom_port}"
    echo "OMNISTAT Figure of Merit (FOM): port=${_fom_port}"
fi

# ============================================================================
# NVMe burst buffer setup
# ============================================================================

NNODES=${SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=8
NTASKS=$((NNODES * GPUS_PER_NODE))

_nvme_args=""
NVME_DIR=""
if [[ "$INFER_SCRIPT" == *write_adios* ]]; then
    NVME_DIR="/tmp/${USER}/hydragnn_${SLURM_JOB_ID}"
    mkdir -p $NVME_DIR
    _nvme_args="--nvme_dir ${NVME_DIR}"
    echo "NVME_DIR=$NVME_DIR (node-local, staged out to Lustre after inference)"
fi

# ============================================================================
# Print job summary
# ============================================================================

echo "============================================================"
echo "HydraGNN Fused Inference — Job Summary"
echo "============================================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
echo "NNODES=$NNODES  GPUS_PER_NODE=$GPUS_PER_NODE  NTASKS=$NTASKS"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "HYDRAGNN_ROOT=$HYDRAGNN_ROOT"
echo "CHECKPOINT_LOGDIR=$CHECKPOINT_LOGDIR"
echo "MLP_CHECKPOINT=$MLP_CHECKPOINT"
echo "PRECISION=$PRECISION"
echo "INFER_BATCH_SIZE=$INFER_BATCH_SIZE"
echo "INFER_NUM_STRUCTURES=$INFER_NUM_STRUCTURES"
echo "INFER_SCRIPT=$INFER_SCRIPT"
echo "RESULTS_DIR=$RESULTS_DIR"
# echo "CONDA_ENV=$CONDA_ENV"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "============================================================"
env | grep '^\SLURM_' | sort || true
echo "============================================================"

# ============================================================================
# Launch fused inference (single srun — recommended for Frontier at scale)
# ============================================================================

_struct_args="--min_atoms 2 --max_atoms 500 --box_size 10.0"
_fused_args="--fused_energy_grad --encoder_reuse --num_streams 1 --profile_stages"

echo "Launching inference: srun -N${NNODES} -n${NTASKS} ..."
set +e
time srun -N"$NNODES" -n"$NTASKS" --ntasks-per-node="$GPUS_PER_NODE" \
    -c7 --gpu-bind=closest -l --kill-on-bad-exit=1 \
    --export=ALL,MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},HYDRAGNN_MASTER_ADDR=${HYDRAGNN_MASTER_ADDR},HYDRAGNN_MASTER_PORT=${HYDRAGNN_MASTER_PORT} \
    bash -c "
        export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
        $PYTHON_BIN -u ${EXAMPLE_DIR}/${INFER_SCRIPT} \
            --logdir ${CHECKPOINT_LOGDIR} \
            --mlp_checkpoint ${MLP_CHECKPOINT} \
            --num_structures ${INFER_NUM_STRUCTURES} \
            --batch_size ${INFER_BATCH_SIZE} \
            --precision ${PRECISION} \
            ${_struct_args} \
            ${_nvme_args} \
            ${_fom_args} \
            ${_fused_args}
    "
_srun_exit=$?
set -e
echo "Inference srun exit code: $_srun_exit"

# ============================================================================
# Stage-out: copy ADIOS results from each node's NVMe to Lustre
# ============================================================================

# Combine the 8 files in /tmp into a new file in /tmp
time srun -n $SLURM_JOB_NUM_NODES -N $SLURM_JOB_NUM_NODES python3 -u $EXAMPLE_DIR/combine_adios.py $NVME_DIR

# Combine all files into a single tar file on each node
time srun -n $SLURM_JOB_NUM_NODES -N $SLURM_JOB_NUM_NODES bash -c "cd $NVME_DIR && tar -cf inference_fused_results-\$SLURMD_NODENAME.tar inference_fused_results_all-*.bp"

# Copy the combined file to Orion
time srun -n $SLURM_JOB_NUM_NODES -N $SLURM_JOB_NUM_NODES bash -c "cp -r $NVME_DIR/inference_fused_results-\$SLURMD_NODENAME.tar $RESULTS_DIR/."


# ============================================================================
# Omnistat teardown — stop exporters, query data, stop server
# ============================================================================

if [ "$_omnistat_started" = "1" ]; then
    ${OMNISTAT_WRAPPER} usermode --stop
    echo "Omnistat data saved to external database"
fi

# ============================================================================
# Done
# ============================================================================

echo "============================================================"
echo "Job complete.  Exit code: $_srun_exit"
echo "Results:       $RESULTS_DIR"
echo "============================================================"

exit "$_srun_exit"
