#!/bin/bash
# Retrain the no-physics HEAT baseline while monitoring DC and AC violations.

#SBATCH -A LRN070
#SBATCH -J OPF-H-MON500
#SBATCH -o job-opf-heat-monitor-case500-%j.out
#SBATCH -e job-opf-heat-monitor-case500-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1

set -euo pipefail

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
RUN_NAME=${RUN_NAME:-heat_attr_case500_monitor_physics_${SLURM_JOB_ID}}

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "$HYDRAGNN_ROOT/HydraGNN-Installation-Frontier/hydragnn_venv"

export PYTHONPATH=$HYDRAGNN_ROOT:${PYTHONPATH:-}
export PYTHONPATH=$HYDRAGNN_ROOT/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH

module unload darshan-runtime

export OMP_NUM_THREADS=7
export HYDRAGNN_FORCE_DDP=1
export HYDRAGNN_MASTER_PORT_RETRIES=16
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1
export GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS

cd "$HYDRAGNN_ROOT/examples/opf"

echo "Starting monitor-only case500 HEAT training at $(date)"
echo "RUN_NAME=$RUN_NAME"
echo "TRAINING_OBJECTIVE=MSE only"
echo "MONITORED_CONSTRAINTS=voltage, angle, DC flow, AC apparent flow"
echo "RANKS=8 GPUS=8"

srun --export=ALL,HYDRAGNN_DIAG=1,HYDRAGNN_DIAG_RANK=0 \
    -N1 -n8 -c7 \
    --gpus-per-task=1 --gpu-bind=closest \
    python -u train_opf_solution_heterogeneous.py \
    --hdf5 \
    --modelname OPF_Solution_Hetero_case500 \
    --log "$RUN_NAME" \
    --inputfile physics-experiments/heat_attr_case500_monitor_physics.json

echo "Finished monitor-only case500 HEAT training at $(date): $RUN_NAME"
