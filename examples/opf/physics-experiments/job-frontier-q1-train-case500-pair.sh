#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J OPF-Q1-C500
#SBATCH -o job-opf-q1-train-case500-pair-%j.out
#SBATCH -e job-opf-q1-train-case500-pair-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
##SBATCH -C nvme
##SBATCH -S 1

set -euo pipefail

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "$HYDRAGNN_ROOT/HydraGNN-Installation-Frontier/hydragnn_venv"

export PYTHONPATH=$HYDRAGNN_ROOT:${PYTHONPATH:-}
export PYTHONPATH=$HYDRAGNN_ROOT/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH

module unload darshan-runtime

export PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PLUGIN_PATH}/lib
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_NET_PLUGIN=${PLUGIN_PATH}/lib/librccl-net.so
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export OMP_NUM_THREADS=7
export HYDRAGNN_FORCE_DDP=1
export HYDRAGNN_MASTER_PORT_RETRIES=16
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1

# Each comparison run occupies one node and uses all eight GPUs via DDP.
unset NCCL_NET_PLUGIN NCCL_NET NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_SOCKET_IFNAME
export NCCL_P2P_LEVEL=SYS

cd "$HYDRAGNN_ROOT/examples/opf"

run_train() {
    local modelname=$1
    local run_name=$2
    local config_name=$3

    echo "Starting case500 training at $(date)"
    echo "MODELNAME=$modelname"
    echo "RUN_NAME=$run_name"
    echo "CONFIG_NAME=$config_name"
    echo "RANKS_PER_RUN=8"

    srun --exclusive --export=ALL,HYDRAGNN_DIAG=1,HYDRAGNN_DIAG_RANK=0 \
        -N1 -n8 -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_solution_heterogeneous.py \
        --hdf5 \
        --modelname "$modelname" \
        --log "$run_name" \
        --inputfile "physics-experiments/$config_name.json"

    echo "Finished case500 training at $(date): $run_name"
}

run_train \
    OPF_Solution_Hetero_case500 \
    heat_attr_case500_no_physics \
    heat_attr_case500_no_physics &
attr_pid=$!

run_train \
    OPF_Solution_Hetero_no_attr_case500 \
    heat_no_attr_case500_no_physics \
    heat_no_attr_case500_no_physics &
noattr_pid=$!

status=0
wait "$attr_pid" || status=$?
wait "$noattr_pid" || status=$?

if [[ $status -ne 0 ]]; then
    echo "At least one case500 Q1 training run failed at $(date)"
    exit "$status"
fi

echo "Both case500 Q1 training runs complete at $(date)"
