#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J OPF4661-ER
#SBATCH -o job-opf4661-er-%j.out
#SBATCH -e job-opf4661-er-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 8

set -euo pipefail

: "${CONFIG:?Submit with CONFIG set to an OPF configuration path}"
: "${RUN_NAME:?Submit with RUN_NAME set to a unique run name}"

HYDRAGNN_ROOT=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN
HYDRAGNN_ENV=${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
OPF_DIR=${HYDRAGNN_ROOT}/examples/opf
DATASET_NAME=case4661_effective_resistance_qk

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "${HYDRAGNN_ENV}"

export PYTHONPATH=${HYDRAGNN_ROOT}:${HYDRAGNN_ENV}/lib/python3.11/site-packages:${PYTHONPATH:-}
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

module unload darshan-runtime

PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${PLUGIN_PATH}/lib
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export NCCL_NET_PLUGIN=${PLUGIN_PATH}/lib/librccl-net.so
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1

cd "${OPF_DIR}"

if [[ ! -d "dataset/${DATASET_NAME}.h5" || ! -f "dataset/${DATASET_NAME}.h5/meta.h5" ]]; then
    echo "Missing dataset/${DATASET_NAME}.h5; preprocessing dependency did not produce it." >&2
    exit 1
fi

echo "Configuration: ${CONFIG}"
echo "Run name:      ${RUN_NAME}"
echo "Dataset:       ${OPF_DIR}/dataset/${DATASET_NAME}.h5"

srun --export=ALL,HYDRAGNN_DIAG=1,HYDRAGNN_DIAG_RANK=0 \
    -N"${SLURM_JOB_NUM_NODES}" -n$((SLURM_JOB_NUM_NODES * 8)) -c7 \
    --gpus-per-task=1 --gpu-bind=closest \
    python -u train_opf_solution_heterogeneous.py \
    --hdf5 \
    --inputfile "${CONFIG}" \
    --modelname "${DATASET_NAME}" \
    --log "${RUN_NAME}"
