#!/bin/bash
#SBATCH -A LRN087
#SBATCH -J pglearn_pre
#SBATCH -o pglearn-preonly-%j.out
#SBATCH -e pglearn-preonly-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HYDRAGNN_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
FRONTIER_VENV_BIN=${FRONTIER_VENV_BIN:-${HYDRAGNN_ROOT}/HydraGNN-Installation-Frontier-ROCm713/hydragnn_venv_rocm713/bin}

source "${SCRIPT_DIR}/module-to-load-frontier-rocm713.sh"
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}
export PATH="${FRONTIER_VENV_BIN}:${PATH}"
export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"

export http_proxy=${http_proxy:-http://proxy.ccs.ornl.gov:3128/}
export https_proxy=${https_proxy:-http://proxy.ccs.ornl.gov:3128/}
export no_proxy=${no_proxy:-localhost,127.0.0.0/8,*.ccs.ornl.gov}
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export NO_PROXY="${no_proxy}"

NHOSTS=${SLURM_JOB_NUM_NODES}
NGPU_PER_HOST=${SLURM_GPUS_ON_NODE:-8}
NGPUS="$((NHOSTS * NGPU_PER_HOST))"
export MASTER_ADDR=$(srun --overlap -N 1 -n 1 --nodelist=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1) hostname -I | awk '{print $1}')
export MASTER_PORT=${MASTER_PORT:-29500}

export OMP_NUM_THREADS=7
export PYTHONWARNINGS=ignore
export TMPDIR=/tmp
export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

# ROCm 7.13 settings adapted from lumina-sdk Frontier jobs.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export MIOPEN_DISABLE_CACHE=${MIOPEN_DISABLE_CACHE:-1}
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export ROCM_HOME=${ROCM_PATH:-}

cd "${SCRIPT_DIR}"

echo "python: $(which python)"
echo "job: ${SLURM_JOB_ID}"
echo "nodes: ${NHOSTS}, gpus/node: ${NGPU_PER_HOST}, total ranks: ${NGPUS}"

REPO=${REPO:-PGLearn/PGLearn-Small}
CASE_NAME=${CASE_NAME:-14_ieee}
FORMULATION=${FORMULATION:-ACOPF}
TASK=${TASK:-auto}
MODELNAME=${MODELNAME:-PGLearn_Solution_Hetero}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pglearn_solution_heterogeneous.py}
FORMAT_FLAG=${FORMAT_FLAG:---hdf5}
MAX_SAMPLES_ARG=${MAX_SAMPLES_ARG:-}

srun -N ${NHOSTS} \
    --ntasks=${NGPUS} \
    --ntasks-per-node=${NGPU_PER_HOST} \
    -c7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    "${SCRIPT_DIR}/launch_rank_frontier.sh" \
    python -u "${TRAIN_SCRIPT}" ${FORMAT_FLAG} --preonly \
    --repo "${REPO}" --case_name "${CASE_NAME}" --formulation "${FORMULATION}" \
    --task "${TASK}" --modelname "${MODELNAME}" ${MAX_SAMPLES_ARG}
