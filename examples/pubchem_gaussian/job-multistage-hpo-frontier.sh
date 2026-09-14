#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J PubChem-Multistage-HPO
#SBATCH -o pubchem-multistage-hpo-%j.out
#SBATCH -e pubchem-multistage-hpo-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 16

set -euo pipefail
: "${HYDRAGNN_ROOT:?Set HYDRAGNN_ROOT to the HydraGNN checkout}"

export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_GRAPH_PARALLEL_GROUP_SIZE=1
export HYDRAGNN_VALTEST=1

# Set PREPROCESS_DATASET=0 when resuming from an already complete 3M cache.
PREPROCESS_DATASET="${PREPROCESS_DATASET:-1}"
DATASET_FORMAT="${DATASET_FORMAT:-adios}"
if [[ "${DATASET_FORMAT}" != "adios" && "${DATASET_FORMAT}" != "pickle" ]]; then
    echo "DATASET_FORMAT must be adios or pickle" >&2
    exit 2
fi
USE_DDSTORE="${USE_DDSTORE:-0}"
USE_SHMEM="${USE_SHMEM:-0}"
if [[ "${USE_DDSTORE}" == "1" && "${USE_SHMEM}" == "1" ]]; then
    echo "USE_DDSTORE and USE_SHMEM cannot both be enabled" >&2
    exit 2
fi
DATASET_OPTIONS=("--${DATASET_FORMAT}")
if [[ "${USE_DDSTORE}" == "1" ]]; then
    DATASET_OPTIONS+=(--ddstore)
    if [[ -n "${DDSTORE_WIDTH:-}" ]]; then
        DATASET_OPTIONS+=("--ddstore-width=${DDSTORE_WIDTH}")
    fi
fi
if [[ "${USE_SHMEM}" == "1" ]]; then
    DATASET_OPTIONS+=(--shmem)
fi
if [[ "${PREPROCESS_DATASET}" == "1" ]]; then
    srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" \
        --ntasks-per-node=1 \
        python -u "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/train.py" \
        --preonly --num-molecules 3000000 "--${DATASET_FORMAT}"
fi

# Each candidate is trained with DDP across four nodes and all eight GPUs per
# node. Four independent candidates run concurrently in the default 16-node
# allocation. Override NNODES_PER_TRIAL for larger or smaller DDP jobs.
NNODES_PER_TRIAL="${NNODES_PER_TRIAL:-4}"
TASKS_PER_NODE="${TASKS_PER_NODE:-8}"
if ((SLURM_JOB_NUM_NODES % NNODES_PER_TRIAL != 0)); then
    echo "SLURM_JOB_NUM_NODES must be divisible by NNODES_PER_TRIAL" >&2
    exit 2
fi
CONCURRENCY=$((SLURM_JOB_NUM_NODES / NNODES_PER_TRIAL))

python -u \
    "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_gaussian_multistage_hpo.py" \
    --initial-candidates 512 \
    --concurrency "${CONCURRENCY}" \
    --nodes-per-trial "${NNODES_PER_TRIAL}" \
    --tasks-per-node "${TASKS_PER_NODE}" \
    "${DATASET_OPTIONS[@]}" \
    --schedule "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_hpo_stages.json" \
    --output-dir "pubchem-multistage-hpo-${SLURM_JOB_ID}"
