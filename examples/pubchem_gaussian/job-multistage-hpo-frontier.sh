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
if [[ "${PREPROCESS_DATASET}" == "1" ]]; then
    srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" \
        --ntasks-per-node=1 \
        python -u "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/train.py" \
        --preonly --num-molecules 3000000
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
    --schedule "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_hpo_stages.json" \
    --output-dir "pubchem-multistage-hpo-${SLURM_JOB_ID}"
