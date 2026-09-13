#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J PubChem-Multistage-HPO
#SBATCH -o pubchem-multistage-hpo-%j.out
#SBATCH -e pubchem-multistage-hpo-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 4

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

# Four concurrent one-GPU trials per Frontier node. Reduce this if UMA or MACE
# establishes a lower memory limit during the screen stage.
TRIALS_PER_NODE="${TRIALS_PER_NODE:-4}"
CONCURRENCY=$((SLURM_JOB_NUM_NODES * TRIALS_PER_NODE))

python -u \
    "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_gaussian_multistage_hpo.py" \
    --initial-candidates 512 \
    --concurrency "${CONCURRENCY}" \
    --schedule "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_hpo_stages.json" \
    --output-dir "pubchem-multistage-hpo-${SLURM_JOB_ID}"
