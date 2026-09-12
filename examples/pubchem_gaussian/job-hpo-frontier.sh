#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J PubChem-Gaussian-HPO
#SBATCH -o pubchem-hpo-%j.out
#SBATCH -e pubchem-hpo-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 4

set -euo pipefail

: "${HYDRAGNN_ROOT:?Set HYDRAGNN_ROOT to the HydraGNN checkout}"

export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_GRAPH_PARALLEL_GROUP_SIZE=1
export HYDRAGNN_VALTEST=1
export TASKS_PER_NODE=1
export NNODES_PER_TRIAL=1
export NUM_CONCURRENT_TRIALS="${SLURM_JOB_NUM_NODES}"
export DEEPHYPER_LOG_DIR="pubchem-hpo-logs-${SLURM_JOB_ID}"
export DEEPHYPER_SEARCH_DIR="pubchem-hpo-${SLURM_JOB_ID}"

# The PubChem cache must be created before launching concurrent trials:
# srun -N1 -n1 python examples/pubchem_gaussian/train.py --preonly
python -u "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_gaussian_hpo.py" \
    --max-evals 100
