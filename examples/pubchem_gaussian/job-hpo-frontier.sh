#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J PubChem-Gaussian-HPO
#SBATCH -o pubchem-hpo-%j.out
#SBATCH -e pubchem-hpo-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 4

set -euo pipefail

module unload darshan-runtime 2>/dev/null || true
module load cpe/24.07 cce/18.0.0 rocm/7.2.0 amd-mixed/7.2.0 \
    craype-accel-amd-gfx90a PrgEnv-gnu miniforge3/23.11.0-0 git-lfs

HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HYDRAGNN_VENV="${HYDRAGNN_VENV:-/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
: "${PUBCHEM_DATASET:=/lustre/orion/lrn070/world-shared/kmehta/hydragnn/datasets/pubchem_gaussian.bp}"
HPO_MPNN_TYPES="${HPO_MPNN_TYPES:-EGNN,SchNet,DimeNet,MACE,PAINN,PNAEq,AllScAIP,UMA}"
HPO_MAX_EVALS="${HPO_MAX_EVALS:-100}"
[[ -x "${HYDRAGNN_VENV}/bin/python" ]] || { echo "Missing ${HYDRAGNN_VENV}/bin/python" >&2; exit 1; }
[[ -e "${PUBCHEM_DATASET}" ]] || { echo "Missing ${PUBCHEM_DATASET}" >&2; exit 1; }

export PATH="${HYDRAGNN_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
export PUBCHEM_DATASET
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
    --dataset-path "${PUBCHEM_DATASET}" \
    --mpnn-types "${HPO_MPNN_TYPES}" \
    --max-evals "${HPO_MAX_EVALS}"
