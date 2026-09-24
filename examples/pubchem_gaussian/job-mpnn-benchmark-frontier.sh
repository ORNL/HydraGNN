#!/bin/bash
#SBATCH -A LRN087
#SBATCH -J pubchem-mpnn-benchmark
#SBATCH -o pubchem-mpnn-benchmark-%j.out
#SBATCH -e pubchem-mpnn-benchmark-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1

set -euo pipefail

module unload darshan-runtime 2>/dev/null || true
module load cpe/24.07 cce/18.0.0 rocm/7.2.0 amd-mixed/7.2.0 \
    craype-accel-amd-gfx90a PrgEnv-gnu miniforge3/23.11.0-0 git-lfs

ROOT="${HYDRAGNN_ROOT:-/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN}"
VENV="${HYDRAGNN_VENV:-/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
DATASET="${PUBCHEM_DATASET:-/lustre/orion/lrn070/world-shared/kmehta/hydragnn/datasets/pubchem_gaussian.bp}"
OUTPUT_DIR="${ROOT}/pubchem-mpnn-benchmark-${SLURM_JOB_ID}"
MODELS="${PUBCHEM_MODELS:-PAINN,MACE,SchNet,DimeNet,UMA,AllScAIP}"

export PATH="${VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_GRAPH_PARALLEL_GROUP_SIZE=1
export NCCL_NET_PLUGIN=none
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/${USER}/miopen-${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

cd "${ROOT}"
python -u examples/pubchem_gaussian/pubchem_gaussian_mpnn_benchmark.py \
    --dataset-path "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --models "${MODELS}" \
    --epochs 1 \
    --train-samples 64 \
    --val-samples 16 \
    --test-samples 16 \
    --subset-seed 0 \
    --concurrency 1