#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J lpe-sign-sweep
#SBATCH -o job-lpe-sign-sweep-%j.out
#SBATCH -e job-lpe-sign-sweep-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1

set -eo pipefail

HYDRAGNN_ROOT=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN
HYDRAGNN_ENV=${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
OUTPUT_DIR=${OUTPUT_DIR:-${HYDRAGNN_ROOT}/examples/opf/logs/lpe_sign_sweep_${SLURM_JOB_ID}}

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "${HYDRAGNN_ENV}"

export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3

cd "${HYDRAGNN_ROOT}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u examples/opf/evaluate_lpe_sign_sweep.py \
    --device cuda \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "TensorBoard results: ${OUTPUT_DIR}"
