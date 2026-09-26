#!/bin/bash
#SBATCH -A LRN087
#SBATCH -J PubChem-FSDP2-Memory
#SBATCH -o pubchem-fsdp2-memory-%j.out
#SBATCH -e pubchem-fsdp2-memory-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 8

set -euo pipefail

module unload darshan-runtime 2>/dev/null || true
module load cpe/24.07 cce/18.0.0 rocm/7.2.0 amd-mixed/7.2.0 \
    craype-accel-amd-gfx90a PrgEnv-gnu miniforge3/23.11.0-0 git-lfs

HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
HYDRAGNN_VENV="${HYDRAGNN_VENV:-/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72}"
PUBCHEM_DATASET="${PUBCHEM_DATASET:-${HYDRAGNN_ROOT}/examples/pubchem_gaussian/dataset/pubchem_gaussian.bp}"
CONFIG="${HYDRAGNN_ROOT}/examples/pubchem_gaussian/pubchem_gaussian_dimenet_oom.json"
RESULT_DIR="${RESULT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/pubchem-fsdp2-memory-${SLURM_JOB_ID}}"

[[ -x "${HYDRAGNN_VENV}/bin/python" ]] || { echo "Missing ${HYDRAGNN_VENV}/bin/python" >&2; exit 1; }
[[ -e "${PUBCHEM_DATASET}" ]] || { echo "Missing ${PUBCHEM_DATASET}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Missing ${CONFIG}" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"

export PATH="${HYDRAGNN_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
export HYDRAGNN_GRAPH_PARALLEL_GROUP_SIZE=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_VALTEST=1
export OMP_NUM_THREADS=7
RUN_DDP="${RUN_DDP:-1}"
RUN_FSDP2="${RUN_FSDP2:-1}"
if [[ "${RUN_DDP}" != "1" && "${RUN_FSDP2}" != "1" ]]; then
    echo "At least one of RUN_DDP or RUN_FSDP2 must be 1" >&2
    exit 2
fi

run_mode() {
    local mode="$1"
    local log_path="${RESULT_DIR}/${mode}.log"
    local -a fsdp_args=()

    if [[ "${mode}" == "fsdp2" ]]; then
        export HYDRAGNN_USE_FSDP=1
        export HYDRAGNN_FSDP_VERSION=2
        export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD
        export HYDRAGNN_VERIFY_FSDP2_SHARDING=1
        fsdp_args+=(--allow-experimental-fsdp2)
    else
        export HYDRAGNN_USE_FSDP=0
        unset HYDRAGNN_FSDP_VERSION HYDRAGNN_FSDP_STRATEGY \
            HYDRAGNN_VERIFY_FSDP2_SHARDING
    fi

    set +e
    srun --exclusive --exact -N 8 -n 64 --ntasks-per-node=8 \
        --gpus-per-node=8 --gpus-per-task=1 --gpu-bind=closest \
        --kill-on-bad-exit=1 \
        "${HYDRAGNN_VENV}/bin/python" -u \
        "${HYDRAGNN_ROOT}/examples/pubchem_gaussian/train.py" \
        --inputfile="${CONFIG}" \
        --dataset-path="${PUBCHEM_DATASET}" \
        --num-epoch=1 \
        --num-train-samples=64 \
        --num-val-samples=64 \
        --num-test-samples=64 \
        --subset-seed=0 \
        --log="pubchem_dimenet_${mode}_${SLURM_JOB_ID}" \
        --adios \
        "${fsdp_args[@]}" >"${log_path}" 2>&1
    local status=$?
    set -e

    local outcome="failed"
    if grep -q -E 'OutOfMemoryError|out of memory' "${log_path}"; then
        outcome="oom"
    elif [[ ${status} -eq 0 ]]; then
        outcome="completed"
    fi
    printf '%s\texit_status=%d\toutcome=%s\n' "${mode}" "${status}" "${outcome}" \
        | tee -a "${RESULT_DIR}/summary.txt"
    grep -E 'fsdp2-sharding|Max memory allocated after optimizer step|OutOfMemoryError|out of memory' \
        "${log_path}" | tail -n 16 >>"${RESULT_DIR}/summary.txt" || true
}

: >"${RESULT_DIR}/summary.txt"
if [[ "${RUN_DDP}" == "1" ]]; then
    run_mode ddp
fi
if [[ "${RUN_FSDP2}" == "1" ]]; then
    run_mode fsdp2
fi
cat "${RESULT_DIR}/summary.txt"