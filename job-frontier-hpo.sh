#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -N 8626
#SBATCH -C nvme
##SBATCH --signal=SIGUSR1@180

set -euo pipefail

# ============================================================
# MPI / ROCm / Runtime environment
# ============================================================

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=NUMA
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi
export PYTHONNOUSERSITE=1

# ============================================================
# FIXED INSTALL LOCATION
# ============================================================

INSTALL_ROOT="/lustre/orion/lrn070/world-shared/AmSC_HydraGNN_GFM"

INSTALL_TARBALL="HydraGNN-Installation-Frontier.tar.gz"
INSTALL_DIR="HydraGNN-Installation-Frontier"

MODULE_SCRIPT="${INSTALL_ROOT}/module-to-load-frontier-rocm640.sh"
VENV_PATH="${INSTALL_ROOT}/${INSTALL_DIR}/hydragnn_venv"

# Force Python 3.11 everywhere
export ENV_PYVER="3.11"

# ============================================================
# Node health utilities
# ============================================================

function check_node()
{
    [ ! -d .node.status ] && mkdir .node.status
    ssh "$1" hostname 2> /dev/null
    [ $? -eq 0 ] && touch ".node.status/$1"
}

function check_badnodes()
{
    for NODE in $(scontrol show hostnames); do
        check_node "$NODE" &
    done
    wait

    ls .node.status/* | tail -n 2 | xargs rm -f || true

    BAD_NODELIST=""
    for NODE in $(scontrol show hostnames); do
        [ ! -f ".node.status/$NODE" ] && BAD_NODELIST="$NODE,$BAD_NODELIST"
    done
    [ -n "${BAD_NODELIST}" ] && BAD_NODELIST="${BAD_NODELIST%,}"

    export HYDRAGNN_EXCLUDE_NODELIST="${BAD_NODELIST}"
    echo "HYDRAGNN_EXCLUDE_NODELIST: ${HYDRAGNN_EXCLUDE_NODELIST}"
}

# ============================================================
# Ensure HydraGNN install tree exists
# ============================================================

function setup_install_tree()
{
    local TAR="${INSTALL_ROOT}/${INSTALL_TARBALL}"
    local DIR="${INSTALL_ROOT}/${INSTALL_DIR}"

    echo "INSTALL_ROOT    = ${INSTALL_ROOT}"
    echo "INSTALL_TARBALL = ${INSTALL_TARBALL}"
    echo "INSTALL_DIR     = ${INSTALL_DIR}"
    echo "MODULE_SCRIPT   = ${MODULE_SCRIPT}"
    echo "VENV_PATH       = ${VENV_PATH}"

    if [[ ! -f "${TAR}" ]]; then
        echo "ERROR: tarball not found: ${TAR}"
        exit 1
    fi

    if [[ ! -d "${DIR}" ]]; then
        echo "Untarring HydraGNN installation..."
        tar -xzf "${TAR}" -C "${INSTALL_ROOT}"
    else
        echo "Install directory already present."
    fi

    if [[ ! -f "${MODULE_SCRIPT}" ]]; then
        echo "ERROR: module script not found: ${MODULE_SCRIPT}"
        exit 1
    fi

    if [[ ! -d "${VENV_PATH}" ]]; then
        echo "ERROR: venv not found: ${VENV_PATH}"
        exit 1
    fi
}

# ============================================================
# Run HPO with proper sourced environment
# ============================================================

function run_hpo()
{
    export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

    # ---------------- HPO configuration ----------------
    export NNODES="${SLURM_JOB_NUM_NODES}"
    export NNODES_PER_TRIAL=128
    export NUM_CONCURRENT_TRIALS=$(( NNODES / NNODES_PER_TRIAL ))

    export NTOTGPUS=$(( NNODES * 8 ))
    export NGPUS_PER_TRIAL=$(( 8 * NNODES_PER_TRIAL ))
    export NTOT_DEEPHYPER_RANKS=$(( NTOTGPUS / NGPUS_PER_TRIAL ))

    export OMP_NUM_THREADS=4

    if [ "${NTOTGPUS}" -lt $(( NGPUS_PER_TRIAL * NUM_CONCURRENT_TRIALS )) ]; then
        echo "ERROR: Not enough GPUs"
        exit 1
    fi

    export DEEPHYPER_LOG_DIR="deephyper-experiment-${SLURM_JOB_ID}"
    mkdir -p "${DEEPHYPER_LOG_DIR}"
    export DEEPHYPER_DB_HOST="$(hostname)"

    # ---------------- Bad node handling ----------------
    BAD_NODELIST=""
    if [ -f "omnistat_failed_hosts.${SLURM_JOB_ID}.out" ]; then
        BAD_NODELIST="$(tr '\n' ',' < omnistat_failed_hosts.${SLURM_JOB_ID}.out)"
        BAD_NODELIST="${BAD_NODELIST%,}"
    fi
    export HYDRAGNN_EXCLUDE_NODELIST="${BAD_NODELIST}"

    # ---------------- Runtime environment ----------------
    # shellcheck disable=SC1090
    source "${MODULE_SCRIPT}"

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    fi

    # shellcheck disable=SC1090
    source activate "${VENV_PATH}"

    # ADIOS2 v2.10.2 Python path (Python 3.11)
    export PYTHONPATH="${VENV_PATH}/lib/python${ENV_PYVER}/site-packages:${PYTHONPATH}"

    echo "========== Runtime check =========="
    which python
    python -c "import sys; print(sys.version)"
    python -c "import torch; print('torch', torch.__version__, torch.__file__)"
    python -c "import adios2; print('adios2', adios2.__version__, adios2.__file__)" || true
    echo "PYTHONPATH=${PYTHONPATH}"
    echo "==================================="

    python gfm_deephyper_multi.py
}

# ============================================================
# Main
# ============================================================

SRC_DIR="examples/multidataset_hpo"
WDIR="examples/multidataset_hpo_NN${SLURM_JOB_NUM_NODES}_${SLURM_JOB_ID}"

echo "workdir: ${WDIR}"

# Create working directory
mkdir -p "${WDIR}"

# Copy everything except the heavy datasets directory
rsync -a --exclude 'datasets' "${SRC_DIR}/" "${WDIR}/"

# Create a symbolic link for datasets
ln -s "$(realpath "${SRC_DIR}/datasets")" "${WDIR}/datasets"

# Sanity check
if [ ! -L "${WDIR}/datasets" ]; then
    echo "ERROR: datasets is not a symlink!"
    exit 1
fi

cd "${WDIR}"

# Ensure installation exists
setup_install_tree

# Run inside login shell to preserve conda semantics
bash -lc "$(declare -f run_hpo); run_hpo"

echo "Done."
