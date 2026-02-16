#!/usr/bin/env bash
# setup_env_perlmutter.sh
# Complete automated setup for HydraGNN environment and dependencies on NERSC Perlmutter (CUDA/A100).
#
# CORRECTIONS INCLUDED:
#  1) Robust conda shell initialization (no deprecated `source activate`)
#  2) Avoid PyG binary wheels that can require newer GLIBC (e.g. GLIBC_2.33)
#     -> build PyG compiled deps from source on Perlmutter
#  3) Ensure a modern GCC is used for building PyTorch C++ extensions
#     -> load gcc-native/13.2 and force CC/CXX to gcc/g++
#
# Usage:
#   chmod +x setup_env_perlmutter.sh
#   ./setup_env_perlmutter.sh
#
# Optional env vars:
#   VENV_PATH=/path/to/env
#   PYTHON_VERSION=3.11
#   RECREATE_ENV=1
#   EXPECTED_CUDA_MM=12.4
#   TORCH_CUDA_TAG=cu124
#   BUILD_PYG_LIB=0   # default 0 (skip pyg-lib). set 1 to try building from source.
#   TORCH_CUDA_ARCH_LIST=8.0
#   MAX_JOBS=16
#   INSTALL_ROOT=/some/path/HydraGNN-Installation-Perlmutter

set -Eeuo pipefail

# =========================
# Pretty printing helpers
# =========================
hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

banner "Starting HydraGNN environment setup on Perlmutter ($(date))"

# ============================================================
# Module initialization
# ============================================================
banner "Configure Perlmutter Modules"
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    source /usr/share/lmod/lmod/init/bash
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
  fi
fi

if ! command -v module >/dev/null 2>&1; then
  echo "❌ 'module' command not found. Ensure you're running on Perlmutter login/compute nodes."
  exit 1
fi

# Cray "hard reset" (avoids warnings about module reset not fully restoring defaults)
if [[ -f /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh || true
fi

module reset
ml nersc-default/1.0 || true

# Cray programming environment + MPI
ml cpe/24.07
ml PrgEnv-gnu/8.5.0
ml cray-mpich/8.1.30

# A100 target (SM80)
ml craype-accel-nvidia80

# CUDA toolkit (match PyTorch wheels tag below)
EXPECTED_CUDA_MM="${EXPECTED_CUDA_MM:-12.4}"
ml "cudatoolkit/${EXPECTED_CUDA_MM}"

# Modern compiler toolchain for PyTorch C++ extensions
# (Fixes torch-sparse build failing with "too old version of GCC")
ml gcc-native/13.2

# Build helpers
ml cmake/3.30.2 || ml cmake/3.24.3 || true

# Conda
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0 || true

# ============================================================
# Installation root
# ============================================================
banner "Set Base Installation Directory"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/HydraGNN-Installation-Perlmutter}"
mkdir -p "$INSTALL_ROOT"
echo "All installation components will be contained in: $INSTALL_ROOT"
cd "$INSTALL_ROOT"

# ============================================================
# Conda shell init + env creation
# ============================================================
banner "Initialize Conda + Create/Activate Environment"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda command not found (Miniforge module not loaded?)"
  exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1090
  eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)" 2>/dev/null || true
fi

VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RECREATE_ENV="${RECREATE_ENV:-0}"

echo "Virtual environment path: $VENV_PATH"
echo "Python version: ${PYTHON_VERSION}"

if [[ -d "$VENV_PATH" && "$RECREATE_ENV" -eq 1 ]]; then
  echo "Removing existing conda environment at: $VENV_PATH"
  conda deactivate >/dev/null 2>&1 || true
  conda env remove -p "$VENV_PATH" -y || rm -rf "$VENV_PATH"
fi

if [[ -d "$VENV_PATH" ]]; then
  echo "Conda environment already exists at $VENV_PATH"
else
  echo "Creating conda environment at $VENV_PATH with Python $PYTHON_VERSION"
  conda create -y -p "$VENV_PATH" python="$PYTHON_VERSION"
fi

conda activate "$VENV_PATH"
echo "Python in use: $(which python)"
python --version

# Cray libs (often helpful)
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# Force modern compiler for all C++ extensions built via torch.utils.cpp_extension
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

# Verify compiler is modern enough
subbanner "Compiler sanity check (must be GCC >= 9)"
echo "CC=$(which ${CC})"
echo "CXX=$(which ${CXX})"
${CXX} --version | head -n 1

# CUDA build env hints
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A100
export MAX_JOBS="${MAX_JOBS:-16}"

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(dirname "$(which nvcc)")")" && pwd)"
fi

# ============================================================
# pip helpers + NumPy pin
# ============================================================
banner "pip Helpers and NumPy Pin (1.26.4)"

PIP_FLAGS=(--upgrade-strategy only-if-needed)

pip_retry() {
  local tries=3 delay=3
  for ((i=1; i<=tries; i++)); do
    if pip install "${PIP_FLAGS[@]}" "$@"; then
      return 0
    fi
    echo "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

assert_numpy_1264() {
  python - <<'PY'
import numpy as np
expected="1.26.4"
assert np.__version__==expected, f"NumPy is {np.__version__}, expected {expected}"
PY
}

subbanner "Upgrade pip/setuptools/wheel"
pip_retry --disable-pip-version-check -U pip setuptools wheel

subbanner "Install and pin numpy==1.26.4"
pip_retry "numpy==1.26.4"
assert_numpy_1264

# ============================================================
# Core scientific Python deps
# ============================================================
banner "Install Core Python Packages"

pip_retry ninja
pip_retry cmake
pip_retry astunparse
pip_retry expecttest
pip_retry hypothesis
pip_retry numpy==1.26.4
pip_retry psutil==7.1.0
pip_retry pyyaml
pip_retry requests
pip_retry setuptools
pip_retry typing-extensions
pip_retry sympy==1.14.0
pip_retry filelock
pip_retry networkx
pip_retry jinja2
pip_retry tqdm==4.67.1
pip_retry types-dataclasses
pip_retry scipy==1.14.1
pip_retry pyparsing
pip_retry build
pip_retry Cython
pip_retry tensorboard==2.20.0
pip_retry scikit-learn==1.5.1
pip_retry pytest
pip_retry ase==3.26.0
pip_retry rdkit
pip_retry jarvis-tools
pip_retry pymatgen
pip_retry igraph
pip_retry mendeleev==0.16.0
pip_retry lmdb
pip_retry h5py==3.14.0
pip_retry tensorflow
pip_retry tensorflow_datasets
pip_retry vesin==0.4.2

# ============================================================
# CUDA-aware PyTorch (pip wheels)
# ============================================================
banner "Install CUDA PyTorch (Before PyG)"

# Match cudatoolkit/12.4 => cu124 wheels
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu124}"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

subbanner "Install PyTorch from ${PYTORCH_INDEX_URL}"
pip_retry --index-url "${PYTORCH_INDEX_URL}" torch torchvision
assert_numpy_1264

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu name =", torch.cuda.get_device_name(0))
PY

# ============================================================
# PyTorch-Geometric stack (SOURCE BUILDS to avoid GLIBC mismatch)
# ============================================================
banner "PyTorch-Geometric Stack (Build from Source to Avoid GLIBC Issues)"

# IMPORTANT:
#   - Avoid wheels that may require GLIBC_2.33 (Perlmutter system glibc is older)
#   - Force source builds for compiled extensions

PYG_DIR_NAME="PyTorch-Geometric-${TORCH_CUDA_TAG}-source"
PYG_PERLMUTTER="${INSTALL_ROOT}/${PYG_DIR_NAME}"
export PYG_PERLMUTTER
mkdir -p "$PYG_PERLMUTTER"
cd "$PYG_PERLMUTTER"

subbanner "Uninstall any existing PyG components to avoid mixing wheels/source"
pip uninstall -y pyg-lib torch-sparse torch-scatter torch-cluster torch-spline-conv torch-geometric >/dev/null 2>&1 || true

subbanner "Build/install PyG compiled deps from source (no wheels)"
# Note: torch-scatter built for you already; torch-sparse previously failed due to old GCC.
# With gcc-native/13.2 + CC/CXX forced, torch-sparse should compile.

pip_retry --no-binary :all: --no-build-isolation torch-scatter
pip_retry --no-binary :all: --no-build-isolation torch-sparse
pip_retry --no-binary :all: --no-build-isolation torch-cluster
pip_retry --no-binary :all: --no-build-isolation torch-spline-conv

# pyg-lib is optional; many HydraGNN workloads run without it.
BUILD_PYG_LIB="${BUILD_PYG_LIB:-0}"
if [[ "$BUILD_PYG_LIB" -eq 1 ]]; then
  subbanner "Build pyg-lib from source (optional; may fail depending on toolchain)"
  pip_retry --no-binary :all: --no-build-isolation pyg-lib || true
else
  subbanner "Skipping pyg-lib (set BUILD_PYG_LIB=1 to attempt source build)"
fi

subbanner "Install torch-geometric (pure python wrapper package)"
pip_retry torch-geometric
assert_numpy_1264

subbanner "Install e3nn and openequivariance"
pip_retry e3nn openequivariance --verbose
assert_numpy_1264

subbanner "PyG import sanity check"
python - <<'PY'
import torch
import torch_geometric
print("torch:", torch.__version__)
print("pyg:", torch_geometric.__version__)

mods = ["pyg_lib", "torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv"]
for m in mods:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL ({e})")
PY

# ============================================================
# mpi4py
# ============================================================
banner "mpi4py (v4.1.1)"

MPI4PY_PERLMUTTER="${INSTALL_ROOT}/MPI4PY-Perlmutter"
export MPI4PY_PERLMUTTER
mkdir -p "$MPI4PY_PERLMUTTER"
cd "$MPI4PY_PERLMUTTER"

git clone -b 4.1.1 https://github.com/mpi4py/mpi4py.git || true
pushd mpi4py >/dev/null
rm -rf build
CC=cc MPICC=cc pip_retry . --verbose
popd >/dev/null

# ============================================================
# ADIOS2
# ============================================================
banner "ADIOS2 (v2.10.2)"

ADIOS2_PERLMUTTER="${INSTALL_ROOT}/ADIOS2-Perlmutter"
export ADIOS2_PERLMUTTER
mkdir -p "$ADIOS2_PERLMUTTER"
cd "$ADIOS2_PERLMUTTER"

if [[ ! -d ADIOS2/.git ]]; then
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git
fi

mkdir -p adios2-build

CC=cc CXX=CC FC=ftn \
cmake -DCMAKE_INSTALL_PREFIX="$VENV_PATH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_Fortran=OFF \
    -DADIOS2_BUILD_EXAMPLES_EXPERIMENTAL=OFF \
    -DADIOS2_BUILD_TESTING=OFF \
    -DADIOS2_USE_HDF5=OFF \
    -DADIOS2_USE_SST=OFF \
    -DADIOS2_USE_BZip2=OFF \
    -DADIOS2_USE_PNG=OFF \
    -DADIOS2_USE_DataSpaces=OFF \
    -DADIOS2_USE_Python=ON \
    -DPython_EXECUTABLE="$(which python)" \
    -B adios2-build -S ADIOS2

cmake --build adios2-build -j32
cmake --install adios2_build 2>/dev/null || cmake --install adios2-build

# ============================================================
# DDStore
# ============================================================
banner "DDStore"

DDSTORE_PERLMUTTER="${INSTALL_ROOT}/DDStore-Perlmutter"
export DDSTORE_PERLMUTTER
mkdir -p "$DDSTORE_PERLMUTTER"
cd "$DDSTORE_PERLMUTTER"

git clone https://github.com/ORNL/DDStore.git || true
pushd DDStore >/dev/null
CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# DeepHyper
# ============================================================
banner "DeepHyper (develop branch)"

DEEPHYPER_PERLMUTTER="${INSTALL_ROOT}/DeepHyperPerlmutter"
export DEEPHYPER_PERLMUTTER
mkdir -p "$DEEPHYPER_PERLMUTTER"
cd "$DEEPHYPER_PERLMUTTER"

git clone https://github.com/deephyper/deephyper.git || true
cd deephyper
git fetch origin develop
git checkout develop
pip_retry -e ".[hps,hps-tl]" --verbose
assert_numpy_1264

# ============================================================
# GPTL
# ============================================================
banner "GPTL"
GPTL_PERLMUTTER="${INSTALL_ROOT}/GPTLPerlmutter"
export GPTL_PERLMUTTER
mkdir -p "$GPTL_PERLMUTTER"
cd "$GPTL_PERLMUTTER"

wget https://github.com/jmrosinski/GPTL/releases/download/v8.1.1/gptl-8.1.1.tar.gz
tar xvf gptl-8.1.1.tar.gz
pushd gptl-8.1.1 >/dev/null
./configure --prefix=$INSTALL_ROOT --disable-libunwind CC=cc CXX=CC FC=ftn
make install
popd >/dev/null

git clone git@github.com:jychoi-hpc/gptl4py.git || true
pushd gptl4py >/dev/null
GPTL_DIR=$INSTALL_ROOT CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# Final Summary
# ============================================================
banner "Final Summary"
cat <<EOF
Base install:        $INSTALL_ROOT
Virtual environment: $VENV_PATH

Modules baseline:
  cpe/24.07
  PrgEnv-gnu/8.5.0
  cray-mpich/8.1.30
  craype-accel-nvidia80
  cudatoolkit/${EXPECTED_CUDA_MM}
  gcc-native/13.2
  conda/Miniforge3-24.11.3-0 (or fallback)

PyTorch:
  CUDA wheel tag:    ${TORCH_CUDA_TAG}
  Index URL:         ${PYTORCH_INDEX_URL}

PyTorch-Geometric:
  Built from source: torch-scatter, torch-sparse, torch-cluster, torch-spline-conv
  pyg-lib:           $( [[ "${BUILD_PYG_LIB}" -eq 1 ]] && echo "attempted from source" || echo "skipped" )

MPI4PY:              $MPI4PY_PERLMUTTER
ADIOS2:              $ADIOS2_PERLMUTTER
DDStore:             $DDSTORE_PERLMUTTER
DeepHyper:           $DEEPHYPER_PERLMUTTER
EOF

echo "✅ HydraGNN-Installation-Perlmutter environment setup complete!"
echo ""
echo "Module load + activation (for future sessions):"
cat <<EOF
module reset
ml nersc-default/1.0 || true
ml cpe/24.07
ml PrgEnv-gnu/8.5.0
ml cray-mpich/8.1.30
ml craype-accel-nvidia80
ml cudatoolkit/${EXPECTED_CUDA_MM}
ml gcc-native/13.2
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0 || true
source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || eval "\$(\"\$(conda info --base)/bin/conda\" shell.bash hook)"
conda activate ${VENV_PATH}
EOF

