#!/usr/bin/env bash
# setup_env_rocm72.sh
# Complete automated setup for HydraGNN environment and dependencies on Frontier.
# Updated to ROCm 7.2 (module rocm/7.2.0) and PyTorch wheels from https://download.pytorch.org/whl/rocm7.2
#
# Derived from hydragnn_installation_bash_script_frontier-rocm71.sh.
# Key changes vs ROCm 7.1:
#   - Module stack:  rocm/7.2.0  amd-mixed/7.2.0
#   - EXPECTED_ROCM_MM="7.2"
#   - PYTORCH_ROCM_INDEX_URL="https://download.pytorch.org/whl/rocm7.2"
#   - Install root: HydraGNN-Installation-Frontier-ROCm72
#   - vLLM: No pre-built pip wheels exist for ROCm 7.2 (only Docker images).
#           vLLM is built from source using a pre-cloned source tree at
#           ${VLLM_SRC_DIR} (must be cloned before running on a compute node).
#
# Usage:
#   bash hydragnn_installation_bash_script_frontier-rocm72.sh
#
# Override variables:
#   VENV_PATH        Override conda env path
#   PYTHON_VERSION   Python version (default: 3.11)
#   RECREATE_ENV     Set to 1 to delete and recreate the conda env
#   VLLM_SRC_DIR     Path to pre-cloned vLLM source tree (default: auto-detected)
#   SKIP_VLLM        Set to 1 to skip vLLM build entirely
#   ROCM_MM          Override ROCm major.minor detection (e.g. "7.2")

set -Eeuo pipefail

# =========================
# Pretty printing helpers
# =========================
hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

banner "Starting HydraGNN environment setup (ROCm 7.2) ($(date))"

# ============================================================
# Module initialization & Frontier stack
# ============================================================
banner "Configure Frontier Modules"
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
  echo "⚠️  'module' command not found. Ensure you're running on the target HPC system."
else
  module reset
  ml cpe/24.07
  ml cce/18.0.0

  # --- ROCm 7.2 toolchain ---
  ml rocm/7.2.0
  ml amd-mixed/7.2.0

  ml craype-accel-amd-gfx90a
  ml PrgEnv-gnu
  ml miniforge3/23.11.0-0
  ml git-lfs
  module unload darshan-runtime
fi

# ============================================================
# Installation root
# ============================================================
banner "Set Base Installation Directory"
INSTALL_ROOT="${PWD}/HydraGNN-Installation-Frontier-ROCm72"
mkdir -p "$INSTALL_ROOT"
echo "All installation components will be contained in: $INSTALL_ROOT"
cd "$INSTALL_ROOT"

# ============================================================
# Env vars & Conda env creation
# ============================================================
banner "Create and Activate Conda Environment"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv_rocm72}"
echo "Virtual environment path: $VENV_PATH"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
echo "Python version: ${PYTHON_VERSION}"

RECREATE_ENV="${RECREATE_ENV:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ Conda command not found. Ensure miniforge3 is properly loaded."
  exit 1
fi

if [[ -d "$VENV_PATH" && "$RECREATE_ENV" -eq 1 ]]; then
  echo "Removing existing conda environment at: $VENV_PATH"
  source deactivate >/dev/null 2>&1 || true
  conda env remove -p "$VENV_PATH" -y || rm -rf "$VENV_PATH"
fi

if [[ -d "$VENV_PATH" ]]; then
  echo "Conda environment already exists at $VENV_PATH"
else
  echo "Creating conda environment at $VENV_PATH with Python $PYTHON_VERSION"
  conda create -y -p "$VENV_PATH" python="$PYTHON_VERSION"
fi

# shellcheck disable=SC1091
source activate "$VENV_PATH"
echo "Python in use: $(which python)"
python --version

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
# Core scientific Python dependencies
# ============================================================
banner "Install Core Python Packages"

pip_retry ninja
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
#pip_retry sqlite
pip_retry igraph
pip_retry mendeleev==0.16.0
pip_retry lmdb
pip_retry h5py==3.14.0
pip_retry tensorflow
pip_retry tensorflow_datasets
pip_retry vesin==0.4.2

# ============================================================
# ROCm detection + ROCm-aware PyTorch
# ============================================================
banner "ROCm Detection and ROCm-aware PyTorch Install (Before PyG)"
detect_rocm_mm() {
  local v=""
  if command -v module >/dev/null 2>&1; then
    local mlist
    mlist="$(module -t list 2>&1 || true)"
    v="$(grep -Eo 'rocm/[0-9]+\.[0-9]+' <<<"$mlist" | head -n1 | sed 's#rocm/##')"
  fi
  if [[ -z "$v" ]] && command -v hipcc >/dev/null 2>&1; then
    v="$(hipcc --version 2>&1 | grep -Eo 'HIP version:\s*[0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -n1 || true)"
  fi
  echo "$v"
}

# ---- ROCm 7.2 major.minor ----
EXPECTED_ROCM_MM="7.2"
ROCM_MM="${ROCM_MM:-$(detect_rocm_mm)}"
if [[ -z "$ROCM_MM" ]]; then
  echo "❌ Could not detect ROCm version. Ensure the rocm module is loaded."
  exit 1
fi
echo "Detected ROCm: $ROCM_MM"
if [[ "$ROCM_MM" != "$EXPECTED_ROCM_MM" ]]; then
  echo "❌ ROCm version mismatch. Detected $ROCM_MM but expecting rocm${EXPECTED_ROCM_MM}."
  exit 1
fi

# ---- PyTorch ROCm 7.2 index URL ----
PYTORCH_ROCM_INDEX_URL="https://download.pytorch.org/whl/rocm7.2"
subbanner "Install ROCm PyTorch from ${PYTORCH_ROCM_INDEX_URL}"
pip_retry --index-url "${PYTORCH_ROCM_INDEX_URL}" torch torchvision
assert_numpy_1264

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
print("torch.cuda.is_available() =", torch.cuda.is_available())  # ROCm uses torch.cuda API
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("GPU[0] =", torch.cuda.get_device_name(0))
PY

# ============================================================
# PyTorch-Geometric stack
# ============================================================
banner "PyTorch-Geometric Stack (ROCm ${ROCM_MM})"

# Recommended for MI250X builds
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a}"

PYG_DIR_NAME="PyTorch-Geometric-${ROCM_MM}"
PYG_FRONTIER="${INSTALL_ROOT}/${PYG_DIR_NAME}"
export PYG_FRONTIER
mkdir -p "$PYG_FRONTIER"
cd "$PYG_FRONTIER"

subbanner "pytorch_geometric (official)"
if [[ ! -d pytorch_geometric/.git ]]; then
  git clone --recursive git@github.com:pyg-team/pytorch_geometric.git
fi
pushd pytorch_geometric >/dev/null
rm -rf build
pip_retry . --verbose
assert_numpy_1264
popd >/dev/null

# --- pytorch_scatter (ROCm fork pinned) ---
subbanner "pytorch_scatter (ROCm fork pinned to 9799c51; temporary until upstream merges)"
if [[ ! -d pytorch_scatter/.git ]]; then
  # Official upstream (kept for reference; will switch back after merge):
  # git clone --recursive git@github.com:rusty1s/pytorch_scatter.git
  # Temporary ROCm fork (use until fixes merge upstream):
  git clone --recursive https://github.com/Looong01/pytorch_scatter-rocm.git pytorch_scatter
fi
pushd pytorch_scatter >/dev/null
git fetch --all
git checkout 9799c51
git submodule update --init --recursive
rm -rf build
CC=gcc CXX=g++ python setup.py build
CC=gcc CXX=g++ python setup.py install
assert_numpy_1264
echo "pytorch_scatter pinned to commit: $(git rev-parse --short HEAD)"
popd >/dev/null

# --- pytorch_sparse (ROCm fork pinned) ---
subbanner "pytorch_sparse (ROCm fork pinned to 2340737; temporary until upstream merges)"
if [[ ! -d pytorch_sparse/.git ]]; then
  # Official upstream (kept for reference; will switch back after merge):
  # git clone --recursive git@github.com:rusty1s/pytorch_sparse.git
  # Temporary ROCm fork (use until fixes merge upstream):
  git clone --recursive https://github.com/Looong01/pytorch_sparse-rocm.git pytorch_sparse
fi
pushd pytorch_sparse >/dev/null
git fetch --all
git checkout 2340737
rm -rf build
CC=gcc CXX=g++ python setup.py build
CC=gcc CXX=g++ python setup.py install
assert_numpy_1264
echo "pytorch_sparse pinned to commit: $(git rev-parse --short HEAD)"
popd >/dev/null

# --- pytorch_cluster (official pinned) ---
subbanner "pytorch_cluster (official @ 1.6.3-11-g4126a52)"
if [[ ! -d pytorch_cluster/.git ]]; then
  git clone --recursive git@github.com:rusty1s/pytorch_cluster.git
fi
pushd pytorch_cluster >/dev/null
git fetch --all
git checkout 1.6.3-11-g4126a52
rm -rf build
CC=gcc CXX=g++ python setup.py build
CC=gcc CXX=g++ python setup.py install
assert_numpy_1264
popd >/dev/null

# --- pytorch_spline_conv (official pinned) ---
subbanner "pytorch_spline_conv (official @ 1.2.2-9-ga6d1020)"
if [[ ! -d pytorch_spline_conv/.git ]]; then
  git clone --recursive git@github.com:rusty1s/pytorch_spline_conv.git
fi
pushd pytorch_spline_conv >/dev/null
git fetch --all
git checkout 1.2.2-9-ga6d1020
rm -rf build
CC=gcc CXX=g++ python setup.py build
CC=gcc CXX=g++ python setup.py install
assert_numpy_1264
popd >/dev/null

subbanner "Install e3nn"
pip_retry e3nn --verbose
assert_numpy_1264

# NOTE: unload ROCm for mpi4py build
module unload craype-accel-amd-gfx90a
module unload rocm
#######

# ============================================================
# mpi4py
# ============================================================
banner "mpi4py (v4.1.1)"
MPI4PY_FRONTIER="${INSTALL_ROOT}/MPI4PY-Frontier"
export MPI4PY_FRONTIER
mkdir -p "$MPI4PY_FRONTIER"
cd "$MPI4PY_FRONTIER"

git clone -b 4.1.1 https://github.com/mpi4py/mpi4py.git || true
pushd mpi4py >/dev/null
rm -rf build
CC=cc MPICC=cc pip_retry . --verbose
popd >/dev/null

# ============================================================
# ADIOS2
# ============================================================
banner "ADIOS2 (v2.10.2)"
ADIOS2_FRONTIER="${INSTALL_ROOT}/ADIOS2-Frontier"
export ADIOS2_FRONTIER
mkdir -p "$ADIOS2_FRONTIER"
cd "$ADIOS2_FRONTIER"

if [[ ! -d ADIOS2/.git ]]; then
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git
fi

mkdir -p adios2-build

CC=cc CXX=CC FC=ftn \
cmake -DCMAKE_INSTALL_PREFIX=$VENV_PATH \
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
    -DPython_EXECUTABLE=$(which python) \
    -B adios2-build -S ADIOS2

cmake --build adios2-build -j32
cmake --install adios2_build 2>/dev/null || cmake --install adios2-build

# ============================================================
# DDStore
# ============================================================
banner "DDStore"
DDSTORE_FRONTIER="${INSTALL_ROOT}/DDStore-Frontier"
export DDSTORE_FRONTIER
mkdir -p "$DDSTORE_FRONTIER"
cd "$DDSTORE_FRONTIER"

git clone git@github.com:ORNL/DDStore.git || true
pushd DDStore >/dev/null
CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# DeepHyper
# ============================================================
banner "DeepHyper (develop branch)"
DEEPHYPER_FRONTIER="${INSTALL_ROOT}/DeepHyperFrontier"
export DEEPHYPER_FRONTIER
mkdir -p "$DEEPHYPER_FRONTIER"
cd "$DEEPHYPER_FRONTIER"

git clone https://github.com/deephyper/deephyper.git || true
cd deephyper
pip_retry . --verbose
assert_numpy_1264

# ============================================================
# GPTL
# ============================================================
banner "GPTL"
GPTL_FRONTIER="${INSTALL_ROOT}/GPTLFrontier"
export GPTL_FRONTIER
mkdir -p "$GPTL_FRONTIER"
cd "$GPTL_FRONTIER"

wget https://github.com/jmrosinski/GPTL/releases/download/v8.1.1/gptl-8.1.1.tar.gz
tar xvf gptl-8.1.1.tar.gz
pushd gptl-8.1.1 >/dev/null
./configure --prefix=$VENV_PATH --disable-libunwind CC=cc CXX=CC FC=ftn
make install
popd >/dev/null

git clone https://github.com/jychoi-hpc/gptl4py.git || true
pushd gptl4py >/dev/null
GPTL_DIR=$VENV_PATH CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# vLLM (build from source — no pre-built ROCm 7.2 pip wheels)
#
# Why no pre-built wheels:
#   vLLM for ROCm is only distributed as Docker images (github.com/ROCm/vllm).
#   The main vllm-project PyPI wheel is CUDA-only.
#   There is no vllm+rocm7.2 wheel on PyPI or download.pytorch.org/whl/rocm7.2/.
#
# Source must be pre-cloned on the login node (compute nodes have no internet):
#   git clone --depth=1 --branch v0.20.0 \
#       https://github.com/vllm-project/vllm.git /path/to/vllm-src
#
# Set VLLM_SRC_DIR to the pre-cloned path, or it defaults to the standard
# location used by install_matsim_frontier.sh.
# Set SKIP_VLLM=1 to skip this section entirely.
# ============================================================
banner "vLLM (build from source for ROCm ${ROCM_MM})"

SKIP_VLLM="${SKIP_VLLM:-0}"
if [[ "$SKIP_VLLM" -eq 1 ]]; then
  echo "ℹ️  SKIP_VLLM=1: skipping vLLM build."
else
  # Re-load ROCm for the vLLM build (was unloaded for mpi4py above)
  ml rocm/7.2.0
  ml amd-mixed/7.2.0
  ml craype-accel-amd-gfx90a

  # Locate pre-cloned vLLM source
  # install_matsim_frontier.sh clones to: <project>/cache/vllm-src/vllm
  SCRIPT_DIR_VLLM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VLLM_SRC_DIR="${VLLM_SRC_DIR:-${SCRIPT_DIR_VLLM}/../../../cache/vllm-src/vllm}"
  VLLM_SRC_DIR="$(realpath -m "$VLLM_SRC_DIR")"

  if [[ ! -d "$VLLM_SRC_DIR/.git" ]]; then
    echo "❌ vLLM source tree not found at: $VLLM_SRC_DIR"
    echo "   On the login node, pre-clone it with:"
    echo "     git clone --depth=1 --branch v0.20.0 \\"
    echo "         https://github.com/vllm-project/vllm.git $VLLM_SRC_DIR"
    echo "   Then re-run this script on a compute node."
    echo "   Or set SKIP_VLLM=1 to skip vLLM entirely."
    exit 1
  fi

  VLLM_COMMIT="$(cd "$VLLM_SRC_DIR" && git rev-parse --short HEAD)"
  echo "Building vLLM from: $VLLM_SRC_DIR (commit: $VLLM_COMMIT)"

  # Install vLLM build-time dependencies first.
  # Frontier compute nodes have no outbound internet: these must be installed
  # while the login node is accessible (install_matsim_frontier.sh does this
  # by running: pip install -r requirements/rocm.txt  before submitting the job).
  VLLM_ROCM_REQ="$VLLM_SRC_DIR/requirements/rocm.txt"
  if [[ -f "$VLLM_ROCM_REQ" ]]; then
    subbanner "Installing vLLM ROCm requirements from $VLLM_ROCM_REQ"
    pip_retry -r "$VLLM_ROCM_REQ"
  else
    subbanner "vLLM ROCm requirements file not found; installing minimal build tooling"
    pip_retry cmake ninja packaging pybind11 setuptools_scm setuptools wheel amdsmi
  fi
  assert_numpy_1264

  subbanner "Building vLLM for gfx90a (ROCm 7.2)"
  cd "$VLLM_SRC_DIR"

  # Target only MI250X (gfx90a) to keep build times reasonable.
  # Remove the cache to avoid stale CUDA-targeted artifacts from any prior run.
  export PYTORCH_ROCM_ARCH="gfx90a"
  export ROCM_HOME="/opt/rocm-7.2.0"
  export HIP_HOME="/opt/rocm-7.2.0"

  # vLLM uses cmake; point it at ROCm 7.2.
  export CMAKE_PREFIX_PATH="/opt/rocm-7.2.0:${CMAKE_PREFIX_PATH:-}"

  pip_retry . --no-build-isolation --verbose

  python - <<'PY'
import vllm
print("vllm.__version__ =", vllm.__version__)
PY

  assert_numpy_1264

  # Apply flashinfer ROCm patch (no libcudart; use libamdhip64 fallback).
  # vLLM's own cuda_wrapper.py already handles this, but flashinfer/comm/cuda_ipc.py
  # has its own copy that only searches for libcudart. Patch it so every vLLM worker
  # subprocess can import flashinfer without error on ROCm.
  FLASHINFER_CUDA_IPC="$VENV_PATH/lib/python3.11/site-packages/flashinfer/comm/cuda_ipc.py"
  if [[ -f "$FLASHINFER_CUDA_IPC" ]]; then
    if grep -q "libamdhip64" "$FLASHINFER_CUDA_IPC"; then
      echo "flashinfer ROCm patch already applied; skipping."
    else
      echo "Patching flashinfer/comm/cuda_ipc.py for ROCm (libamdhip64 fallback)..."
      python - "$FLASHINFER_CUDA_IPC" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = (
    '    def __init__(self, so_file: Optional[str] = None):\n'
    '        if so_file is None:\n'
    '            so_file = find_loaded_library("libcudart")\n'
    '            assert so_file is not None, "libcudart is not loaded in the current process"'
)
new = (
    '    # ROCm/HIP: libamdhip64 uses hipXxx names instead of cudaXxx\n'
    '    def __init__(self, so_file: Optional[str] = None):\n'
    '        import os\n'
    '        is_hip = False\n'
    '        if so_file is None:\n'
    '            so_file = find_loaded_library("libcudart")\n'
    '            if so_file is None:\n'
    '                so_file = find_loaded_library("libamdhip64")\n'
    '                if so_file is not None:\n'
    '                    is_hip = True\n'
    '            if so_file is None:\n'
    '                so_file = os.environ.get("VLLM_CUDART_SO_PATH")\n'
    '                if so_file is not None and "amdhip" in so_file:\n'
    '                    is_hip = True\n'
    '            assert so_file is not None, "libcudart is not loaded in the current process"\n'
    '        else:\n'
    '            is_hip = "amdhip" in so_file'
)

if old in src:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("Patch applied successfully.")
else:
    print("WARNING: patch target not found — flashinfer may have changed. Manual review needed.")
PYEOF
    fi
  else
    echo "flashinfer not installed (or cuda_ipc.py not found); skipping patch."
  fi
fi  # end SKIP_VLLM

# ============================================================
# Final Summary
# ============================================================
banner "Final Summary"
cat <<EOF
Base install:        $INSTALL_ROOT
Virtual environment: $VENV_PATH
ROCm version:        ${ROCM_MM}
PyTorch index:       ${PYTORCH_ROCM_INDEX_URL}
PyTorch-Geometric:   $PYG_FRONTIER
  - pytorch_scatter fork: https://github.com/Looong01/pytorch_scatter-rocm.git @ 9799c51 (temporary)
  - pytorch_sparse fork:  https://github.com/Looong01/pytorch_sparse-rocm.git @ 2340737 (temporary)
  - pytorch_cluster:      1.6.3-11-g4126a52
  - pytorch_spline_conv:  1.2.2-9-ga6d1020
MPI4PY:              $MPI4PY_FRONTIER
ADIOS2:              $ADIOS2_FRONTIER
DDStore:             $DDSTORE_FRONTIER
DeepHyper:           $DEEPHYPER_FRONTIER
vLLM:                $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "SKIP_VLLM=1 or build failed")

vLLM notes:
  - No pre-built pip wheels exist for ROCm 7.2 (only Docker images via github.com/ROCm/vllm).
  - vLLM was built from source against ROCm 7.2 + torch+rocm7.2.
  - RCCL: use /opt/rocm-7.2.0/lib/librccl.so.1 (different build from ROCm 7.1.1).
    The ROCm 7.2.0 RCCL (2.27.7, commit fc0010cf6a) may fix the
    ncclDevKernel_Generic_4 HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION on gfx90a
    that was present in the ROCm 7.1.1 RCCL (commit 26aae437f6).
  - In smoke test: export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1
EOF

echo "✅ HydraGNN-ROCm72 environment setup complete!"
