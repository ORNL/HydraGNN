#!/usr/bin/env bash
# hydragnn_installation_bash_script_aurora.sh
#
# Aurora approach (with ADIOS2/DDStore build-from-source):
# - PyTorch: use "module load frameworks" (do NOT pip-install torch+xpu wheels)
#   https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/
# - PyG: on Aurora, install base torch_geometric in a venv that inherits system site-packages
#   https://docs.alcf.anl.gov/aurora/data-science/frameworks/pyg/#pyg-on-aurora
# - Robust Lmod/module handling under `set -u` (nohup-safe)
# - Load Aurora "frameworks" for PyTorch (XPU) via modules (do NOT pip-install torch wheels)
# - Build ADIOS2 from source (MPI + Python) with DataSpaces engine OFF (as in Frontier script)
# - Install DDStore from source (clone + pip install .) (independent of ADIOS2 DataSpaces engine)
# - Install PyTorch Geometric base only (torch_geometric) per ALCF guidance
# - Install HydraGNN editable
#
# Notes:
# - ADIOS2 is installed under: ${INSTALL_ROOT}/adios2
# - DDStore is installed into the venv site-packages via pip
# - We intentionally do NOT install DataSpaces / enable ADIOS2_USE_DataSpaces (kept OFF)
#
# Run:
#   nohup ./hydragnn_installation_bash_script_aurora.sh > installation_aurora.log 2>&1 &

set -Eeuo pipefail

# --- Make Lmod safe under nounset/non-interactive shells (nohup) ---
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

# Save/restore nounset helper
_nounset_off() {
  NOUNSET_WAS_ON=0
  case "$-" in
    *u*) NOUNSET_WAS_ON=1; set +u ;;
  esac
}
_nounset_restore() {
  if [[ "${NOUNSET_WAS_ON:-0}" -eq 1 ]]; then
    set -u
  fi
  unset NOUNSET_WAS_ON
}

banner "Starting HydraGNN Aurora environment setup ($(date))"

# ============================================================
# Modules (robust under `set -u` + nohup)
# ============================================================
banner "Modules: use Aurora-provided frameworks"

# Ensure module command exists
if ! command -v module >/dev/null 2>&1; then
  _nounset_off
  export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
  _nounset_restore
fi

if ! command -v module >/dev/null 2>&1; then
  echo "❌ 'module' command not found after init. Are you on Aurora?"
  exit 1
fi

# Run module commands with nounset disabled (Lmod internals may reference unset vars)
_nounset_off
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

subbanner 'module reset'
module reset

# Optional: uncomment only if you need extra packages from /soft
# module use /soft/modulefiles

subbanner "Load Aurora provided PyTorch (XPU) via frameworks module"
module load frameworks

subbanner "Loaded modules"
module -t list 2>&1 || true

_nounset_restore

# ============================================================
# Installation root
# ============================================================
banner "Set Base Installation Directory"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/HydraGNN-Installation-Aurora}"
mkdir -p "$INSTALL_ROOT"
echo "INSTALL_ROOT = $INSTALL_ROOT"

# ============================================================
# Create venv that inherits frameworks packages
# ============================================================
banner "Create Python venv (inherits frameworks site-packages)"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
RECREATE_ENV="${RECREATE_ENV:-0}"
echo "VENV_PATH    = $VENV_PATH"
echo "RECREATE_ENV = $RECREATE_ENV"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ python/python3 not found after module load frameworks"
  exit 1
fi

echo "Python used for venv: $PYTHON_BIN"
"$PYTHON_BIN" --version

if [[ -d "$VENV_PATH" && "$RECREATE_ENV" -eq 1 ]]; then
  subbanner "Removing existing venv: $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  subbanner "Creating venv (--system-site-packages)"
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_PATH"
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Determine python X.Y for PYTHONPATH additions later
PYTHON_XY="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "Python X.Y in venv: ${PYTHON_XY}"

# ============================================================
# pip helpers
# ============================================================
banner "pip bootstrap"
python -m pip install -U pip setuptools wheel

pip_retry() {
  local tries=3 delay=3
  for ((i=1; i<=tries; i++)); do
    if python -m pip install --upgrade-strategy only-if-needed "$@"; then
      return 0
    fi
    echo "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

# ============================================================
# Sanity check: PyTorch import must work (from frameworks)
# ============================================================
banner "Sanity check: torch import (from frameworks)"
python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.xpu.is_available() =", hasattr(torch, "xpu") and torch.xpu.is_available())
if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("xpu device_count =", torch.xpu.device_count())
PY

# ============================================================
# IMPORTANT: ensure pybind11 is installed BEFORE ADIOS2 configure
# ============================================================
banner "Install pybind11 (required for ADIOS2 Python bindings)"
python -m pip install --upgrade-strategy only-if-needed pybind11

# ============================================================
# Build ADIOS2 from source (Frontier-parity: DataSpaces engine OFF)
# ============================================================
banner "ADIOS2 (build from source; DataSpaces engine OFF)"
ADIOS2_VERSION="${ADIOS2_VERSION:-v2.10.2}"

ADIOS2_SRC="${INSTALL_ROOT}/ADIOS2-src"
ADIOS2_BUILD="${ADIOS2_SRC}/build"
ADIOS2_INSTALL="${INSTALL_ROOT}/adios2"

echo "ADIOS2_VERSION = $ADIOS2_VERSION"
echo "ADIOS2_INSTALL = $ADIOS2_INSTALL"

if [[ ! -d "${ADIOS2_SRC}/.git" ]]; then
  git clone https://github.com/ornladios/ADIOS2.git "$ADIOS2_SRC"
fi

pushd "$ADIOS2_SRC" >/dev/null
git fetch --all --tags
git checkout "$ADIOS2_VERSION"
popd >/dev/null

mkdir -p "$ADIOS2_BUILD"
pushd "$ADIOS2_BUILD" >/dev/null

# Use MPI compilers in environment (provided by Aurora stack)
MPICC_BIN="${MPICC_BIN:-$(command -v mpicc || true)}"
MPICXX_BIN="${MPICXX_BIN:-$(command -v mpicxx || true)}"
if [[ -z "$MPICC_BIN" || -z "$MPICXX_BIN" ]]; then
  echo "❌ mpicc/mpicxx not found. Ensure frameworks (and MPI) are available."
  exit 1
fi

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$ADIOS2_INSTALL" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DADIOS2_BUILD_TESTING=OFF \
  -DADIOS2_BUILD_EXAMPLES_EXPERIMENTAL=OFF \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_USE_Python=ON \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DADIOS2_USE_HDF5=OFF \
  -DADIOS2_USE_SST=OFF \
  -DADIOS2_USE_BZip2=OFF \
  -DADIOS2_USE_PNG=OFF \
  -DADIOS2_USE_DataSpaces=OFF \
  -DADIOS2_USE_DataMan=OFF \
  -DADIOS2_USE_CUDA=OFF \
  -DADIOS2_USE_HIP=OFF \
  -DADIOS2_USE_SYCL=OFF \
  -DCMAKE_C_COMPILER="$MPICC_BIN" \
  -DCMAKE_CXX_COMPILER="$MPICXX_BIN"

cmake --build . -j"${ADIOS2_BUILD_JOBS:-16}"
cmake --install .

popd >/dev/null

# Export runtime + python paths so the venv can import adios2 from this build
export ADIOS2_DIR="$ADIOS2_INSTALL"
export PATH="$ADIOS2_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$ADIOS2_INSTALL/lib:$LD_LIBRARY_PATH"
# ADIOS2 python bindings typically live under .../lib/pythonX.Y/site-packages
export PYTHONPATH="$ADIOS2_INSTALL/lib/python${PYTHON_XY}/site-packages:${PYTHONPATH:-}"

# ============================================================
# DDStore (clone + pip install .) — Frontier-style
# ============================================================
banner "DDStore (clone + pip install .)"
DDSTORE_FRONTIER="${INSTALL_ROOT}/DDStore-Source"
export DDSTORE_FRONTIER

if [[ ! -d "${DDSTORE_FRONTIER}/DDStore/.git" ]]; then
  mkdir -p "$DDSTORE_FRONTIER"
  pushd "$DDSTORE_FRONTIER" >/dev/null
  # Use HTTPS for portability (no ssh key requirement)
  git clone https://github.com/ORNL/DDStore.git
  popd >/dev/null
fi

pushd "${DDSTORE_FRONTIER}/DDStore" >/dev/null
# Build/install into venv
pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# PyTorch Geometric (base only)
# ============================================================
banner "Install PyTorch Geometric base (torch_geometric)"
pip_retry torch_geometric

python - <<'PY'
import torch
import torch_geometric
print("torch =", torch.__version__)
print("torch_geometric =", torch_geometric.__version__)
print("xpu available =", hasattr(torch, "xpu") and torch.xpu.is_available())
PY

# ============================================================
# Additional python deps (NO torch install here!)
# ============================================================
banner "Install HydraGNN dependencies (do NOT install torch here)"
pip_retry "numpy<2" scipy pyyaml requests tqdm filelock psutil
pip_retry networkx jinja2
pip_retry tensorboard scikit-learn pytest
pip_retry ase h5py lmdb
pip_retry mendeleev
pip_retry rdkit jarvis-tools pymatgen || true
pip_retry igraph || true

# ============================================================
# Sanity check: ADIOS2 + DataSpaces bindings (user-provided block)
# ============================================================
banner "Sanity check: ADIOS2 + DataSpaces Python bindings"

python - <<'PY'
import adios2
print("adios2 version:", adios2.__version__)

try:
    import pyddstore
    print("pyddstore available")
except ImportError as e:
    print("WARNING: pyddstore not importable:", e)

try:
    import thapi
    print("thapi available")
except ImportError as e:
    print("WARNING: thapi not importable:", e)
PY

# ============================================================
# Install HydraGNN (editable)
# ============================================================
banner "Install HydraGNN (editable)"
HYDRAGNN_SRC="${HYDRAGNN_SRC:-${PWD}/HydraGNN}"
echo "HYDRAGNN_SRC = $HYDRAGNN_SRC"

if [[ -d "$HYDRAGNN_SRC" ]]; then
  pip_retry -e "$HYDRAGNN_SRC"
else
  echo "⚠️  HydraGNN source directory not found at: $HYDRAGNN_SRC"
  echo "    Export HYDRAGNN_SRC=/path/to/HydraGNN and rerun the HydraGNN install step."
fi

# ============================================================
# Final summary / activation instructions
# ============================================================
banner "Final Summary"
cat <<EOF
Base install:        $INSTALL_ROOT
Venv (system-site):  $VENV_PATH

ADIOS2:
  - source:   $ADIOS2_SRC @ $ADIOS2_VERSION
  - install:  $ADIOS2_INSTALL
  - NOTE: ADIOS2_USE_DataSpaces=OFF (Frontier-parity)

DDStore:
  - source:   ${DDSTORE_FRONTIER}/DDStore
  - python:   installed into venv (pip)

To activate later:
  module reset
  module load frameworks
  source ${VENV_PATH}/bin/activate

To ensure adios2 python bindings are visible in new shells:
  export ADIOS2_DIR=$ADIOS2_INSTALL
  export PATH=$ADIOS2_INSTALL/bin:\$PATH
  export LD_LIBRARY_PATH=$ADIOS2_INSTALL/lib:\$LD_LIBRARY_PATH
  export PYTHONPATH=$ADIOS2_INSTALL/lib/python${PYTHON_XY}/site-packages:\$PYTHONPATH
EOF

echo "✅ Aurora HydraGNN environment setup complete!"