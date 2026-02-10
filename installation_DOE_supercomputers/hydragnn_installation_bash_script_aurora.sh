#!/usr/bin/env bash
# hydragnn_installation_bash_script_aurora.sh
#
# Aurora approach (with ADIOS2/DDStore build-from-source):
# - PyTorch: use "module load frameworks" (do NOT pip-install torch wheels)
# - PyG: install base torch_geometric in a venv that inherits system site-packages
# - Build ADIOS2 from source (MPI + Python) with DataSpaces engine OFF
# - IMPORTANT on Aurora: ADIOS2 Python bindings are NOT a pip project.
#   We link the built/installed python package into the venv site-packages.
# - Install DDStore from source
# - Install torch_geometric (base only)
# - Install HydraGNN editable
#
# Run:
#   nohup ./hydragnn_installation_bash_script_aurora.sh > installation_aurora.log 2>&1 &

set -Eeuo pipefail
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

_nounset_off() {
  NOUNSET_WAS_ON=0
  case "$-" in
    *u*) NOUNSET_WAS_ON=1; set +u ;;
  esac
}
_nounset_restore() {
  if [[ "${NOUNSET_WAS_ON:-0}" -eq 1 ]]; then set -u; fi
  unset NOUNSET_WAS_ON
}

banner "Starting HydraGNN Aurora environment setup ($(date))"

# ============================================================
# Modules
# ============================================================
banner "Modules: use Aurora-provided frameworks"

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

_nounset_off
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

subbanner 'module reset'
module reset

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
# Create venv
# ============================================================
banner "Create Python venv (inherits frameworks site-packages)"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
RECREATE_ENV="${RECREATE_ENV:-0}"
echo "VENV_PATH    = $VENV_PATH"
echo "RECREATE_ENV = $RECREATE_ENV"

PYTHON_BIN="$(command -v python3 || true)"
[[ -z "$PYTHON_BIN" ]] && PYTHON_BIN="$(command -v python || true)"
[[ -z "$PYTHON_BIN" ]] && { echo "❌ python/python3 not found after module load frameworks"; exit 1; }

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

PYTHON_XY="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "Python X.Y in venv: ${PYTHON_XY}"

VENV_SITEPKG="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
echo "VENV_SITEPKG = $VENV_SITEPKG"

# ============================================================
# pip helpers + numpy pin
# ============================================================
banner "pip bootstrap"
python -m pip install -U pip setuptools wheel

CONSTRAINTS_FILE="${INSTALL_ROOT}/pip-constraints.txt"
cat > "$CONSTRAINTS_FILE" <<'EOF'
numpy==1.26.4
EOF
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"
echo "PIP_CONSTRAINT = $PIP_CONSTRAINT"
cat "$CONSTRAINTS_FILE"

_pip_filter_unsatisfied() {
  python - "$@" <<'PY'
import sys
from pathlib import Path

args = sys.argv[1:]

def is_option(a: str) -> bool: return a.startswith("-")
def is_localish(a: str) -> bool:
  if a in (".","..") or a.startswith(("./","../","/")): return True
  if "://" in a or a.startswith(("git+","hg+","svn+","bzr+","file:")): return True
  if a.endswith(".txt") and Path(a).exists(): return True
  return False

tokens=[]
skip_next=False
for a in args:
  if skip_next: skip_next=False; continue
  if is_option(a):
    if a in ("-r","--requirement","-c","--constraint","-t","--target","--prefix","--root","--index-url","--extra-index-url","--find-links"):
      skip_next=True
    continue
  tokens.append(a)

try:
  from packaging.requirements import Requirement
  from importlib import metadata
except Exception:
  print("\n".join(tokens))
  raise SystemExit(0)

unsatisfied=[]
for t in tokens:
  if is_localish(t) or t=="-e":
    unsatisfied.append(t); continue
  try:
    req=Requirement(t)
  except Exception:
    unsatisfied.append(t); continue
  try:
    ver=metadata.version(req.name)
  except metadata.PackageNotFoundError:
    unsatisfied.append(t); continue
  if req.specifier and (ver not in req.specifier):
    unsatisfied.append(t); continue

print("\n".join(unsatisfied))
PY
}

pip_retry() {
  local tries=3 delay=3
  local -a raw_args=("$@")

  local do_filter=1
  for a in "${raw_args[@]}"; do
    case "$a" in
      "."|"-e"|./*|../*|/*|git+*|http*|https*|file:*)
        do_filter=0; break;;
    esac
  done

  local -a to_install=()
  if [[ "$do_filter" -eq 1 ]]; then
    mapfile -t to_install < <(_pip_filter_unsatisfied "${raw_args[@]}" || true)
  else
    to_install=("${raw_args[@]}")
  fi

  if [[ "${#to_install[@]}" -eq 0 ]]; then
    echo "✅ pip_skip: all requested requirements already satisfied: $*"
    return 0
  fi

  echo "pip will install (unsatisfied only): ${to_install[*]}"
  for ((i=1; i<=tries; i++)); do
    if python -m pip install --upgrade-strategy only-if-needed "${to_install[@]}"; then
      return 0
    fi
    echo "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

banner "Pin NumPy to 1.26.4 inside venv (shadow system-site numpy)"
pip_retry "numpy==1.26.4"
python - <<'PY'
import numpy as np
print("numpy.__version__ =", np.__version__)
print("numpy.__file__    =", np.__file__)
PY

# ============================================================
# Sanity: torch
# ============================================================
banner "Sanity check: torch import (from frameworks)"
python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.xpu.is_available() =", hasattr(torch, "xpu") and torch.xpu.is_available())
PY

# ============================================================
# ADIOS2 build deps
# ============================================================
banner "Install pybind11 (required for ADIOS2 Python bindings)"
pip_retry pybind11

# ============================================================
# Build ADIOS2
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

MPICC_BIN="${MPICC_BIN:-$(command -v mpicc || true)}"
MPICXX_BIN="${MPICXX_BIN:-$(command -v mpicxx || true)}"
[[ -z "$MPICC_BIN" || -z "$MPICXX_BIN" ]] && { echo "❌ mpicc/mpicxx not found"; exit 1; }

PYTHON_EXEC="$(command -v python)"
PYTHON3_INCLUDE_DIR="$(python - <<'PY'
import sysconfig
print(sysconfig.get_path("include"))
PY
)"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$ADIOS2_INSTALL" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DADIOS2_BUILD_TESTING=OFF \
  -DADIOS2_BUILD_EXAMPLES_EXPERIMENTAL=OFF \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_USE_Python=ON \
  -DPython_EXECUTABLE="$PYTHON_EXEC" \
  -DPython3_EXECUTABLE="$PYTHON_EXEC" \
  -DPython3_INCLUDE_DIR="$PYTHON3_INCLUDE_DIR" \
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

# ============================================================
# Runtime paths (lib vs lib64)
# ============================================================
export ADIOS2_DIR="$ADIOS2_INSTALL"
export PATH="$ADIOS2_INSTALL/bin:$PATH"
if [[ -d "$ADIOS2_INSTALL/lib64" ]]; then ADIOS2_LIBDIR="lib64"; else ADIOS2_LIBDIR="lib"; fi
export LD_LIBRARY_PATH="$ADIOS2_INSTALL/${ADIOS2_LIBDIR}:${LD_LIBRARY_PATH:-}"

# ============================================================
# FIX: ADIOS2 Python bindings were installed under a *nested absolute path*
# We recover by linking the actually-installed package into the venv site-packages.
# ============================================================
banner "Fix ADIOS2 Python install location and ensure import works"

ADIOS2_PY_BAD_ROOT="${ADIOS2_INSTALL}${VENV_SITEPKG}"
ADIOS2_PY_BAD_PKG="${ADIOS2_PY_BAD_ROOT}/adios2"

if [[ -d "$ADIOS2_PY_BAD_PKG" ]]; then
  echo "Detected nested ADIOS2 Python install:"
  echo "  $ADIOS2_PY_BAD_PKG"
  rm -rf "$VENV_SITEPKG/adios2" 2>/dev/null || true
  ln -s "$ADIOS2_PY_BAD_PKG" "$VENV_SITEPKG/adios2"
  echo "Symlinked into venv:"
  echo "  $VENV_SITEPKG/adios2 -> $ADIOS2_PY_BAD_PKG"
else
  echo "Did not find nested python package at:"
  echo "  $ADIOS2_PY_BAD_PKG"
  echo "Trying standard locations..."

  # Standard candidate locations
  CANDIDATES=(
    "${ADIOS2_INSTALL}/${ADIOS2_LIBDIR}/python${PYTHON_XY}/site-packages/adios2"
    "${ADIOS2_INSTALL}/lib/python${PYTHON_XY}/site-packages/adios2"
    "${ADIOS2_INSTALL}/lib64/python${PYTHON_XY}/site-packages/adios2"
    "${ADIOS2_SRC}/build/lib/python${PYTHON_XY}/site-packages/adios2"
    "${ADIOS2_SRC}/build/lib64/python${PYTHON_XY}/site-packages/adios2"
  )

  FOUND=""
  for c in "${CANDIDATES[@]}"; do
    if [[ -d "$c" ]]; then FOUND="$c"; break; fi
  done

  if [[ -n "$FOUND" ]]; then
    rm -rf "$VENV_SITEPKG/adios2" 2>/dev/null || true
    ln -s "$FOUND" "$VENV_SITEPKG/adios2"
    echo "Symlinked into venv:"
    echo "  $VENV_SITEPKG/adios2 -> $FOUND"
  else
    echo "❌ Could not locate ADIOS2 python package in standard locations either."
    echo "Searched:"
    printf '  - %s\n' "${CANDIDATES[@]}"
    exit 1
  fi
fi

banner "Sanity check: import adios2"
python - <<'PY'
import adios2
print("adios2 import OK")
print("adios2 version:", getattr(adios2, "__version__", "unknown"))
print("adios2 file:", adios2.__file__)
PY

# ============================================================
# DDStore
# ============================================================
banner "DDStore (clone + pip install .)"
DDSTORE_FRONTIER="${INSTALL_ROOT}/DDStore-Source"
export DDSTORE_FRONTIER

if [[ ! -d "${DDSTORE_FRONTIER}/DDStore/.git" ]]; then
  mkdir -p "$DDSTORE_FRONTIER"
  pushd "$DDSTORE_FRONTIER" >/dev/null
  git clone https://github.com/ORNL/DDStore.git
  popd >/dev/null
fi

pushd "${DDSTORE_FRONTIER}/DDStore" >/dev/null
pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# PyTorch Geometric (base only)
# ============================================================
banner "Install PyTorch Geometric base (torch_geometric)"
pip_retry torch_geometric

python - <<'PY'
import torch, torch_geometric
import numpy as np
print("torch =", torch.__version__)
print("torch_geometric =", torch_geometric.__version__)
print("numpy =", np.__version__, "from", np.__file__)
PY

# ============================================================
# Additional deps (keep numpy==1.26.4)
# ============================================================
banner "Install HydraGNN dependencies (do NOT install torch here)"
pip_retry scipy pyyaml requests tqdm filelock psutil
pip_retry networkx jinja2
pip_retry tensorboard scikit-learn pytest
pip_retry ase h5py lmdb
pip_retry "mendeleev<1.1.0"
pip_retry rdkit jarvis-tools pymatgen || true
pip_retry igraph || true

banner "Re-check NumPy pin (must be 1.26.4 in venv)"
pip_retry "numpy==1.26.4"
python - <<'PY'
import numpy as np
print("numpy.__version__ =", np.__version__)
print("numpy.__file__    =", np.__file__)
PY

# ============================================================
# HydraGNN editable
# ============================================================
banner "Install HydraGNN (editable)"
HYDRAGNN_SRC="${HYDRAGNN_SRC:-${PWD}/HydraGNN}"
echo "HYDRAGNN_SRC = $HYDRAGNN_SRC"

if [[ -d "$HYDRAGNN_SRC" ]]; then
  pip_retry -e "$HYDRAGNN_SRC"
else
  echo "⚠️  HydraGNN source directory not found at: $HYDRAGNN_SRC"
fi

banner "Final Summary"
cat <<EOF
Base install:        $INSTALL_ROOT
Venv:               $VENV_PATH
Venv site-packages: $VENV_SITEPKG
Constraints:        $CONSTRAINTS_FILE (numpy==1.26.4)

ADIOS2 install:     $ADIOS2_INSTALL
ADIOS2 libdir:      $ADIOS2_LIBDIR
ADIOS2 runtime:
  export ADIOS2_DIR=$ADIOS2_INSTALL
  export PATH=$ADIOS2_INSTALL/bin:\$PATH
  export LD_LIBRARY_PATH=$ADIOS2_INSTALL/${ADIOS2_LIBDIR}:\$LD_LIBRARY_PATH

NOTE:
- If ADIOS2 installs python into the nested path "$ADIOS2_INSTALL$VENV_SITEPKG",
  this script auto-links it back into "$VENV_SITEPKG/adios2".
EOF

echo "✅ Aurora HydraGNN environment setup complete!"