#!/usr/bin/env bash
# hydragnn_installation_bash_script_aurora.sh
#
# Aurora approach (with ADIOS2/DDStore build-from-source):
# - PyTorch: use "module load frameworks" (do NOT pip-install torch wheels)
# - PyG: install base torch_geometric in a venv that inherits system site-packages
# - Build ADIOS2 from source (MPI + Python) with DataSpaces engine OFF
# - IMPORTANT on Aurora: ADIOS2 Python bindings are produced by CMake as a built module
#   under lib{,64}/pythonX.Y/site-packages (often in the *build* tree). We symlink that
#   into the venv site-packages so "import adios2" works reliably.
# - Install DDStore from source
# - Install torch_geometric (base only)
# - Install HydraGNN editable
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

# Determine python X.Y for path additions later
PYTHON_XY="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "Python X.Y in venv: ${PYTHON_XY}"

# Venv site-packages (for symlinks later)
VENV_SITEPKG="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
echo "VENV_SITEPKG = $VENV_SITEPKG"

# ============================================================
# pip helpers:
# - PIP_CONSTRAINT pins numpy to 1.26.4 (venv shadows system numpy>=2)
# - pip_retry skips requirements already satisfied (avoids churn on system-site-packages)
# ============================================================
banner "pip bootstrap"
python -m pip install -U pip setuptools wheel

CONSTRAINTS_FILE="${INSTALL_ROOT}/pip-constraints.txt"
cat > "$CONSTRAINTS_FILE" <<'EOF'
numpy==1.26.4
EOF
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"
echo "PIP_CONSTRAINT = $PIP_CONSTRAINT"
echo "Constraints file contents:"
cat "$CONSTRAINTS_FILE"

_pip_filter_unsatisfied() {
  python - "$@" <<'PY'
import sys
from pathlib import Path

args = sys.argv[1:]

def is_option(a: str) -> bool:
  return a.startswith("-")

def is_localish(a: str) -> bool:
  if a in (".", "..") or a.startswith(("./", "../", "/")):
    return True
  if "://" in a or a.startswith(("git+", "hg+", "svn+", "bzr+", "file:")):
    return True
  if a.endswith(".txt") and Path(a).exists():
    return True
  return False

tokens = []
skip_next = False
for a in args:
  if skip_next:
    skip_next = False
    continue
  if is_option(a):
    if a in ("-r","--requirement","-c","--constraint","-t","--target","--prefix","--root","--index-url","--extra-index-url","--find-links"):
      skip_next = True
    continue
  tokens.append(a)

try:
  from packaging.requirements import Requirement
  from importlib import metadata
except Exception:
  print("\n".join(tokens))
  raise SystemExit(0)

unsatisfied = []
for t in tokens:
  if is_localish(t) or t == "-e":
    unsatisfied.append(t)
    continue

  try:
    req = Requirement(t)
  except Exception:
    unsatisfied.append(t)
    continue

  name = req.name
  try:
    ver = metadata.version(name)
  except metadata.PackageNotFoundError:
    unsatisfied.append(t)
    continue

  if req.specifier and (ver not in req.specifier):
    unsatisfied.append(t)
    continue

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
        do_filter=0
        break
        ;;
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

# Ensure the venv has numpy==1.26.4 and that it shadows system numpy>=2.
banner "Pin NumPy to 1.26.4 inside venv (shadow system-site numpy)"
pip_retry "numpy==1.26.4"

python - <<'PY'
import numpy as np, sys
print("numpy.__version__ =", np.__version__)
print("numpy.__file__    =", np.__file__)
print("sys.prefix        =", sys.prefix)
PY

# ============================================================
# Sanity check: PyTorch import must work (from frameworks)
# ============================================================
banner "Sanity check: torch import (from frameworks)"
python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.xpu.is_available() =", hasattr(torch, "xpu") and torch.xpu.is_available())
PY

# ============================================================
# Ensure build deps for ADIOS2 Python bindings exist in *this* Python
# ============================================================
banner "Install pybind11 (required for ADIOS2 Python bindings)"
pip_retry pybind11

# mpi4py should already be in frameworks; keep this check lightweight
banner "Sanity check: mpi4py import (from frameworks/venv)"
python - <<'PY'
try:
    import mpi4py
    print("mpi4py =", mpi4py.__version__)
except Exception as e:
    print("WARNING: mpi4py not importable:", e)
PY

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

MPICC_BIN="${MPICC_BIN:-$(command -v mpicc || true)}"
MPICXX_BIN="${MPICXX_BIN:-$(command -v mpicxx || true)}"
if [[ -z "$MPICC_BIN" || -z "$MPICXX_BIN" ]]; then
  echo "❌ mpicc/mpicxx not found. Ensure frameworks (and MPI) are available."
  exit 1
fi

# Force CMake to use the venv Python (critical on Aurora so Python bindings are actually built)
PYTHON_EXEC="$(command -v python)"
PYTHON3_INCLUDE_DIR="$(python - <<'PY'
import sysconfig
print(sysconfig.get_path("include"))
PY
)"
PYTHON3_LIBRARY_HINT="$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
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
# Export runtime paths (Fix 1: lib vs lib64)
# ============================================================
export ADIOS2_DIR="$ADIOS2_INSTALL"
export PATH="$ADIOS2_INSTALL/bin:$PATH"

if [[ -d "$ADIOS2_INSTALL/lib64" ]]; then
  ADIOS2_LIBDIR="lib64"
else
  ADIOS2_LIBDIR="lib"
fi

export LD_LIBRARY_PATH="$ADIOS2_INSTALL/${ADIOS2_LIBDIR}:${LD_LIBRARY_PATH:-}"

# ============================================================
# Make ADIOS2 Python module importable
# Strategy:
#   1) Look for built python site-packages under build tree
#   2) Look for installed python site-packages under install tree
#   3) Symlink discovered "adios2" package (or adios2*.so) into venv site-packages
# ============================================================
banner "Locate ADIOS2 Python module and link into venv site-packages"

find_adios2_sitepkgs() {
  local base="$1"
  local libdir="$2"
  local pyxy="$3"
  local cand1="${base}/${libdir}/python${pyxy}/site-packages"
  local cand2="${base}/lib/python${pyxy}/site-packages"
  local cand3="${base}/lib64/python${pyxy}/site-packages"

  for c in "$cand1" "$cand2" "$cand3"; do
    if [[ -d "$c" ]]; then
      # Must contain adios2 package dir or adios2*.so
      if [[ -d "$c/adios2" ]] || compgen -G "$c/adios2*.so" >/dev/null 2>&1; then
        echo "$c"
        return 0
      fi
    fi
  done
  return 1
}

ADIOS2_PY_SITEPKG=""

# First prefer BUILD tree (most reliable on Aurora)
if ADIOS2_PY_SITEPKG="$(find_adios2_sitepkgs "${ADIOS2_SRC}/build" "lib"    "${PYTHON_XY}")"; then :; \
elif ADIOS2_PY_SITEPKG="$(find_adios2_sitepkgs "${ADIOS2_SRC}/build" "lib64" "${PYTHON_XY}")"; then :; \
elif ADIOS2_PY_SITEPKG="$(find_adios2_sitepkgs "${ADIOS2_INSTALL}" "${ADIOS2_LIBDIR}" "${PYTHON_XY}")"; then :; \
else
  echo "❌ Could not find ADIOS2 python module under build/install trees."
  echo "Searched for directories like:"
  echo "  - ${ADIOS2_SRC}/build/lib/python${PYTHON_XY}/site-packages"
  echo "  - ${ADIOS2_SRC}/build/lib64/python${PYTHON_XY}/site-packages"
  echo "  - ${ADIOS2_INSTALL}/${ADIOS2_LIBDIR}/python${PYTHON_XY}/site-packages"
  echo
  echo "Tip: check whether Python bindings were enabled during CMake configure."
  echo "      In CMake output, you should see Python being detected; otherwise bindings won't be built."
  exit 1
fi

echo "Found ADIOS2 python site-packages at: $ADIOS2_PY_SITEPKG"

mkdir -p "$VENV_SITEPKG"

# Link package directory if present
if [[ -d "$ADIOS2_PY_SITEPKG/adios2" ]]; then
  rm -rf "$VENV_SITEPKG/adios2" 2>/dev/null || true
  ln -s "$ADIOS2_PY_SITEPKG/adios2" "$VENV_SITEPKG/adios2"
  echo "Symlinked adios2 package dir into venv:"
  echo "  $VENV_SITEPKG/adios2 -> $ADIOS2_PY_SITEPKG/adios2"
fi

# Also link top-level extension module(s) if they exist (some layouts expose adios2*.so)
for so in "$ADIOS2_PY_SITEPKG"/adios2*.so; do
  if [[ -f "$so" ]]; then
    bn="$(basename "$so")"
    rm -f "$VENV_SITEPKG/$bn" 2>/dev/null || true
    ln -s "$so" "$VENV_SITEPKG/$bn"
    echo "Symlinked $bn into venv site-packages"
  fi
done

# Verify import now (and print its location)
banner "Sanity check: import adios2 (must succeed)"
python - <<'PY'
import adios2
print("adios2 version:", getattr(adios2, "__version__", "unknown"))
print("adios2 module:", adios2.__file__)
PY

# ============================================================
# DDStore (clone + pip install .) — Frontier-style
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
import torch, torch_geometric, numpy as np
print("torch =", torch.__version__)
print("torch_geometric =", torch_geometric.__version__)
print("xpu available =", hasattr(torch, "xpu") and torch.xpu.is_available())
print("numpy =", np.__version__, "from", np.__file__)
PY

# ============================================================
# Additional python deps (NO torch install here!)
# - PIP_CONSTRAINT enforces numpy==1.26.4 globally
# - mendeleev >= 1.1.0 requires numpy>=2; pin to <1.1.0 to remain compatible
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
# Sanity check: ADIOS2 + DDStore bindings
# ============================================================
banner "Sanity check: ADIOS2 + DDStore Python bindings"
python - <<'PY'
import adios2
print("adios2 version:", getattr(adios2, "__version__", "unknown"))

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
Venv site-packages:  $VENV_SITEPKG

Pip constraints:
  - file:   $CONSTRAINTS_FILE
  - numpy:  pinned to 1.26.4 inside venv (shadows system numpy>=2)

ADIOS2:
  - source:   $ADIOS2_SRC @ $ADIOS2_VERSION
  - build:    $ADIOS2_BUILD
  - install:  $ADIOS2_INSTALL
  - libdir:   ${ADIOS2_LIBDIR}
  - python:   linked from $ADIOS2_PY_SITEPKG into $VENV_SITEPKG
  - NOTE: ADIOS2_USE_DataSpaces=OFF (Frontier-parity)

DDStore:
  - source:   ${DDSTORE_FRONTIER}/DDStore
  - python:   installed into venv (pip)

To activate later:
  module reset
  module load frameworks
  source ${VENV_PATH}/bin/activate

Runtime library vars:
  export ADIOS2_DIR=$ADIOS2_INSTALL
  export PATH=$ADIOS2_INSTALL/bin:\$PATH
  export LD_LIBRARY_PATH=$ADIOS2_INSTALL/${ADIOS2_LIBDIR}:\$LD_LIBRARY_PATH

To keep numpy pinned for any future pip installs in this venv:
  export PIP_CONSTRAINT=$CONSTRAINTS_FILE
EOF

echo "✅ Aurora HydraGNN environment setup complete!"