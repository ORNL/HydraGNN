#!/usr/bin/env bash
# setup_env_andes.sh
# Complete automated setup for HydraGNN environment and dependencies on Andes (CPU-only).

set -Eeuo pipefail

# =========================
# Pretty printing helpers
# =========================
hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

timeit() {
  SECONDS=0
  local start_time=$(date +"%T")
  echo "[$start_time] Starting: $*" >&2
  "$@"
  local status=$?
  local end_time=$(date +"%T")
  echo "[$end_time] Finished: $* (Elapsed: ${SECONDS}s, Status: $status)" >&2
  return $status
}

# Spinner function
progress() {
  local pid=$1
  local delay=0.1

  spinstr[0]="-"
  spinstr[1]="\\"
  spinstr[2]="|"
  spinstr[3]="/"

  echo -n "${spinstr[0]}"
  while kill -0 $pid 2>/dev/null; do
    for c in "${spinstr[@]}"; do
      echo -ne "\b$c"
      sleep $delay
    done
  done
  echo -ne "\b"
}

banner "Starting HydraGNN environment setup on ANDES ($(date))"

# ============================================================
# Module initialization & Andes stack
# ============================================================
banner "Configure Andes Modules"
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
  echo "⚠️  'module' command not found. Ensure you're running on Andes."
else
  module reset
  ml hsi/5.0.2.p5
  ml gcc/9.3.0
  ml openmpi/4.0.4
  ml DefApps
  ml cmake/3.22.2
  ml git-lfs/2.11.0
  ml miniforge3/23.11.0-0
  ml libfabric/1.14.0
  ml git-lfs
fi

# ============================================================
# Installation root
# ============================================================
banner "Set Base Installation Directory"
INSTALL_ROOT="${PWD}/HydraGNN-Installation-Andes"
mkdir -p "$INSTALL_ROOT"
echo "All installation components will be contained in: $INSTALL_ROOT"
cd "$INSTALL_ROOT"

# ============================================================
# Env vars & Conda env creation
# ============================================================
banner "Create and Activate Conda Environment"
# Andes is not Cray; no CRAY_LD_LIBRARY_PATH reference here
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
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

# ============================================================
# mpi4py
# ============================================================
banner "mpi4py (v3.1.5)"
MPI4PY_ANDES="${INSTALL_ROOT}/MPI4PY-Andes"
export MPI4PY_ANDES
mkdir -p "$MPI4PY_ANDES"
cd "$MPI4PY_ANDES"

git clone -b 3.1.5 https://github.com/mpi4py/mpi4py.git || true
pushd mpi4py >/dev/null
rm -rf build
CC=mpicc MPICC=mpicc pip_retry . --verbose
popd >/dev/null

# ============================================================
# PyTorch (CPU-only) + torchvision
# ============================================================
banner "Install CPU-only PyTorch"
PYTORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"
pip_retry --index-url "${PYTORCH_CPU_INDEX_URL}" torch torchvision
assert_numpy_1264

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("CUDA available?    =", torch.cuda.is_available())
PY

# ============================================================
# PyTorch-Geometric stack (CPU build)
# ============================================================
banner "PyTorch-Geometric Stack (CPU)"
PYG_DIR_NAME="PyTorch-Geometric-CPU"
PYG_ANDES="${INSTALL_ROOT}/${PYG_DIR_NAME}"
export PYG_ANDES
mkdir -p "$PYG_ANDES"
cd "$PYG_ANDES"

subbanner "pytorch_geometric (official)"
if [[ ! -d pytorch_geometric/.git ]]; then
  git clone --recursive git@github.com:pyg-team/pytorch_geometric.git
fi
pushd pytorch_geometric >/dev/null
rm -rf build
pip_retry . --verbose
assert_numpy_1264
popd >/dev/null

# --- pytorch_scatter (official repo & stable ref for CPU) ---
build_pytorch_scatter() {
  subbanner "pytorch_scatter (official @ 2.1.2-9-g7cabb53)"
  if [[ ! -d pytorch_scatter/.git ]]; then
    git clone --recursive git@github.com:rusty1s/pytorch_scatter.git
  fi
  pushd pytorch_scatter >/dev/null
  git fetch --all
  git checkout 2.1.2-9-g7cabb53
  git submodule update --init --recursive
  rm -rf build
  CC=mpicc CXX=mpicxx python setup.py build
  CC=mpicc CXX=mpicxx python setup.py install
  assert_numpy_1264
  popd >/dev/null
}

# --- pytorch_sparse (official pinned) ---
build_pytorch_sparse() {
  subbanner "pytorch_sparse (official @ 0.6.18-8-gcdbf561)"
  if [[ ! -d pytorch_sparse/.git ]]; then
    git clone --recursive git@github.com:rusty1s/pytorch_sparse.git
  fi
  pushd pytorch_sparse >/dev/null
  git fetch --all
  git checkout 0.6.18-8-gcdbf561
  rm -rf build
  CC=mpicc CXX=mpicxx python setup.py build
  CC=mpicc CXX=mpicxx python setup.py install
  assert_numpy_1264
  popd >/dev/null
}

# --- pytorch_cluster (official pinned) ---
build_pytorch_cluster() {
  subbanner "pytorch_cluster (official @ 1.6.3-11-g4126a52)"
  if [[ ! -d pytorch_cluster/.git ]]; then
    git clone --recursive git@github.com:rusty1s/pytorch_cluster.git
  fi
  pushd pytorch_cluster >/dev/null
  git fetch --all
  git checkout 1.6.3-11-g4126a52
  rm -rf build
  CC=mpicc CXX=mpicxx python setup.py build
  CC=mpicc CXX=mpicxx python setup.py install
  assert_numpy_1264
  popd >/dev/null
}

# --- pytorch_spline_conv (official pinned) ---
build_pytorch_spline_conv() {
  subbanner "pytorch_spline_conv (official @ 1.2.2-9-ga6d1020)"
  if [[ ! -d pytorch_spline_conv/.git ]]; then
    git clone --recursive git@github.com:rusty1s/pytorch_spline_conv.git
  fi
  pushd pytorch_spline_conv >/dev/null
  git fetch --all
  git checkout 1.2.2-9-ga6d1020
  rm -rf build
  CC=mpicc CXX=mpicxx python setup.py build
  CC=mpicc CXX=mpicxx python setup.py install
  assert_numpy_1264
  popd >/dev/null
}

## parallel build
banner "pytorch geometric dependencies build (in parallel)"
timeit build_pytorch_scatter > build_pytorch_scatter.log & pid1=$!
timeit build_pytorch_sparse > build_pytorch_sparse.log & pid2=$!
timeit build_pytorch_cluster > build_pytorch_cluster.log & pid3=$!
timeit build_pytorch_spline_conv > build_pytorch_spline_conv.log & pid4=$!

for pid in $pid1 $pid2 $pid3 $pid4; do
    progress $pid &
done
wait

# ============================================================
# e3nn and openequivariance
# ============================================================
banner "Install e3nn and openequivariance"
pip_retry e3nn openequivariance --verbose
assert_numpy_1264

# ============================================================
# ADIOS2
# ============================================================
ADIOS2_ANDES="${INSTALL_ROOT}/ADIOS2-Andes"
export ADIOS2_ANDES
build_adios() {
  banner "ADIOS2 (v2.10.2)"
  mkdir -p "$ADIOS2_ANDES"
  cd "$ADIOS2_ANDES"

  if [[ ! -d ADIOS2/.git ]]; then
    git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git
  fi

  mkdir -p adios2-build

  CC=mpicc CXX=mpicxx FC=mpifort \
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

  cmake --build adios2-build -j$(nproc || echo 16)
  cmake --install adios2-build
}

# ============================================================
# DDStore
# ============================================================
DDSTORE_ANDES="${INSTALL_ROOT}/DDStore-Andes"
export DDSTORE_ANDES
build_ddstore() {
  banner "DDStore"
  mkdir -p "$DDSTORE_ANDES"
  cd "$DDSTORE_ANDES"

  git clone git@github.com:ORNL/DDStore.git || true
  pushd DDStore >/dev/null
  CC=mpicc CXX=mpicxx pip_retry . --no-build-isolation --verbose
  popd >/dev/null
}

# ============================================================
# DeepHyper
# ============================================================
DEEPHYPER_ANDES="${INSTALL_ROOT}/DeepHyper-Andes"
export DEEPHYPER_ANDES
build_deephyper() {
  banner "DeepHyper (develop branch)"
  mkdir -p "$DEEPHYPER_ANDES"
  cd "$DEEPHYPER_ANDES"

  git clone https://github.com/deephyper/deephyper.git || true
  cd deephyper
  git fetch origin develop
  git checkout develop
  pip_retry -e ".[hps,hps-tl]" --verbose
  assert_numpy_1264
}

## parallel build
cd "$INSTALL_ROOT"
banner "Adios, DDStore and DeepHyper build (in parallel)"
timeit build_adios > build_adios.log & pid1=$!
timeit build_ddstore > build_ddstore.log & pid2=$!
timeit build_deephyper > build_deephyper.log & pid3=$!

for pid in $pid1 $pid2 $pid3; do
    progress $pid &
done
wait

# ============================================================
# Final Summary
# ============================================================
banner "Final Summary (Andes)"
cat <<EOF
Base install:        $INSTALL_ROOT
Virtual environment: $VENV_PATH

PyTorch (CPU-only):  from ${PYTORCH_CPU_INDEX_URL}
PyTorch-Geometric:   $PYG_ANDES
  - pytorch_scatter:     2.1.2-9-g7cabb53 (official)
  - pytorch_sparse:      0.6.18-8-gcdbf561 (official)
  - pytorch_cluster:     1.6.3-11-g4126a52 (official)
  - pytorch_spline_conv: 1.2.2-9-ga6d1020 (official)

MPI4PY:              $MPI4PY_ANDES
ADIOS2:              $ADIOS2_ANDES
DDStore:             $DDSTORE_ANDES
DeepHyper:           $DEEPHYPER_ANDES
EOF

echo "✅ HydraGNN-Installation-Andes environment setup complete!"

echo ""
echo "Use the following commands to activate the new HydraGNN python environment:"
echo "  module reset"
echo "  ml hsi/5.0.2.p5"
echo "  ml gcc/9.3.0"
echo "  ml openmpi/4.0.4"
echo "  ml DefApps"
echo "  ml cmake/3.22.2"
echo "  ml git-lfs/2.11.0"
echo "  ml miniforge3/23.11.0-0"
echo "  ml libfabric/1.14.0"
echo ""
echo "  source activate ${VENV_PATH}"


