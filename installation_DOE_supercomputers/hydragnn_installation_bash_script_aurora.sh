#!/usr/bin/env bash
# setup_env_aurora.sh
# Automated setup for HydraGNN env + PyTorch(XPU) + PyG(base) on ALCF Aurora.

set -Eeuo pipefail

# =========================
# Pretty printing helpers
# =========================
hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

banner "Starting HydraGNN environment setup on ALCF Aurora ($(date))"

# ============================================================
# Module initialization
# ============================================================
banner "Configure Aurora Modules"
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # Lmod init (common)
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    source /usr/share/lmod/lmod/init/bash
  fi
fi

if ! command -v module >/dev/null 2>&1; then
  echo "❌ 'module' command not found. Are you on an Aurora login/compute node?"
  exit 1
fi

module reset

# Aurora stack choices (based on your module avail list)
module load oneapi/release/2025.2.0
module load miniforge3/24.3.0-0
module load cmake/3.31.8
module load ninja/1.12.1
module load gcc/13.3.0
module load git-lfs/3.5.1

# Optional (uncomment if you want)
# module unload darshan-runtime/3.4.7

# ============================================================
# Installation root
# ============================================================
banner "Set Base Installation Directory"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/HydraGNN-Installation-Aurora}"
mkdir -p "$INSTALL_ROOT"
echo "All installation components will be contained in: $INSTALL_ROOT"
cd "$INSTALL_ROOT"

# ============================================================
# Conda activation
# ============================================================
banner "Initialize Conda"
# miniforge3 module should provide conda, but ensure shell integration:
if [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
elif [[ -f "${CONDA_ROOT:-}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  # fallback: try conda hook
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ Conda command not found. Ensure miniforge3/24.3.0-0 is loaded."
  exit 1
fi

# ============================================================
# Create and activate env
# ============================================================
banner "Create and Activate Conda Environment"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
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

# shellcheck disable=SC1091
conda activate "$VENV_PATH"

echo "Python in use: $(which python)"
python --version

# ============================================================
# pip helpers + pins
# ============================================================
banner "pip Helpers and Core Pins"
PIP_FLAGS=(--upgrade-strategy only-if-needed)

pip_retry() {
  local tries=3 delay=3
  for ((i=1; i<=tries; i++)); do
    if python -m pip install "${PIP_FLAGS[@]}" "$@"; then
      return 0
    fi
    echo "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

subbanner "Upgrade pip/setuptools/wheel"
pip_retry --disable-pip-version-check -U pip setuptools wheel

# NumPy pin: keep consistent with a lot of HPC python stacks
NUMPY_VER="${NUMPY_VER:-1.26.4}"
subbanner "Install numpy==${NUMPY_VER}"
pip_retry "numpy==${NUMPY_VER}"

python - <<PY
import numpy as np
print("numpy =", np.__version__)
PY

# ============================================================
# Core scientific Python dependencies (customize as needed)
# ============================================================
banner "Install Core Python Packages"
pip_retry ninja
pip_retry pyyaml requests tqdm filelock
pip_retry "psutil==7.1.0" "sympy==1.14.0" "scipy==1.14.1"
pip_retry pytest build Cython
pip_retry "tensorboard==2.20.0" "scikit-learn==1.5.1"
pip_retry ase "h5py==3.14.0" lmdb

# ============================================================
# PyTorch XPU (Intel GPU)
# ============================================================
banner "Install PyTorch with XPU support"
# PyTorch upstream provides XPU wheels via the nightly/xpu index URL (commonly used on Intel GPU setups). :contentReference[oaicite:1]{index=1}
TORCH_XPU_INDEX_URL="${TORCH_XPU_INDEX_URL:-https://download.pytorch.org/whl/nightly/xpu}"
echo "Installing torch/torchvision/torchaudio from: ${TORCH_XPU_INDEX_URL}"
pip_retry --index-url "${TORCH_XPU_INDEX_URL}" torch torchvision torchaudio

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("xpu available =", hasattr(torch, "xpu") and torch.xpu.is_available())
if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("xpu device count =", torch.xpu.device_count())
    x = torch.randn(8, device="xpu")
    print("xpu tensor ok:", x.device, x.dtype, x.shape)
PY

# ============================================================
# PyTorch-Geometric (BASE ONLY)
# ============================================================
banner "Install PyTorch-Geometric (base torch_geometric only)"
# PyG base library can be installed without optional deps and can utilize Intel GPUs via xpu. :contentReference[oaicite:2]{index=2}
pip_retry torch_geometric

python - <<'PY'
import torch
import torch_geometric
print("torch_geometric =", torch_geometric.__version__)
print("torch =", torch.__version__)
print("xpu available =", hasattr(torch, "xpu") and torch.xpu.is_available())
PY

# ============================================================
# Optional: install PyG optional deps (NOT recommended on XPU by default)
# ============================================================
banner "Optional PyG deps (disabled by default)"
INSTALL_PYG_OPTIONAL="${INSTALL_PYG_OPTIONAL:-0}"
cat <<EOF
INFO:
  On Aurora/XPU, PyG optional deps (torch-sparse/torch-cluster/pyg-lib/...) may not have XPU wheels and often expect CUDA/CPU builds.
  The base torch_geometric install above is the safe default.

To try anyway (at your own risk), re-run with:
  INSTALL_PYG_OPTIONAL=1 ./setup_env_aurora.sh

EOF

if [[ "$INSTALL_PYG_OPTIONAL" -eq 1 ]]; then
  subbanner "Attempting optional deps build/install (CPU/CUDA-oriented; may fail on XPU)"
  # If you try, do it explicitly and expect to debug compilers/ABI:
  pip_retry pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv || true
fi

# ============================================================
# HydraGNN itself (editable install)
# ============================================================
banner "Install HydraGNN (editable)"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_BRANCH="${HYDRAGNN_BRANCH:-main}"
HYDRAGNN_SRC="${HYDRAGNN_SRC:-${INSTALL_ROOT}/HydraGNN}"

if [[ ! -d "${HYDRAGNN_SRC}/.git" ]]; then
  git clone --recursive -b "${HYDRAGNN_BRANCH}" "${HYDRAGNN_REPO}" "${HYDRAGNN_SRC}"
fi

pushd "${HYDRAGNN_SRC}" >/dev/null
pip_retry -e . --verbose
popd >/dev/null

# ============================================================
# Final Summary
# ============================================================
banner "Final Summary"
cat <<EOF
Base install:        $INSTALL_ROOT
Virtual environment: $VENV_PATH

Modules to load later:
  module reset
  module load oneapi/release/2025.2.0
  module load miniforge3/24.3.0-0
  module load cmake/3.31.8
  module load ninja/1.12.1
  module load gcc/13.3.0
  module load git-lfs/3.5.1

Activate:
  conda activate $VENV_PATH

Notes:
  - torch_geometric installed (base only).
  - Optional PyG deps are disabled by default on XPU.

EOF

echo "✅ HydraGNN-Aurora environment setup complete!"

