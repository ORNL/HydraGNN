#!/usr/bin/env bash
# hydragnn_installation_bash_script_aurora.sh
#
# Aurora official approach:
# - PyTorch: use "module load frameworks" (do NOT pip-install torch+xpu wheels)
#   https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/
# - PyG: on Aurora, install base torch_geometric in a venv that inherits system site-packages
#   https://docs.alcf.anl.gov/aurora/data-science/frameworks/pyg/#pyg-on-aurora
#
# Critical robustness patch:
# - Lmod may reference ZSH_EVAL_CONTEXT even in bash; under `set -u` that can crash.
# - So we:
#   (1) export ZSH_EVAL_CONTEXT="" early
#   (2) temporarily disable nounset (set +u) around module init AND module commands.

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
  # Ensure var exists for Lmod internals
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

echo 'Running "module reset". Resetting modules to system default.'
module reset

# Optional: uncomment only if you need extra packages from /soft
# module use /soft/modulefiles

subbanner "Load Aurora provided PyTorch (XPU) via frameworks module"
module load frameworks

subbanner "Loaded modules"
module -t list 2>&1 || true

_nounset_restore

# ============================================================
# Install root
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
# PyTorch Geometric: follow ALCF PyG-on-Aurora instructions
# ============================================================
banner "Install PyTorch Geometric base (torch_geometric)"
# Per ALCF PyG-on-Aurora: install base torch_geometric only.
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

To activate later:
  module reset
  module load frameworks
  source ${VENV_PATH}/bin/activate
EOF

echo "✅ Aurora HydraGNN environment setup complete!"
