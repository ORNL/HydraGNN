#!/usr/bin/env bash

# Shared module loader for Perlmutter installation scripts.
# Usage:
#   source ".../module_loads_perlmutter.sh"
#   load_perlmutter_modules "12.4"

# Paths are declared as env-overridable variables so callers can adjust
# platform-specific locations without modifying script logic.
MODULES_SH_PATH="${MODULES_SH_PATH:-/etc/profile.d/modules.sh}"
LMOD_INIT_BASH_PATH="${LMOD_INIT_BASH_PATH:-/usr/share/lmod/lmod/init/bash}"
MODULES_INIT_BASH_PATH="${MODULES_INIT_BASH_PATH:-/usr/share/Modules/init/bash}"
PERLMUTTER_LMOD_RESTORE_PATH="${PERLMUTTER_LMOD_RESTORE_PATH:-/opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh}"

PERLMUTTER_NERSC_DEFAULT_MODULE="${PERLMUTTER_NERSC_DEFAULT_MODULE:-nersc-default/1.0}"
PERLMUTTER_CPE_MODULE="${PERLMUTTER_CPE_MODULE:-cpe/24.07}"
PERLMUTTER_PRGENV_MODULE="${PERLMUTTER_PRGENV_MODULE:-PrgEnv-gnu/8.5.0}"
PERLMUTTER_MPICH_MODULE="${PERLMUTTER_MPICH_MODULE:-cray-mpich/8.1.30}"
PERLMUTTER_ACCEL_MODULE="${PERLMUTTER_ACCEL_MODULE:-craype-accel-nvidia80}"
PERLMUTTER_GCC_MODULE="${PERLMUTTER_GCC_MODULE:-gcc-native/13.2}"
PERLMUTTER_CMAKE_PRIMARY_MODULE="${PERLMUTTER_CMAKE_PRIMARY_MODULE:-cmake/3.30.2}"
PERLMUTTER_CMAKE_FALLBACK_MODULE="${PERLMUTTER_CMAKE_FALLBACK_MODULE:-cmake/3.24.3}"
PERLMUTTER_CONDA_PRIMARY_MODULE="${PERLMUTTER_CONDA_PRIMARY_MODULE:-conda/Miniforge3-24.11.3-0}"
PERLMUTTER_CONDA_FALLBACK_MODULE="${PERLMUTTER_CONDA_FALLBACK_MODULE:-conda/Miniforge3-24.7.1-0}"

load_perlmutter_modules() {
  local expected_cuda_mm="$1"

  if ! command -v module >/dev/null 2>&1; then
    if [[ -f "${MODULES_SH_PATH}" ]]; then
      source "${MODULES_SH_PATH}"
    elif [[ -f "${LMOD_INIT_BASH_PATH}" ]]; then
      source "${LMOD_INIT_BASH_PATH}"
    elif [[ -f "${MODULES_INIT_BASH_PATH}" ]]; then
      source "${MODULES_INIT_BASH_PATH}"
    fi
  fi

  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' command not found. Ensure you're running on Perlmutter login/compute nodes."
    return 1
  fi

  if [[ -f "${PERLMUTTER_LMOD_RESTORE_PATH}" ]]; then
    # shellcheck disable=SC1091
    source "${PERLMUTTER_LMOD_RESTORE_PATH}" || true
  fi

  module reset
  ml "${PERLMUTTER_NERSC_DEFAULT_MODULE}" || true

  ml "${PERLMUTTER_CPE_MODULE}"
  ml "${PERLMUTTER_PRGENV_MODULE}"
  ml "${PERLMUTTER_MPICH_MODULE}"
  ml "${PERLMUTTER_ACCEL_MODULE}"
  ml "cudatoolkit/${expected_cuda_mm}"
  ml "${PERLMUTTER_GCC_MODULE}"
  ml "${PERLMUTTER_CMAKE_PRIMARY_MODULE}" || ml "${PERLMUTTER_CMAKE_FALLBACK_MODULE}" || true
  ml "${PERLMUTTER_CONDA_PRIMARY_MODULE}" || ml "${PERLMUTTER_CONDA_FALLBACK_MODULE}" || true
}

print_perlmutter_activation_instructions() {
  local expected_cuda_mm="$1"
  local venv_path="$2"

  cat <<EOF
Module load + activation (for future sessions):
module reset
ml ${PERLMUTTER_NERSC_DEFAULT_MODULE} || true
ml ${PERLMUTTER_CPE_MODULE}
ml ${PERLMUTTER_PRGENV_MODULE}
ml ${PERLMUTTER_MPICH_MODULE}
ml ${PERLMUTTER_ACCEL_MODULE}
ml cudatoolkit/${expected_cuda_mm}
ml ${PERLMUTTER_GCC_MODULE}
ml ${PERLMUTTER_CONDA_PRIMARY_MODULE} || ml ${PERLMUTTER_CONDA_FALLBACK_MODULE} || true
source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || eval "\$("\$(conda info --base)/bin/conda" shell.bash hook)"
conda activate ${venv_path}
EOF
}