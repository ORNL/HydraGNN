#!/usr/bin/env bash

# Shared module loader for Frontier installation scripts.
# Usage:
#   source ".../module_loads_frontier.sh"
#   load_frontier_modules "7.1.1" "7.1.1"

# Paths are declared as env-overridable variables so callers can adjust
# platform-specific locations without modifying script logic.
MODULES_SH_PATH="${MODULES_SH_PATH:-/etc/profile.d/modules.sh}"
LMOD_INIT_BASH_PATH="${LMOD_INIT_BASH_PATH:-/usr/share/lmod/lmod/init/bash}"
MODULES_INIT_BASH_PATH="${MODULES_INIT_BASH_PATH:-/usr/share/Modules/init/bash}"

FRONTIER_CPE_MODULE="${FRONTIER_CPE_MODULE:-cpe/24.07}"
FRONTIER_CCE_MODULE="${FRONTIER_CCE_MODULE:-cce/18.0.0}"
FRONTIER_CRAYPE_ACCEL_MODULE="${FRONTIER_CRAYPE_ACCEL_MODULE:-craype-accel-amd-gfx90a}"
FRONTIER_PRGENV_MODULE="${FRONTIER_PRGENV_MODULE:-PrgEnv-gnu}"
FRONTIER_MINIFORGE_MODULE="${FRONTIER_MINIFORGE_MODULE:-miniforge3/23.11.0-0}"
FRONTIER_GITLFS_MODULE="${FRONTIER_GITLFS_MODULE:-git-lfs}"
FRONTIER_ROCM_MODULE_VERSION="${FRONTIER_ROCM_MODULE_VERSION:-7.1.1}"
FRONTIER_AMD_MIXED_MODULE_VERSION="${FRONTIER_AMD_MIXED_MODULE_VERSION:-7.1.1}"

load_frontier_modules() {
  local rocm_version="${1:-${FRONTIER_ROCM_MODULE_VERSION}}"
  local amd_mixed_version="${2:-${FRONTIER_AMD_MIXED_MODULE_VERSION}}"

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
    echo "WARNING: 'module' command not found. Ensure you're running on the target HPC system."
    return 0
  fi

  module reset
  ml "${FRONTIER_CPE_MODULE}"
  ml "${FRONTIER_CCE_MODULE}"
  ml "rocm/${rocm_version}"
  ml "amd-mixed/${amd_mixed_version}"
  ml "${FRONTIER_CRAYPE_ACCEL_MODULE}"
  ml "${FRONTIER_PRGENV_MODULE}"
  ml "${FRONTIER_MINIFORGE_MODULE}"
  ml "${FRONTIER_GITLFS_MODULE}"
  module unload darshan-runtime
}

print_frontier_activation_instructions() {
  local rocm_version="$1"
  local amd_mixed_version="$2"
  local venv_path="$3"

  cat <<EOF
Use the following commands to activate the new HydraGNN python environment:
  module reset
  ml ${FRONTIER_CPE_MODULE}
  ml ${FRONTIER_CCE_MODULE}
  ml rocm/${rocm_version}
  ml amd-mixed/${amd_mixed_version}
  ml ${FRONTIER_CRAYPE_ACCEL_MODULE}
  ml ${FRONTIER_PRGENV_MODULE}
  ml ${FRONTIER_MINIFORGE_MODULE}
  module unload darshan-runtime

  source activate ${venv_path}
EOF
}