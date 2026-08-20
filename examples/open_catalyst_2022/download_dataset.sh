#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD}"
archive="${output_dir}/DS_jgaid7espcoc_0.tar.xz"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://materials.colabfit.org/dataset-original/DS_jgaid7espcoc_0 \
    "${archive}"
xz -dc "${archive}" | tar -xvf - --ignore-zeros -C "${output_dir}"
