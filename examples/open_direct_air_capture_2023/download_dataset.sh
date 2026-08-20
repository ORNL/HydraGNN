#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD}"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://dl.fbaipublicfiles.com/large_objects/dac/datasets/extxyz_train.tar.gz \
    "${output_dir}/extxyz_train.tar.gz"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://dl.fbaipublicfiles.com/dac/datasets/extxyz_val.tar.gz \
    "${output_dir}/extxyz_val.tar.gz"
