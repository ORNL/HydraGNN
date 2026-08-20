#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD}"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://dl.fbaipublicfiles.com/opencatalystproject/data/op26/op26_train_val_260108.tar.gz \
    "${output_dir}/op26_train_val_260108.tar.gz"
