#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD}"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train.tar.gz \
    "${output_dir}/train.tar.gz"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz \
    "${output_dir}/val.tar.gz"
