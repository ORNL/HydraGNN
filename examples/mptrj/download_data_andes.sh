#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
dataset_dir="${1:-${script_dir}/dataset}"

python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://ndownloader.figshare.com/files/41619375 \
    "${dataset_dir}/MPtrj_2022.9_full.json"
