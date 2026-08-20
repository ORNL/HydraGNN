#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset_dir="${1:-${script_dir}/dataset}"

python -m hydragnn.utils.datasets.download \
    https://ndownloader.figshare.com/files/41619375 \
    "${dataset_dir}/MPtrj_2022.9_full.json"
