#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD}"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://figshare.com/ndownloader/files/36035789 \
    "${output_dir}/transition1x-release.h5"
