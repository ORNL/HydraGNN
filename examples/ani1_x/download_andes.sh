#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD}"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://springernature.figshare.com/ndownloader/files/18112775 \
    "${output_dir}/ani1x-release.h5"
