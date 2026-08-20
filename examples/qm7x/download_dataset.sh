#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
output_dir="${1:-$PWD/dataset/QM7-X}"
mkdir -p "${output_dir}"
python "${repo_root}/hydragnn/utils/datasets/download.py" \
    https://zenodo.org/api/records/3905361/files-archive \
    "${output_dir}/3905361.zip"
cd "${output_dir}"
unzip -n 3905361.zip

for file in *.xz; do
    [ -e "$file" ] || continue  # Skip if no .xz files exist
    tar xvf "$file"
done
