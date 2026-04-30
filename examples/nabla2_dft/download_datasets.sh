#!/usr/bin/env bash
set -euo pipefail

# Download selected Nabla2-DFT SQLite databases as defined in energy_databases.json.
# Available keys:
#   dataset_train_full
#   dataset_train_large
#   dataset_train_medium
#   dataset_train_small
#   dataset_train_tiny
#   dataset_test_structures
#   dataset_test_scaffolds
#   dataset_test_conformations_full
#   dataset_test_conformations_large
#   dataset_test_conformations_medium
#   dataset_test_conformations_small
#   dataset_test_conformations_tiny
#   dataset_test_trajectories_initial
#   dataset_test_trajectories
#   dataset_train_medium_trajectories
#   dataset_trajectories_additional

export all_proxy="socks://proxy.ccs.ornl.gov:3128/"
export ftp_proxy="ftp://proxy.ccs.ornl.gov:3128/"
export http_proxy="http://proxy.ccs.ornl.gov:3128/"
export https_proxy="http://proxy.ccs.ornl.gov:3128/"
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
json_file="$script_dir/energy_databases.json"
target_dir="$script_dir/dataset"

usage() {
    cat <<'EOF'
Usage: download_datasets.sh [KEY ...]

KEY values come from energy_databases.json (databases.*). Examples:
  dataset_train_tiny dataset_train_small dataset_train_medium dataset_train_large dataset_train_full
  dataset_test_structures dataset_test_scaffolds dataset_test_conformations_tiny ...

If no KEY is provided, defaults to: dataset_train_tiny

Downloads are placed in examples/nabla2_dft/dataset.
If the dataset directory already exists, you will be prompted before it is overwritten.
EOF
}

list_keys() {
    python - <<'PY'
import json, pathlib
cfg = json.load(open(pathlib.Path(__file__).with_name("energy_databases.json")))
print("Available keys:")
for k in sorted(cfg["databases"].keys()):
    print(f"  {k}")
PY
}

if [[ ! -f "$json_file" ]]; then
    echo "ERROR: Cannot find $json_file" >&2
    exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    list_keys
    exit 0
fi

declare -a requested_keys=()
if [[ $# -eq 0 ]]; then
    requested_keys+=("dataset_train_tiny")
else
    # Accept space- or comma-separated lists, e.g. "dataset_train_small,dataset_test_structures"
    for arg in "$@"; do
        IFS=',' read -r -a parts <<< "$arg"
        requested_keys+=("${parts[@]}")
    done
fi

# Confirm overwrite of existing dataset directory
if [[ -d "$target_dir" ]]; then
    read -r -p "Dataset directory '$target_dir' exists. Overwrite it? (yes/no): " reply
    if [[ "$reply" != "yes" ]]; then
        echo "Aborting without changes." >&2
        exit 1
    fi
    rm -rf "$target_dir"
fi
mkdir -p "$target_dir"

download_one() {
    local key="$1"
    python - "$key" "$json_file" <<'PY'
import json, pathlib, sys
key, jf = sys.argv[1], sys.argv[2]
cfg = json.load(open(jf))
url = cfg["databases"].get(key)
etag = cfg.get("etag", {}).get(key)
if url is None:
    sys.stderr.write(f"ERROR: Unknown key '{key}'.\n")
    sys.exit(2)
print(url)
if etag:
    print(etag)
PY
}

for key in "${requested_keys[@]}"; do
    echo "--> Downloading $key"
    mapfile -t meta < <(download_one "$key")
    if [[ ${#meta[@]} -eq 0 ]]; then
        echo "Skipping $key (missing URL)." >&2
        continue
    fi
    url="${meta[0]}"
    etag="${meta[1]:-}"
    outfile="$target_dir/${key}.db"

    echo "    URL: $url"
    [[ -n "$etag" ]] && echo "    ETag: $etag"

    curl -L --fail --retry 3 --retry-delay 2 --progress-bar -o "$outfile" "$url"
    echo "    Saved to $outfile ($(stat -f%z "$outfile") bytes)"
done

echo "All requested downloads finished."