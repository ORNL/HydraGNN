#!/bin/bash

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# List of URLs to download from
URLS=(
  "https://alexandria.icams.rub.de/data/pbe/2024.12.15/"
  "https://alexandria.icams.rub.de/data/pbe_2d/"
  "https://alexandria.icams.rub.de/data/pbe_1d/"
)

# Directory where files will be saved
OUTPUT_DIR="./dataset/compressed_data"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over all URLs
for URL in "${URLS[@]}"; do
  echo "Downloading from $URL"

  wget --recursive \
       --no-parent \
       --continue \
       --no-clobber \
       --convert-links \
       --cut-dirs=1 \
       --no-check-certificate \
       --reject-regex="(/older/|/geo_opt_paths/)" \
       --reject "*index.html*" \
       --directory-prefix="$OUTPUT_DIR" \
       "$URL"
done

echo "Download complete. All files saved to $OUTPUT_DIR."

