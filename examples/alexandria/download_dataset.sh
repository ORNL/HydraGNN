#!/bin/bash

# URL to download from
URL="https://alexandria.icams.rub.de/data/"

# Directory where files will be saved
OUTPUT_DIR="./dataset/compressed_data"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Use wget to recursively download all files and directories
wget --recursive \
     --no-parent \
     --continue \
     --no-clobber \
     --convert-links \
     --cut-dirs=1 \
     --no-check-certificate \
     --reject-regex="(/older/|/geo_opt_paths/)" \
     --directory-prefix="$OUTPUT_DIR" \
     "$URL"

echo "Download complete. All files saved to $OUTPUT_DIR."
