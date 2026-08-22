#!/bin/bash

# URL to download the zip file from
URL="https://figshare.com/ndownloader/articles/6815699/versions/10"

# Directory where the file will be saved
OUTPUT_DIR="dataset/JARVIS-DFT"

mkdir -p "$OUTPUT_DIR"

# Use curl to follow redirects and download the file
curl -L -o "$OUTPUT_DIR/6815699.zip" "$URL"
