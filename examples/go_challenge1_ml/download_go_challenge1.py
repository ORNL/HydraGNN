#!/usr/bin/env python3
"""Download and optionally extract Challenge 1 archives from OEDI URLs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from go_challenge1.download import extract_archive, stream_download


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--url", required=True, help="User-supplied OEDI file URL")
    parser.add_argument("--output", required=True, help="Path to downloaded archive")
    parser.add_argument("--extract", action="store_true", help="Extract after download")
    parser.add_argument("--extract-dir", default=None, help="Extraction directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("download_go_challenge1")

    output = Path(args.output)
    downloaded = stream_download(args.url, output, force=args.force, logger=logger)

    if args.extract:
        extract_dir = Path(args.extract_dir) if args.extract_dir else output.parent.parent / "extracted"
        extract_archive(downloaded, extract_dir, force=args.force, logger=logger)


if __name__ == "__main__":
    main()
