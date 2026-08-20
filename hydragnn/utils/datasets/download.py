##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

"""Reusable, resumable dataset download and safe archive extraction."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
from urllib.request import Request, urlopen


def _sha256(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: str | os.PathLike,
    *,
    sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download ``url`` atomically, resuming a retained partial file.

    Completed files are reused. When ``sha256`` is supplied, both reused and
    newly downloaded files must match it. Data is first written to a sibling
    ``.part`` file and atomically renamed only after successful completion.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        if sha256 is None or _sha256(destination, chunk_size) == sha256.lower():
            return destination
        raise ValueError(f"SHA-256 mismatch for existing file: {destination}")
    if destination.exists():
        raise ValueError(
            f"Download destination exists and is not a file: {destination}"
        )

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urlopen(Request(url, headers=headers)) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=chunk_size)

    if sha256 is not None and _sha256(partial, chunk_size) != sha256.lower():
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch for downloaded file: {destination}")
    partial.replace(destination)
    return destination


def safe_extract_tar(
    archive: str | os.PathLike, destination: str | os.PathLike
) -> Path:
    """Extract a tar archive while rejecting links and path traversal."""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported archive entry: {member.name}")
            tar.extract(member, path=destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--sha256")
    parser.add_argument("--extract-to")
    parser.add_argument("--remove-archive", action="store_true")
    args = parser.parse_args()

    archive = download_file(args.url, args.destination, sha256=args.sha256)
    if args.extract_to:
        safe_extract_tar(archive, args.extract_to)
        if args.remove_archive:
            archive.unlink()


if __name__ == "__main__":
    main()
