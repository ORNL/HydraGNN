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
"""Version PyG processed data without duplicating downloaded raw datasets."""

from pathlib import Path
import shutil

CACHE_VERSION_FILE = ".hydragnn-cache-version"


def _migrate_legacy_raw(cache_root, legacy_cache_directories):
    root = Path(cache_root)
    raw = root / "raw"
    for directory in legacy_cache_directories:
        legacy = root / directory
        if not legacy.is_dir():
            continue
        legacy_raw = legacy / "raw"
        if legacy_raw.is_dir():
            raw.mkdir(parents=True, exist_ok=True)
            for source in legacy_raw.iterdir():
                destination = raw / source.name
                if not destination.exists():
                    shutil.move(str(source), str(destination))
        shutil.rmtree(legacy)


def prepare_pyg_cache(cache_root, expected_version, legacy_cache_directories=()):
    """Retain one raw download and remove an incompatible processed cache.

    Legacy version directories are collapsed into ``cache_root``. Their raw
    files are moved into the stable ``raw`` directory when not already there;
    versioned processed artifacts are discarded because PyG can rebuild them.
    """
    root = Path(cache_root)
    _migrate_legacy_raw(root, legacy_cache_directories)

    processed = root / "processed"
    marker = processed / CACHE_VERSION_FILE
    current_version = (
        marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    )
    if processed.is_dir() and current_version != expected_version:
        shutil.rmtree(processed)


def mark_pyg_cache_current(cache_root, version):
    """Mark a successfully built PyG processed cache with its exact format."""
    marker = Path(cache_root) / "processed" / CACHE_VERSION_FILE
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(version + "\n", encoding="utf-8")
