##############################################################################
# Copyright (c) 2022, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Helper for skipping UMA tests when the optional backbone deps are absent.

The vendored FairChem UMA backbone pulls a handful of fairchem-core
third-party dependencies (omegaconf, hydra, torchtnt, ...) that are *not*
part of HydraGNN's core requirements (they live in ``requirements-optional``).
CI installs only the core requirements, so UMA tests must skip -- rather than
error -- when those dependencies cannot be imported.
"""

import functools


@functools.lru_cache(maxsize=1)
def uma_available() -> bool:
    """Return ``True`` iff the optional UMA backbone can be imported."""
    try:
        from hydragnn.models.UMAStack import _load_uma_backbones

        _load_uma_backbones()
    except Exception:
        return False
    return True
