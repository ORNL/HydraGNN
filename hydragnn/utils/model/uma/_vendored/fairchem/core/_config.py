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
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import shutil

CACHE_DIR = os.environ.get(
    "FAIRCHEM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache/fairchem")
)
os.makedirs(CACHE_DIR, exist_ok=True)


def clear_cache():
    try:
        shutil.rmtree(CACHE_DIR)
    except FileNotFoundError:
        print(f"No FAIRChem cache directory found at {CACHE_DIR}")
