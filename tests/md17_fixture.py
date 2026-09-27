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

from pathlib import Path

import numpy as np


def create_md17_raw_fixture(root: Path, num_samples: int = 32) -> Path:
    raw_dir = root / "uracil" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "md17_uracil.npz"

    rng = np.random.default_rng(0)
    atomic_numbers = np.array([6, 6, 7, 7, 8, 8, 1, 1], dtype=np.int64)
    positions = rng.normal(size=(num_samples, atomic_numbers.size, 3)).astype(
        np.float32
    )
    energies = np.square(positions).sum(axis=(1, 2), keepdims=True).astype(np.float32)
    forces = (-2.0 * positions).astype(np.float32)
    np.savez(raw_path, z=atomic_numbers, R=positions, E=energies, F=forces)
    return raw_path
