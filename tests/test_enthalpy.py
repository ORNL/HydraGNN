##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import numpy as np
import pytest
from hydragnn.utils.lsms import (
    convert_raw_data_energy_to_gibbs,
)


def _write_lsms_fixture(path, index, element_types):
    """Write the minimal LSMS-style text consumed by the conversion utility."""
    total_energy = float(sum(element_types))
    lines = [f"{total_energy}\n"]
    lines.extend(
        f"{element_type} {atom_id} 0.0 0.0 0.0\n"
        for atom_id, element_type in enumerate(element_types)
    )
    (path / f"output{index}.txt").write_text("".join(lines))


def unittest_formation_enthalpy(tmp_path):

    data_dir = tmp_path / "unit_test_enthalpy"
    data_dir.mkdir()

    # The energy is the sum of the element identifiers, so every mixture has
    # zero formation enthalpy relative to the two pure components.
    mixtures = ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1])
    for index in range(10):
        _write_lsms_fixture(data_dir, index, mixtures[index % len(mixtures)])

    # Create pure components.
    _write_lsms_fixture(data_dir, 10, [0, 0, 0, 0])
    _write_lsms_fixture(data_dir, 11, [1, 1, 1, 1])

    convert_raw_data_energy_to_gibbs(str(data_dir), [0, 1], create_plots=False)

    new_dir = str(data_dir) + "_gibbs_energy"
    for filename in os.listdir(new_dir):
        path = os.path.join(new_dir, filename)
        enthalpy = np.loadtxt(path, max_rows=1)
        assert enthalpy == 0


@pytest.mark.mpi_skip()
def pytest_formation_enthalpy(tmp_path):
    unittest_formation_enthalpy(tmp_path)
