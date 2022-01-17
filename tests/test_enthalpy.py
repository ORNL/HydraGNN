##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import pandas
import numpy as np
import hydragnn, tests
import pytest
from utils.lsms import (
    convert_raw_data_energy_to_gibbs,
)


def unittest_formation_enthalpy():

    dir = "dataset/unit_test_enthalpy"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create random samples.
    num_config = 10
    tests.deterministic_graph_data(
        dir,
        num_config,
        number_types=2,
        linear_only=True,
    )

    # Create pure components.
    tests.deterministic_graph_data(
        dir,
        number_configurations=1,
        configuration_start=num_config,
        number_types=1,
        types=[0],
        linear_only=True,
    )
    tests.deterministic_graph_data(
        dir,
        number_configurations=1,
        configuration_start=num_config + 1,
        number_types=1,
        types=[1],
        linear_only=True,
    )

    convert_raw_data_energy_to_gibbs(dir, [0, 1])

    new_dir = dir + "_gibbs_energy"
    for filename in os.listdir(new_dir):
        path = os.path.join(new_dir, filename)
        enthalpy = np.loadtxt(path, max_rows=1)
        assert enthalpy == 0


@pytest.mark.mpi_skip()
def pytest_formation_enthalpy():
    unittest_formation_enthalpy()
