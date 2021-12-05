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
from utils.lsms.convert_total_energy_to_formation_gibbs import (
    compute_formation_enthalpy,
)


def unittest_formation_enthalpy():

    path_to_dir = "dataset/unit_test_enthalpy/"
    tests.linear_mixing_internal_energy_data(path_to_dir)

    check_sum = 0

    for filename in os.listdir(path_to_dir):

        _, _, _, enthalpy = compute_formation_enthalpy(
            path_to_dir + filename, [0, 1], [0, 1]
        )

        check_sum = check_sum + enthalpy

    assert 0 == check_sum


@pytest.mark.mpi_skip()
def pytest_formation_enthalpy():
    unittest_formation_enthalpy()
