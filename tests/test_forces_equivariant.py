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
import pytest

import subprocess


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "model_type", ["SchNet", "EGNN", "DimeNet", "PAINN", "PNAPlus", "MACE"]
)
@pytest.mark.mpi_skip()
def pytest_examples(example, model_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")  # Assuming different model scripts
    return_code = subprocess.call(["python", file_path, "--model_type", model_type])

    # Check the file ran without error.
    assert return_code == 0
