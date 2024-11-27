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
import pytest

import subprocess


@pytest.mark.parametrize(
    "model_type",
    [
        "SAGE",
        "GIN",
        "GAT",
        "MFC",
        "PNA",
        "PNAPlus",
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
        "PAINN",
    ],
)
@pytest.mark.parametrize("example", ["qm9", "md17"])
@pytest.mark.mpi_skip()
def pytest_examples_energy(example, model_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")

    # Add the --model_type argument for the subprocess call
    return_code = subprocess.call(["python", file_path, "--model_type", model_type])

    # Check the file ran without error.
    assert return_code == 0


# NOTE the grad forces example with LennardJones requires
#      there to be a positional gradient via using
#      positions in torch operations for message-passing.
@pytest.mark.parametrize(
    "model_type",
    [
        "PNAPlus",
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
        "PAINN",
        "MACE",
    ],
)
@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.mpi_skip()
def pytest_examples_grad_forces(example, model_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")

    # Add the --model_type argument for the subprocess call
    return_code = subprocess.call(["python", file_path, "--model_type", model_type])

    # Check the file ran without error.
    assert return_code == 0
