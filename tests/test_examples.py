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
import pdb
import subprocess
import sys

import pytest


# Test examples with GPS global attention
# Note: MACE is excluded due to e3nn tensor dimension incompatibilities with global attention
@pytest.mark.parametrize(
    "global_attn_engine",
    ["GPS"],
)
@pytest.mark.parametrize("global_attn_type", ["multihead"])
@pytest.mark.parametrize(
    "mpnn_type",
    [
        "SAGE",
        "GIN",
        "GAT",
        "MFC",
        "PNA",
        "PNAPlus",
        "CGCNN",
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
        "PAINN",
    ],
)
@pytest.mark.parametrize("example", ["qm9", "md17"])
@pytest.mark.mpi_skip()
def pytest_examples_energy_gps(
    example, mpnn_type, global_attn_engine, global_attn_type
):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    # Use sys.executable to get the current Python interpreter
    python_executable = sys.executable
    # Add the --mpnn_type and --num_epoch arguments for the subprocess call
    return_code = subprocess.call(
        [
            python_executable,
            file_path,
            "--mpnn_type",
            mpnn_type,
            "--global_attn_engine",
            global_attn_engine,
            "--global_attn_type",
            global_attn_type,
            "--num_epoch",
            "2",
        ]
    )
    assert return_code == 0


# Test examples with EquiformerV2 global attention
# Note: MACE is excluded due to e3nn tensor dimension incompatibilities with global attention
# Note: EquiformerV2 doesn't use global_attn_type parameter (it's ignored)
@pytest.mark.parametrize(
    "global_attn_engine",
    ["EquiformerV2"],
)
@pytest.mark.parametrize(
    "mpnn_type",
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
def pytest_examples_energy_equiformer(example, mpnn_type, global_attn_engine):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    # Use sys.executable to get the current Python interpreter
    python_executable = sys.executable

    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    hydragnn_root = os.path.join(os.path.dirname(__file__), "..")
    env["PYTHONPATH"] = os.path.abspath(hydragnn_root)

    # Add the --mpnn_type and --num_epoch arguments for the subprocess call
    # Note: global_attn_type is not needed for EquiformerV2 as it's ignored
    return_code = subprocess.call(
        [
            python_executable,
            file_path,
            "--mpnn_type",
            mpnn_type,
            "--global_attn_engine",
            global_attn_engine,
            "--num_epoch",
            "2",
        ],
        env=env,
    )
    assert return_code == 0


# NOTE the grad forces example with LennardJones requires
#      there to be a positional gradient via using
#      positions in torch operations for message-passing.
@pytest.mark.parametrize(
    "mpnn_type",
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
def pytest_examples_grad_forces(example, mpnn_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")

    # Add the --mpnn_type and --num_epoch arguments for the subprocess call
    return_code = subprocess.call([sys.executable, file_path, "--mpnn_type", mpnn_type, "--num_epoch", "2"])

    # Check the file ran without error.
    assert return_code == 0
