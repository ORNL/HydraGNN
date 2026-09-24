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
import sys
import pytest
import pdb
import subprocess

from tests.md17_fixture import create_md17_raw_fixture


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
        "SchNet",
        "DimeNet",
        "EGNN",
        "PNAEq",
        "PAINN",
    ],
)
@pytest.mark.parametrize("example", ["qm9", "md17"])
@pytest.mark.mpi_skip()
def test_examples_energy(
    tmp_path, example, mpnn_type, global_attn_engine, global_attn_type
):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    # Add the --mpnn_type argument for the subprocess call
    env = os.environ.copy()
    if example == "md17":
        cache_root = tmp_path / "md17"
        create_md17_raw_fixture(cache_root)
        env["HYDRAGNN_MD17_ROOT"] = str(cache_root)

    return_code = subprocess.call(
        [
            sys.executable,
            file_path,
            "--mpnn_type",
            mpnn_type,
            "--global_attn_engine",
            global_attn_engine,
            "--global_attn_type",
            global_attn_type,
        ],
        env=env,
    )

    # Check the file ran without error.
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
def test_examples_grad_forces(example, mpnn_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")

    # Add the --mpnn_type argument for the subprocess call
    return_code = subprocess.call([sys.executable, file_path, "--mpnn_type", mpnn_type])

    # Check the file ran without error.
    assert return_code == 0
