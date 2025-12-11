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
import json
import tempfile
import shutil
import sys

import subprocess


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type", ["SchNet", "EGNN", "DimeNet", "PAINN", "PNAPlus", "MACE"]
)
@pytest.mark.mpi_skip()
def pytest_examples(example, mpnn_type):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")  # Assuming different model scripts
    return_code = subprocess.call(["python", file_path, "--mpnn_type", mpnn_type])

    # Check the file ran without error.
    assert return_code == 0


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type,head_type",
    [
        ("SchNet", "conv"),
        ("EGNN", "conv"),
        ("DimeNet", "conv"),
        ("PAINN", "conv"),
        ("SchNet", "rotation_invariant_mlp"),
        ("EGNN", "rotation_invariant_mlp"),
        ("PAINN", "rotation_invariant_mlp"),
        ("MACE", "rotation_invariant_mlp"),
    ],
)
@pytest.mark.mpi_skip()
def pytest_equivariant_heads(example, mpnn_type, head_type):
    """Test equivariant models with different head types (conv vs rotation_invariant_mlp)."""
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    config_path = os.path.join(path, "LJ.json")

    # Load base config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modify config to use the specified head type
    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["type"] = head_type

    # Set equivariance=True for rotation_invariant_mlp head types
    if head_type in ["rotation_invariant_mlp", "rotation_invariant_mlp_per_node"]:
        config["NeuralNetwork"]["Architecture"]["equivariance"] = True

    # Create temporary config file
    temp_dir = tempfile.mkdtemp()
    try:
        temp_config_path = os.path.join(temp_dir, "temp_config.json")
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        return_code = subprocess.call(
            [
                sys.executable,
                file_path,
                "--mpnn_type",
                mpnn_type,
                "--inputfile",
                temp_config_path,
            ]
        )

        # Check the file ran without error
        assert return_code == 0
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type,head_type",
    [
        ("SchNet", "mlp"),
        ("EGNN", "mlp"),
    ],
)
@pytest.mark.mpi_skip()
def pytest_expanded_x_features(example, mpnn_type, head_type):
    """Test that MLP heads work correctly with expanded x features (including equiv_norm)."""
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    config_path = os.path.join(path, "LJ.json")

    # Load base config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modify config to use MLP head type
    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["type"] = head_type

    # Create temporary config file
    temp_dir = tempfile.mkdtemp()
    try:
        temp_config_path = os.path.join(temp_dir, "temp_config.json")
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        return_code = subprocess.call(
            [
                sys.executable,
                file_path,
                "--mpnn_type",
                mpnn_type,
                "--inputfile",
                temp_config_path,
            ]
        )

        # Check the file ran without error
        assert return_code == 0
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
