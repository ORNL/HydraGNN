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
    python_exec = sys.executable
    return_code = subprocess.call([python_exec, file_path, "--mpnn_type", mpnn_type])

    # Check the file ran without error.
    assert return_code == 0


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type,head_level,head_type,graph_pooling",
    [
        # Node heads: conv only where supported by equivariance tests
        ("SchNet", "node", "conv", None),
        ("EGNN", "node", "conv", None),
        ("PAINN", "node", "conv", None),
        # Node heads: shared MLP for all equivariant stacks under test
        ("SchNet", "node", "mlp", None),
        ("EGNN", "node", "mlp", None),
        ("DimeNet", "node", "mlp", None),
        ("PAINN", "node", "mlp", None),
        ("MACE", "node", "mlp", None),
        # Graph-level head + sum pooling to exercise EnhancedModelWrapper graph branch
        ("SchNet", "graph", None, "add"),
        ("EGNN", "graph", None, "add"),
        ("DimeNet", "graph", None, "add"),
        ("PAINN", "graph", None, "add"),
        ("MACE", "graph", None, "add"),
    ],
)
@pytest.mark.mpi_skip()
def pytest_equivariant_heads(example, mpnn_type, head_level, head_type, graph_pooling):
    """Test equivariant models with different head placements (node vs graph) and pooling choices."""
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    config_path = os.path.join(path, "LJ.json")

    # Load base config
    with open(config_path, "r") as f:
        config = json.load(f)

    if head_level == "node":
        # Modify config to use the specified node head type
        config["NeuralNetwork"]["Architecture"]["output_heads"]["node"][
            "type"
        ] = head_type
    else:
        # Switch to graph-level head and enforce sum pooling for force loss compatibility
        config["NeuralNetwork"]["Architecture"]["graph_pooling"] = (
            graph_pooling or "add"
        )
        config["NeuralNetwork"]["Architecture"]["output_heads"] = {
            "graph": {
                "num_headlayers": 2,
                "dim_headlayers": [60, 20],
                "num_sharedlayers": 2,
                "dim_sharedlayers": 20,
            }
        }
        config["NeuralNetwork"]["Architecture"]["task_weights"] = [1]
        var_cfg = config["NeuralNetwork"]["Variables_of_interest"]
        var_cfg["type"] = ["graph"]
        var_cfg["output_dim"] = [1]
        var_cfg["output_index"] = [0]

    # Create temporary config file
    temp_dir = tempfile.mkdtemp()
    try:
        temp_config_path = os.path.join(temp_dir, "temp_config.json")
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        python_exec = sys.executable
        return_code = subprocess.call(
            [
                python_exec,
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

        python_exec = sys.executable
        return_code = subprocess.call(
            [
                python_exec,
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
