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

import os
import pytest
import json

from tests._script_runner import example_script, run_script

pytestmark = pytest.mark.examples


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type", ["SchNet", "EGNN", "DimeNet", "PAINN", "PNAPlus", "MACE", "UMA"]
)
@pytest.mark.mpi_skip()
def test_examples(example, mpnn_type):
    run_script(example_script(example), "--mpnn_type", mpnn_type)


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
        # UMA is a graph-level model; its energy = sum of L=0 (invariant)
        # node features, and forces come from autograd of that energy.
        ("UMA", "graph", None, "add"),
    ],
)
@pytest.mark.mpi_skip()
def test_equivariant_heads(
    tmp_path, example, mpnn_type, head_level, head_type, graph_pooling
):
    """Test equivariant models with different head placements (node vs graph) and pooling choices."""
    file_path = example_script(example)
    config_path = os.path.join(os.path.dirname(file_path), "LJ.json")

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
        config["Variables"]["outputs"] = [
            {"name": "energy", "level": "graph", "dim": 1}
        ]

    temp_config_path = tmp_path / "temp_config.json"
    temp_config_path.write_text(json.dumps(config, indent=2))
    run_script(
        file_path, "--mpnn_type", mpnn_type, "--inputfile", str(temp_config_path)
    )


@pytest.mark.parametrize("example", ["LennardJones"])
@pytest.mark.parametrize(
    "mpnn_type,head_type",
    [
        ("SchNet", "mlp"),
        ("EGNN", "mlp"),
    ],
)
@pytest.mark.mpi_skip()
def test_expanded_x_features(tmp_path, example, mpnn_type, head_type):
    """Test that MLP heads work correctly with expanded x features (including equiv_norm)."""
    file_path = example_script(example)
    config_path = os.path.join(os.path.dirname(file_path), "LJ.json")

    # Load base config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modify config to use MLP head type
    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["type"] = head_type

    temp_config_path = tmp_path / "temp_config.json"
    temp_config_path.write_text(json.dumps(config, indent=2))
    run_script(
        file_path, "--mpnn_type", mpnn_type, "--inputfile", str(temp_config_path)
    )
