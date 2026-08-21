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

import pytest
import torch
from torch_geometric.data import Batch, Data

from hydragnn.utils.input_config_parsing.variable_schema import (
    get_variable_schema,
    parse_variable_schema,
    prepare_data_from_schema,
)

VARIABLES = {
    "inputs": [
        {"name": "species", "level": "node", "dim": 2},
        {"name": "positions", "level": "node", "dim": 3},
        {"name": "bonds", "level": "edge", "dim": 2},
        {"name": "state", "level": "graph", "dim": 2},
    ],
    "outputs": [
        {"name": "energy", "level": "graph", "dim": 1},
        {"name": "state_target", "level": "graph", "dim": 2},
        {"name": "forces", "level": "node", "dim": 3},
        {"name": "node_target", "level": "node", "dim": 1},
    ],
}


def _sample(num_nodes=3):
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    return Data(
        species=torch.randn(num_nodes, 2),
        positions=torch.randn(num_nodes, 3),
        edge_index=edge_index,
        bonds=torch.randn(edge_index.shape[1], 2),
        state=torch.randn(1, 2),
        energy=torch.randn(1, 1),
        state_target=torch.randn(1, 2),
        forces=torch.randn(num_nodes, 3),
        node_target=torch.randn(num_nodes, 1),
        num_nodes=num_nodes,
    )


def pytest_named_variables_compile_all_input_levels_and_outputs():
    data = _sample()
    prepare_data_from_schema(data, parse_variable_schema(VARIABLES))

    assert data.x.shape == (3, 5)
    assert data.edge_attr.shape == (4, 2)
    assert data.graph_attr.shape == (1, 2)
    assert data.graph_output.shape == (1, 3)
    assert data.node_output.shape == (3, 4)
    assert data.y.shape == (15, 1)
    assert data.y_loc.tolist() == [[0, 1, 3, 12, 15]]
    assert hasattr(data, "species") and hasattr(data, "energy")


def pytest_graph_inputs_batch_to_one_row_per_graph():
    schema = parse_variable_schema(VARIABLES)
    samples = [prepare_data_from_schema(_sample(), schema) for _ in range(2)]
    batch = Batch.from_data_list(samples)

    assert batch.graph_attr.shape == (2, 2)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("positions", torch.randn(3), "shape (3, 3)"),
        ("bonds", torch.randn(3, 2), "shape (4, 2)"),
        ("state", torch.randn(2), "shape (1, 2)"),
        ("energy", torch.randn(1), "shape (1, 1)"),
    ],
)
def pytest_named_variables_reject_wrong_shapes(attribute, value, message):
    data = _sample()
    setattr(data, attribute, value)

    with pytest.raises(
        ValueError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        prepare_data_from_schema(data, parse_variable_schema(VARIABLES))


def pytest_named_variables_require_exact_attribute_name():
    data = _sample()
    del data["forces"]

    with pytest.raises(ValueError, match="missing configured node attribute 'forces'"):
        prepare_data_from_schema(data, parse_variable_schema(VARIABLES))


def pytest_named_variable_schema_rejects_duplicate_names():
    variables = {**VARIABLES, "outputs": [VARIABLES["inputs"][0]]}
    with pytest.raises(ValueError, match="Variable names must be unique: species"):
        parse_variable_schema(variables)


def pytest_named_variable_configuration_is_required():
    with pytest.raises(ValueError, match="top-level Variables section is required"):
        get_variable_schema({"NeuralNetwork": {}})
