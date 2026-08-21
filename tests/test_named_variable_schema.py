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
    schema_dimensions,
)

VARIABLES = {
    "inputs": [
        {"name": "species", "level": "node", "dim": 2},
        {"name": "pos", "level": "node", "dim": 3, "role": "position"},
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
        pos=torch.randn(num_nodes, 3),
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
    original_pos = data.pos
    prepare_data_from_schema(data, parse_variable_schema(VARIABLES))

    assert data.x.shape == (3, 2)
    assert data.pos is original_pos
    assert schema_dimensions(parse_variable_schema(VARIABLES), "node", "inputs") == 2
    assert data.edge_attr.shape == (4, 2)
    assert data.graph_attr.shape == (1, 2)
    assert data.graph_output.shape == (1, 3)
    assert data.node_output.shape == (3, 4)
    assert data.y.shape == (15, 1)
    assert data.y_loc.tolist() == [[0, 1, 3, 12, 15]]
    assert hasattr(data, "species") and hasattr(data, "energy")


def pytest_positions_do_not_change_invariant_node_features():
    schema = parse_variable_schema(VARIABLES)
    first = _sample()
    second = first.clone()
    second.pos = first.pos @ torch.linalg.qr(torch.randn(3, 3)).Q.T + 5.0

    prepare_data_from_schema(first, schema)
    prepare_data_from_schema(second, schema)

    torch.testing.assert_close(first.x, second.x)
    assert not torch.equal(first.pos, second.pos)


def pytest_graph_inputs_batch_to_one_row_per_graph():
    schema = parse_variable_schema(VARIABLES)
    samples = [prepare_data_from_schema(_sample(), schema) for _ in range(2)]
    batch = Batch.from_data_list(samples)

    assert batch.graph_attr.shape == (2, 2)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("pos", torch.randn(3), "shape (3, 3)"),
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
    variables = {**VARIABLES, "outputs": [VARIABLES["outputs"][0]] * 2}
    with pytest.raises(
        ValueError, match="Variable names within outputs must be unique: energy"
    ):
        parse_variable_schema(variables)


@pytest.mark.parametrize(
    "position",
    [
        {"name": "pos", "level": "node", "dim": 3},
        {"name": "pos", "level": "node", "dim": 3, "role": "feature"},
    ],
)
def pytest_pos_cannot_be_declared_as_an_ordinary_feature(position):
    variables = {
        "inputs": [
            {"name": "species", "level": "node", "dim": 2},
            position,
        ],
        "outputs": [],
    }
    with pytest.raises(ValueError, match="must declare role 'position'"):
        parse_variable_schema(variables)


@pytest.mark.parametrize(
    "position",
    [
        {"name": "positions", "level": "node", "dim": 3, "role": "position"},
        {"name": "pos", "level": "graph", "dim": 3, "role": "position"},
        {"name": "pos", "level": "node", "dim": 2, "role": "position"},
    ],
)
def pytest_position_role_requires_pyg_pos_shape_contract(position):
    variables = {
        "inputs": [
            {"name": "species", "level": "node", "dim": 2},
            position,
        ],
        "outputs": [],
    }
    with pytest.raises(
        ValueError, match="must have name 'pos', level 'node', and dim 3"
    ):
        parse_variable_schema(variables)


def pytest_named_variable_configuration_is_required():
    with pytest.raises(ValueError, match="top-level Variables section is required"):
        get_variable_schema({"NeuralNetwork": {}})


def pytest_internal_output_names_are_rejected():
    variables = {
        "inputs": [{"name": "features", "level": "node", "dim": 2}],
        "outputs": [{"name": "x", "level": "node", "dim": 1}],
    }
    with pytest.raises(ValueError, match="internal tensors: x"):
        parse_variable_schema(variables)


def pytest_schema_recompilation_removes_stale_derived_attributes():
    data = _sample()
    prepare_data_from_schema(data, parse_variable_schema(VARIABLES))

    node_only = {
        "inputs": [{"name": "species", "level": "node", "dim": 2}],
        "outputs": [{"name": "energy", "level": "graph", "dim": 1}],
    }
    prepare_data_from_schema(data, parse_variable_schema(node_only))

    assert "edge_attr" not in data
    assert "graph_attr" not in data
    assert "node_output" not in data
    assert "edge_output" not in data
    assert data.graph_output.shape == (1, 1)

    no_outputs = {
        "inputs": [{"name": "species", "level": "node", "dim": 2}],
        "outputs": [],
    }
    prepare_data_from_schema(data, parse_variable_schema(no_outputs))

    for name in ("node_output", "edge_output", "graph_output", "y", "y_loc"):
        assert name not in data
