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

import pytest
import torch
from torch_geometric.data import HeteroData

from hydragnn.models.create import create_model
from hydragnn.utils.model.model import update_multibranch_heads


def _build_simple_hetero_graph(input_dim: int = 4, edge_dim: int = None):
    data = HeteroData()

    # Node features
    data["a"].x = torch.randn(4, input_dim)
    data["b"].x = torch.randn(3, input_dim)

    # Edges: a -> b
    edge_index_ab = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]], dtype=torch.long)
    data[("a", "to", "b")].edge_index = edge_index_ab
    if edge_dim is not None:
        data[("a", "to", "b")].edge_attr = torch.randn(edge_index_ab.size(1), edge_dim)

    # Edges: b -> a
    edge_index_ba = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data[("b", "to", "a")].edge_index = edge_index_ba
    if edge_dim is not None:
        data[("b", "to", "a")].edge_attr = torch.randn(edge_index_ba.size(1), edge_dim)

    return data


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "mpnn_type,edge_dim,pna_deg",
    [
        ("HeteroGIN", None, None),
        ("HeteroSAGE", None, None),
        ("HeteroGAT", 3, None),
        ("HeteroPNA", 3, [1, 2, 3, 2]),
    ],
)
def pytest_hetero_graph_head_forward(mpnn_type, edge_dim, pna_deg):
    data = _build_simple_hetero_graph(edge_dim=edge_dim)

    output_heads = {
        "graph": {
            "num_sharedlayers": 1,
            "dim_sharedlayers": 16,
            "num_headlayers": 1,
            "dim_headlayers": [8],
        }
    }

    config_args = {
        "mpnn_type": mpnn_type,
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": [2],
        "pe_dim": 1,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["graph"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": False,
        "use_graph_attr_conditioning": False,
        "graph_pooling": "mean",
        "hetero_pooling_mode": "sum",
    }
    if edge_dim is not None:
        config_args["edge_dim"] = edge_dim
    if pna_deg is not None:
        config_args["pna_deg"] = pna_deg

    model = create_model(**config_args)
    model.eval()

    outputs = model(data)
    assert isinstance(outputs, list)
    assert outputs[0].shape == (1, 2)


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "mpnn_type,edge_dim,pna_deg",
    [
        ("HeteroGIN", None, None),
        ("HeteroSAGE", None, None),
        ("HeteroGAT", 3, None),
        ("HeteroPNA", 3, [1, 2, 3, 2]),
    ],
)
def pytest_hetero_node_conv_head_forward(mpnn_type, edge_dim, pna_deg):
    data = _build_simple_hetero_graph(edge_dim=edge_dim)

    output_heads = {
        "node": {
            "num_headlayers": 2,
            "dim_headlayers": [16, 8],
            "type": "conv",
        }
    }

    config_args = {
        "mpnn_type": mpnn_type,
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": [1],
        "pe_dim": 1,
        "global_attn_engine": "",
        "global_attn_type": "",
        "global_attn_heads": 1,
        "output_type": ["node"],
        "output_heads": update_multibranch_heads(output_heads),
        "activation_function": "relu",
        "loss_function_type": "mse",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "equivariance": False,
        "use_graph_attr_conditioning": False,
        "graph_pooling": "mean",
        "hetero_pooling_mode": "sum",
        "node_target_type": "a",
    }
    if edge_dim is not None:
        config_args["edge_dim"] = edge_dim
    if pna_deg is not None:
        config_args["pna_deg"] = pna_deg

    model = create_model(**config_args)
    model.eval()

    outputs = model(data)
    assert isinstance(outputs, list)
    assert outputs[0].shape == (data["a"].num_nodes, 1)
