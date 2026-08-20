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
from e3nn import o3
from torch_geometric.data import Data

from hydragnn.globalAtt.equivariant_local_global import EquivariantLocalGlobalConv
from hydragnn.models.create import create_model
from hydragnn.utils.input_config_parsing.config_utils import (
    validate_equivariant_transformer_config,
)
from hydragnn.utils.model import update_multibranch_heads


def _create_model():
    heads = update_multibranch_heads(
        {
            "graph": {
                "num_sharedlayers": 1,
                "dim_sharedlayers": 4,
                "num_headlayers": 1,
                "dim_headlayers": [4],
            }
        }
    )
    return create_model(
        mpnn_type="MACE",
        input_dim=1,
        hidden_dim=2,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="EquivariantTransformer",
        global_attn_type=None,
        global_attn_heads=1,
        output_type=["graph"],
        output_heads=heads,
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        edge_dim=None,
        num_radial=3,
        radial_type="bessel",
        distance_transform=None,
        radius=3.0,
        equivariance=True,
        correlation=2,
        max_ell=1,
        node_max_ell=1,
        avg_num_neighbors=2.0,
        envelope_exponent=5,
        equivariant_attn_chunk_size=2,
        use_gpu=False,
    )


def _data(positions, edge_shifts=None):
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    if edge_shifts is None:
        edge_shifts = torch.zeros(edge_index.shape[1], 3)
    return Data(
        x=torch.tensor([[1.0], [6.0], [8.0]]),
        pos=positions,
        edge_index=edge_index,
        edge_shifts=edge_shifts,
        batch=torch.zeros(3, dtype=torch.long),
    )


def pytest_mace_equivariant_transformer_forward_backward_and_se3():
    torch.manual_seed(61)
    model = _create_model()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.1, 0.8, 0.3]],
        requires_grad=True,
    )
    rotation = o3.rand_matrix(dtype=positions.dtype)
    translation = torch.tensor([[-0.4, 0.9, 1.2]])

    output = model(_data(positions))[0]
    transformed_output = model(_data(positions.detach() @ rotation.T + translation))[0]

    torch.testing.assert_close(
        transformed_output, output.detach(), rtol=5.0e-5, atol=8.0e-6
    )
    output.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert isinstance(model.graph_convs[0], EquivariantLocalGlobalConv)
    assert not isinstance(model.graph_convs[-1], EquivariantLocalGlobalConv)
    assert any(
        parameter.grad is not None
        for parameter in model.graph_convs[0].global_layer.parameters()
    )


def pytest_mace_equivariant_transformer_rejects_periodic_images():
    model = _create_model()
    shifts = torch.zeros(6, 3)
    shifts[0, 0] = 3.0

    with pytest.raises(ValueError, match="periodic images"):
        model(_data(torch.randn(3, 3), edge_shifts=shifts))


def pytest_mace_config_requires_a_tensor_hidden_layer():
    config = {
        "global_attn_engine": "EquivariantTransformer",
        "mpnn_type": "MACE",
        "global_attn_heads": 1,
        "num_conv_layers": 1,
        "equivariant_attn_lmax": 1,
        "equivariant_attn_num_radial": 8,
        "equivariant_attn_feedforward_multiplier": 2,
        "equivariant_attn_allow_scalar_only": False,
        "equivariant_attn_require_tensor_coupling": True,
    }
    with pytest.raises(ValueError, match="at least two convolution"):
        validate_equivariant_transformer_config(config)
