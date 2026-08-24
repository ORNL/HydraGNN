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
        mpnn_type="PAINN",
        input_dim=1,
        hidden_dim=4,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="EquivariantTransformer",
        global_attn_type=None,
        global_attn_heads=2,
        output_type=["graph"],
        output_heads=heads,
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        edge_dim=None,
        num_radial=4,
        radius=3.0,
        equivariance=True,
        use_gpu=False,
    )


def _data(positions, edge_shifts=None):
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    if edge_shifts is None:
        edge_shifts = torch.zeros(edge_index.shape[1], 3)
    return Data(
        x=torch.tensor([[0.2], [0.7], [-0.4]]),
        pos=positions,
        edge_index=edge_index,
        edge_shifts=edge_shifts,
        batch=torch.zeros(3, dtype=torch.long),
    )


def test_painn_equivariant_transformer_forward_backward_and_rotation():
    torch.manual_seed(31)
    model = _create_model()
    assert all(conv.coupling_mode == "parallel" for conv in model.graph_convs)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.1, 0.8, 0.3]],
        requires_grad=True,
    )
    rotation = o3.rand_matrix(dtype=positions.dtype)
    translation = torch.tensor([[0.4, -1.2, 0.7]])

    output = model(_data(positions))[0]
    transformed_output = model(_data(positions.detach() @ rotation.T + translation))[0]

    torch.testing.assert_close(
        transformed_output, output.detach(), rtol=2.0e-5, atol=3.0e-6
    )
    output.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert any(
        parameter.grad is not None
        for convolution in model.graph_convs
        for parameter in convolution.global_layer.parameters()
    )


def test_painn_equivariant_transformer_rejects_periodic_images():
    model = _create_model()
    positions = torch.randn(3, 3)
    shifts = torch.zeros(6, 3)
    shifts[0, 0] = 2.0

    with pytest.raises(ValueError, match="periodic images"):
        model(_data(positions, edge_shifts=shifts))


def test_equivariant_transformer_config_rejects_untested_model_integration():
    config = {
        "global_attn_engine": "EquivariantTransformer",
        "mpnn_type": "EGNN",
        "global_attn_heads": 2,
        "equivariant_attn_lmax": 1,
        "equivariant_attn_num_radial": 8,
        "equivariant_attn_feedforward_multiplier": 2,
        "equivariant_attn_require_tensor_coupling": True,
    }

    with pytest.raises(ValueError, match="currently supports PAINN, PNAEq"):
        validate_equivariant_transformer_config(config)


def test_equivariant_transformer_config_rejects_unknown_coupling_mode():
    config = {
        "global_attn_engine": "EquivariantTransformer",
        "mpnn_type": "PAINN",
        "global_attn_heads": 2,
        "equivariant_attn_lmax": 1,
        "equivariant_attn_num_radial": 8,
        "equivariant_attn_feedforward_multiplier": 2,
        "equivariant_attn_require_tensor_coupling": True,
        "equivariant_attn_coupling_mode": "unknown",
    }

    with pytest.raises(ValueError, match="parallel.*sequential"):
        validate_equivariant_transformer_config(config)
