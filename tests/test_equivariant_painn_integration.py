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

import copy

import pytest
import torch
from e3nn import o3
from torch_geometric.data import Data

from hydragnn.models.create import create_model
from hydragnn.utils.input_config_parsing.config_utils import (
    validate_equivariant_transformer_config,
)
from hydragnn.utils.model import update_multibranch_heads


def _create_model(**overrides):
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
    options = dict(
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
    options.update(overrides)
    return create_model(**options)


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


@pytest.mark.parametrize("coupling_mode", ["parallel", "sequential"])
def test_painn_equivariant_transformer_coupling_modes_use_all_layers(
    coupling_mode,
):
    torch.manual_seed(32)
    model = _create_model(equivariant_attn_coupling_mode=coupling_mode)
    positions = torch.randn(3, 3, requires_grad=True)

    output = model(_data(positions))[0]
    output.sum().backward()

    assert len(model.graph_convs) == 2
    assert all(conv.coupling_mode == coupling_mode for conv in model.graph_convs)
    assert all(
        any(parameter.grad is not None for parameter in conv.global_layer.parameters())
        for conv in model.graph_convs
    )
    assert positions.grad is not None and torch.isfinite(positions.grad).all()


def test_painn_equivariant_transformer_chunked_full_model_matches_dense():
    torch.manual_seed(33)
    dense = _create_model(equivariant_attn_chunk_size=None)
    chunked = copy.deepcopy(dense)
    for convolution in chunked.graph_convs:
        convolution.global_layer.chunk_size = 2

    dense_positions = torch.randn(3, 3, requires_grad=True)
    chunked_positions = dense_positions.detach().clone().requires_grad_()
    dense_output = dense(_data(dense_positions))[0]
    chunked_output = chunked(_data(chunked_positions))[0]
    dense_output.sum().backward()
    chunked_output.sum().backward()

    torch.testing.assert_close(chunked_output, dense_output)
    torch.testing.assert_close(chunked_positions.grad, dense_positions.grad)
    for dense_parameter, chunked_parameter in zip(
        dense.parameters(), chunked.parameters()
    ):
        if dense_parameter.grad is None:
            assert chunked_parameter.grad is None
        else:
            torch.testing.assert_close(
                chunked_parameter.grad, dense_parameter.grad, rtol=2.0e-5, atol=2.0e-6
            )


def test_painn_equivariant_transformer_state_dict_round_trip():
    torch.manual_seed(34)
    model = _create_model().eval()
    data = _data(torch.randn(3, 3))
    expected = model(data)[0]

    restored = _create_model().eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    actual = restored(data)[0]

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("conditioning_mode", ["film", "concat_node", "fuse_pool"])
def test_painn_equivariant_transformer_supports_graph_conditioning(
    conditioning_mode,
):
    torch.manual_seed(35)
    model = _create_model(
        use_graph_attr_conditioning=True,
        graph_attr_conditioning_mode=conditioning_mode,
    )
    positions = torch.randn(3, 3, requires_grad=True)
    data = _data(positions)
    data.graph_attr = torch.tensor([[0.3, -0.2]])

    output = model(data)[0]
    output.square().sum().backward()

    assert output.shape == (1, 1)
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    conditioner_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "graph_conditioner" in name
        or "graph_concat_projector" in name
        or "graph_pool_projector" in name
    ]
    assert conditioner_parameters
    assert all(parameter.grad is not None for parameter in conditioner_parameters)


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
