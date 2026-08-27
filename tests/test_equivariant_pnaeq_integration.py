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
        mpnn_type="PNAEq",
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
        pna_deg=[1.0, 3.0, 2.0],
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


def test_pnaeq_equivariant_transformer_forward_backward_and_se3():
    torch.manual_seed(41)
    model = _create_model()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.1, 0.8, 0.3]],
        requires_grad=True,
    )
    rotation = o3.rand_matrix(dtype=positions.dtype)
    translation = torch.tensor([[-0.3, 0.5, 1.1]])

    output = model(_data(positions))[0]
    transformed_output = model(_data(positions.detach() @ rotation.T + translation))[0]

    torch.testing.assert_close(
        transformed_output, output.detach(), rtol=3.0e-5, atol=5.0e-6
    )
    output.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert any(
        parameter.grad is not None
        for convolution in model.graph_convs
        for parameter in convolution.global_layer.parameters()
    )


@pytest.mark.parametrize("coupling_mode", ["parallel", "sequential"])
def test_pnaeq_equivariant_transformer_coupling_modes_and_all_layers(
    coupling_mode,
):
    model = _create_model(equivariant_attn_coupling_mode=coupling_mode)
    positions = torch.randn(3, 3, requires_grad=True)

    model(_data(positions))[0].sum().backward()

    assert len(model.graph_convs) == 2
    assert all(conv.coupling_mode == coupling_mode for conv in model.graph_convs)
    assert all(
        any(parameter.grad is not None for parameter in conv.global_layer.parameters())
        for conv in model.graph_convs
    )


def test_pnaeq_equivariant_transformer_state_dict_round_trip():
    model = _create_model().eval()
    data = _data(torch.randn(3, 3))
    expected = model(data)[0]

    restored = _create_model().eval()
    with torch.no_grad():
        for parameter in restored.parameters():
            if parameter.is_floating_point():
                parameter.add_(1.0)
    assert any(
        not torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    )
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)

    torch.testing.assert_close(restored(data)[0], expected)


def test_pnaeq_equivariant_transformer_requires_explicit_periodic_opt_in():
    model = _create_model()
    positions = torch.randn(3, 3)
    shifts = torch.zeros(6, 3)
    shifts[0, 1] = 2.0

    with pytest.raises(ValueError, match="equivariant_attn_periodic=true"):
        model(_data(positions, edge_shifts=shifts))


def test_pnaeq_equivariant_transformer_periodic_full_model_forward_backward():
    model = _create_model(
        equivariant_attn_periodic=True,
        equivariant_attn_periodic_replication=[1, 0, 0],
        equivariant_attn_chunk_size=2,
    )
    shifts = torch.zeros(6, 3)
    shifts[0, 0], shifts[1, 0] = 1.0, -1.0
    positions = torch.randn(3, 3, requires_grad=True)
    data = _data(positions, edge_shifts=shifts)
    cell = torch.eye(3, requires_grad=True)
    data.cell = cell.unsqueeze(0)
    data.pbc = torch.tensor([[True, False, False]])

    output = model(data)[0]
    output.square().sum().backward()

    assert output.shape == (1, 1)
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert cell.grad is not None and torch.isfinite(cell.grad).all()
    assert all(
        any(parameter.grad is not None for parameter in conv.global_layer.parameters())
        for conv in model.graph_convs
    )
