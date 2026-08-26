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


def _create_model(mpnn_type, **overrides):
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
    model_options = {}
    if mpnn_type == "SchNet":
        model_options.update(
            num_gaussians=4, num_filters=4, max_neighbours=8, edge_dim=1
        )
    else:
        model_options.update(
            basis_emb_size=4,
            envelope_exponent=5,
            int_emb_size=4,
            out_emb_size=8,
            num_after_skip=1,
            num_before_skip=1,
            num_spherical=2,
            edge_dim=None,
        )
    options = dict(
        mpnn_type=mpnn_type,
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
        num_radial=4,
        radius=3.0,
        equivariance=False,
        equivariant_attn_allow_scalar_only=True,
        equivariant_attn_require_tensor_coupling=False,
        use_gpu=False,
    )
    options.update(model_options)
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
        edge_attr=torch.zeros(edge_index.shape[1], 1),
        batch=torch.zeros(3, dtype=torch.long),
    )


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
def test_scalar_mpnn_equivariant_transformer_forward_backward_and_se3(mpnn_type):
    torch.manual_seed(51)
    model = _create_model(mpnn_type)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.1, 0.8, 0.3]],
        requires_grad=True,
    )
    rotation = o3.rand_matrix(dtype=positions.dtype)
    translation = torch.tensor([[0.6, -0.2, 1.3]])

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


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
@pytest.mark.parametrize("coupling_mode", ["parallel", "sequential"])
def test_scalar_mpnn_equivariant_transformer_coupling_modes(mpnn_type, coupling_mode):
    model = _create_model(mpnn_type, equivariant_attn_coupling_mode=coupling_mode)
    output = model(_data(torch.randn(3, 3)))[0]

    assert output.shape == (1, 1)
    assert all(conv.coupling_mode == coupling_mode for conv in model.graph_convs)


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
def test_scalar_mpnn_equivariant_transformer_state_dict_round_trip(mpnn_type):
    model = _create_model(mpnn_type).eval()
    data = _data(torch.randn(3, 3))
    expected = model(data)[0]

    restored = _create_model(mpnn_type).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)

    torch.testing.assert_close(restored(data)[0], expected)


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
def test_scalar_mpnn_equivariant_transformer_requires_periodic_opt_in(mpnn_type):
    model = _create_model(mpnn_type)
    shifts = torch.zeros(6, 3)
    shifts[0, 2] = 2.0

    with pytest.raises(ValueError, match="equivariant_attn_periodic=true"):
        model(_data(torch.randn(3, 3), edge_shifts=shifts))


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
def test_scalar_mpnn_periodic_full_model_forward_backward(mpnn_type):
    model = _create_model(
        mpnn_type,
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


@pytest.mark.parametrize("mpnn_type", ["SchNet", "DimeNet"])
def test_scalar_mpnn_config_requires_both_safeguards(mpnn_type):
    config = {
        "global_attn_engine": "EquivariantTransformer",
        "mpnn_type": mpnn_type,
        "global_attn_heads": 2,
        "equivariant_attn_lmax": 1,
        "equivariant_attn_num_radial": 8,
        "equivariant_attn_feedforward_multiplier": 2,
        "equivariant_attn_allow_scalar_only": True,
        "equivariant_attn_require_tensor_coupling": True,
        "equivariance": False,
    }
    with pytest.raises(ValueError, match="cannot provide tensor-valued"):
        validate_equivariant_transformer_config(config)

    config["equivariant_attn_require_tensor_coupling"] = False
    config["equivariant_attn_allow_scalar_only"] = False
    with pytest.raises(ValueError, match="allow_scalar_only"):
        validate_equivariant_transformer_config(config)


def test_schnet_config_rejects_coordinate_updates():
    config = {
        "global_attn_engine": "EquivariantTransformer",
        "mpnn_type": "SchNet",
        "global_attn_heads": 2,
        "equivariant_attn_lmax": 1,
        "equivariant_attn_num_radial": 8,
        "equivariant_attn_feedforward_multiplier": 2,
        "equivariant_attn_allow_scalar_only": True,
        "equivariant_attn_require_tensor_coupling": False,
        "equivariance": True,
    }
    with pytest.raises(ValueError, match="coordinate updates"):
        validate_equivariant_transformer_config(config)
