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
"""Numerical parity tests for HydraGNN's PyG-derived GraphGPS layer."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GPSConv as PyGGPSConv

from hydragnn.globalAtt.gps import (
    HydraGPSConv,
    redraw_performer_projections,
)
from hydragnn.models.create import create_model


class _HydraLocalAdapter(torch.nn.Module):
    """Expose a standard PyG convolution through HydraGNN's two-stream API."""

    def __init__(self, conv: torch.nn.Module):
        super().__init__()
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(
        self,
        inv_node_feat: Tensor,
        equiv_node_feat: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        return self.conv(inv_node_feat, edge_index, **kwargs), equiv_node_feat


def _inputs(batch: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(17)
    x = torch.randn(7, 8)
    equiv = torch.randn(7, 3, 4)
    if batch is None:
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 4],
            [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 4, 6],
        ]
    )
    return x, equiv, edge_index, batch


def _copy_nonlocal_state(ours: HydraGPSConv, native: PyGGPSConv) -> None:
    ours.attn.load_state_dict(copy.deepcopy(native.attn.state_dict()))
    ours.mlp.load_state_dict(copy.deepcopy(native.mlp.state_dict()))
    for ours_norm, native_norm in (
        (ours.norm1, native.norm1),
        (ours.norm2, native.norm2),
        (ours.norm3, native.norm3),
    ):
        if ours_norm is not None:
            ours_norm.load_state_dict(copy.deepcopy(native_norm.state_dict()))


@pytest.mark.parametrize("attn_type", ["multihead", "performer"])
@pytest.mark.parametrize(
    "batch",
    [
        torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        torch.tensor([0, 1, 1, 1, 1, 1, 1]),
        torch.zeros(7, dtype=torch.long),
    ],
    ids=["balanced", "uneven", "single-graph"],
)
def pytest_graphgps_global_branch_matches_pyg(attn_type, batch):
    torch.manual_seed(23)
    native = PyGGPSConv(8, conv=None, heads=2, attn_type=attn_type).eval()
    ours = HydraGPSConv(8, conv=None, heads=2, attn_type=attn_type).eval()
    _copy_nonlocal_state(ours, native)
    x, equiv, edge_index, batch = _inputs(batch)

    with torch.no_grad():
        expected = native(x, edge_index, batch)
        actual, actual_equiv = ours(x, equiv, batch, edge_index=edge_index)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert actual_equiv is equiv


@pytest.mark.parametrize("attn_type", ["multihead", "performer"])
def pytest_graphgps_local_and_global_branches_match_pyg(attn_type):
    torch.manual_seed(29)
    native_local = GCNConv(8, 8)
    native = PyGGPSConv(8, conv=native_local, heads=2, attn_type=attn_type).eval()
    ours = HydraGPSConv(
        8,
        conv=_HydraLocalAdapter(copy.deepcopy(native_local)),
        heads=2,
        attn_type=attn_type,
    ).eval()
    _copy_nonlocal_state(ours, native)
    x, equiv, edge_index, batch = _inputs()

    with torch.no_grad():
        expected = native(x, edge_index, batch)
        actual, actual_equiv = ours(x, equiv, batch, edge_index=edge_index)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual_equiv, equiv)


@pytest.mark.parametrize("attn_type", ["multihead", "performer"])
def pytest_graphgps_training_dropout_matches_pyg(attn_type):
    torch.manual_seed(31)
    native = PyGGPSConv(
        8, conv=None, heads=2, dropout=0.25, attn_type=attn_type
    ).train()
    ours = HydraGPSConv(
        8, conv=None, heads=2, dropout=0.25, attn_type=attn_type
    ).train()
    _copy_nonlocal_state(ours, native)
    x, equiv, edge_index, batch = _inputs()

    torch.manual_seed(37)
    expected = native(x, edge_index, batch)
    torch.manual_seed(37)
    actual, actual_equiv = ours(x, equiv, batch, edge_index=edge_index)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert actual_equiv is equiv


@pytest.mark.parametrize("attn_type", ["multihead", "performer"])
def pytest_graphgps_input_gradients_match_pyg(attn_type):
    torch.manual_seed(41)
    native = PyGGPSConv(8, conv=None, heads=2, attn_type=attn_type).eval()
    ours = HydraGPSConv(8, conv=None, heads=2, attn_type=attn_type).eval()
    _copy_nonlocal_state(ours, native)
    x, equiv, edge_index, batch = _inputs()
    native_x = x.detach().clone().requires_grad_()
    ours_x = x.detach().clone().requires_grad_()

    native(native_x, edge_index, batch).square().sum().backward()
    ours(ours_x, equiv, batch, edge_index=edge_index)[0].square().sum().backward()

    assert torch.allclose(ours_x.grad, native_x.grad, atol=1e-6, rtol=1e-5)
    native_gradients = dict(native.named_parameters())
    for name, parameter in ours.named_parameters():
        native_gradient = native_gradients[name].grad
        if native_gradient is None:
            assert parameter.grad is None
        else:
            assert torch.allclose(parameter.grad, native_gradient, atol=1e-6, rtol=1e-5)


def pytest_graphgps_performer_model_factory_forward_and_backward():
    model = create_model(
        mpnn_type="GIN",
        input_dim=2,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=2,
        global_attn_engine="GPS",
        global_attn_type="performer",
        global_attn_heads=2,
        output_type=["graph"],
        output_heads={
            "graph": [
                {
                    "type": "branch-0",
                    "architecture": {
                        "num_sharedlayers": 1,
                        "dim_sharedlayers": 4,
                        "num_headlayers": 1,
                        "dim_headlayers": [4],
                    },
                }
            ]
        },
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=1,
        use_gpu=False,
    )
    data = Data(
        x=torch.randn(5, 2),
        pe=torch.randn(5, 2),
        pos=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4], [1, 0, 2, 1, 0, 2, 4, 3]]),
        batch=torch.tensor([0, 0, 0, 1, 1]),
    )

    output = model(data)
    assert len(output) == 1
    assert output[0].shape == (2, 1)
    output[0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert isinstance(model.graph_convs[0], HydraGPSConv)


def pytest_graphgps_performer_projection_redraw_is_step_based(monkeypatch):
    layer = HydraGPSConv(8, conv=None, heads=2, attn_type="performer").train()
    redraw_count = 0

    def count_redraw():
        nonlocal redraw_count
        redraw_count += 1

    monkeypatch.setattr(layer.attn, "redraw_projection_matrix", count_redraw)

    assert redraw_performer_projections(layer, redraw_interval=2) == 0
    assert redraw_count == 0
    assert redraw_performer_projections(layer, redraw_interval=2) == 1
    assert redraw_count == 1
    assert redraw_performer_projections(layer, redraw_interval=2) == 0

    layer.eval()
    assert redraw_performer_projections(layer, redraw_interval=1) == 0
    assert redraw_count == 1


def pytest_graphgps_performer_projection_redraw_handles_multiple_layers(monkeypatch):
    layers = torch.nn.ModuleList(
        [
            HydraGPSConv(8, conv=None, heads=2, attn_type="performer"),
            HydraGPSConv(8, conv=None, heads=2, attn_type="multihead"),
            HydraGPSConv(8, conv=None, heads=2, attn_type="performer"),
        ]
    ).train()
    redraw_counts = [0, 0]

    for index, layer in enumerate((layers[0], layers[2])):

        def count_redraw(index=index):
            redraw_counts[index] += 1

        monkeypatch.setattr(layer.attn, "redraw_projection_matrix", count_redraw)

    assert redraw_performer_projections(layers, redraw_interval=None) == 0
    assert redraw_performer_projections(layers, redraw_interval=1) == 2
    assert redraw_counts == [1, 1]


def pytest_graphgps_performer_projection_redraw_rejects_invalid_interval():
    layer = HydraGPSConv(8, conv=None, heads=2, attn_type="performer").train()

    with pytest.raises(ValueError, match="positive or None"):
        redraw_performer_projections(layer, redraw_interval=0)
