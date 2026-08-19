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
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GPSConv as PyGGPSConv

from hydragnn.globalAtt.gps import (
    HydraGPSConv,
    redraw_performer_projections,
)


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


def _inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(17)
    x = torch.randn(7, 8)
    equiv = torch.randn(7, 3, 4)
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
def pytest_graphgps_global_branch_matches_pyg(attn_type):
    torch.manual_seed(23)
    native = PyGGPSConv(8, conv=None, heads=2, attn_type=attn_type).eval()
    ours = HydraGPSConv(8, conv=None, heads=2, attn_type=attn_type).eval()
    _copy_nonlocal_state(ours, native)
    x, equiv, edge_index, batch = _inputs()

    with torch.no_grad():
        expected = native(x, edge_index, batch)
        actual, actual_equiv = ours(x, equiv, batch, edge_index=edge_index)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert actual_equiv is equiv


def pytest_graphgps_local_and_global_branches_match_pyg():
    torch.manual_seed(29)
    native_local = GCNConv(8, 8)
    native = PyGGPSConv(8, conv=native_local, heads=2).eval()
    ours = HydraGPSConv(
        8,
        conv=_HydraLocalAdapter(copy.deepcopy(native_local)),
        heads=2,
    ).eval()
    _copy_nonlocal_state(ours, native)
    x, equiv, edge_index, batch = _inputs()

    with torch.no_grad():
        expected = native(x, edge_index, batch)
        actual, actual_equiv = ours(x, equiv, batch, edge_index=edge_index)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual_equiv, equiv)


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
