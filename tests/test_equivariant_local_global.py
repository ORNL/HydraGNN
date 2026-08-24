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

from hydragnn.globalAtt.equivariant_local_global import EquivariantLocalGlobalConv


class _EquivariantScaleConv(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, inv_node_feat, equiv_node_feat, **kwargs):
        return self.scale * inv_node_feat, self.scale * equiv_node_feat


class _RecordingGlobal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, features, positions, batch):
        self.input = features.detach().clone()
        return 3.0 * features

    def forward_attention(self, features, positions, batch):
        return self(features, positions, batch)

    def forward_feedforward(self, features):
        return features


def _layer(coupling_mode):
    return EquivariantLocalGlobalConv(
        channels=2,
        conv=_EquivariantScaleConv(),
        mpnn_type="PAINN",
        heads=1,
        lmax=1,
        num_radial=4,
        feedforward_multiplier=2,
        allow_scalar_only=False,
        require_tensor_coupling=True,
        local_equivariance=True,
        chunk_size=None,
        coupling_mode=coupling_mode,
    ).double()


@pytest.mark.parametrize(
    ("coupling_mode", "expected_scale", "expected_global_input_scale"),
    [("parallel", 5.0, 1.0), ("sequential", 6.0, 2.0)],
)
def test_equivariant_local_global_coupling_branch_isolation(
    coupling_mode, expected_scale, expected_global_input_scale
):
    layer = _layer(coupling_mode)
    recorder = _RecordingGlobal()
    layer.global_layer = recorder
    scalars = torch.randn(4, 2, dtype=torch.float64)
    vectors = torch.randn(4, 3, 2, dtype=torch.float64)
    positions = torch.randn(4, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1])

    output_scalars, output_vectors = layer(scalars, vectors, positions, batch)
    encoded_input = layer.adapter(scalars, vectors)

    torch.testing.assert_close(
        recorder.input, expected_global_input_scale * encoded_input
    )
    torch.testing.assert_close(output_scalars, expected_scale * scalars)
    torch.testing.assert_close(output_vectors, expected_scale * vectors)


@pytest.mark.parametrize("coupling_mode", ["parallel", "sequential"])
def test_equivariant_local_global_modes_preserve_se3_and_backward(coupling_mode):
    torch.manual_seed(41)
    layer = _layer(coupling_mode)
    scalars = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    vectors = torch.randn(4, 3, 2, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 0])
    rotation = o3.rand_matrix(dtype=torch.float64)
    translation = torch.randn(1, 3, dtype=torch.float64)

    output_scalars, output_vectors = layer(scalars, vectors, positions, batch)
    transformed_scalars, transformed_vectors = layer(
        scalars.detach(),
        torch.einsum("ij,njc->nic", rotation, vectors.detach()),
        positions.detach() @ rotation.T + translation,
        batch,
    )

    torch.testing.assert_close(
        transformed_scalars, output_scalars.detach(), rtol=3.0e-5, atol=5.0e-6
    )
    torch.testing.assert_close(
        transformed_vectors,
        torch.einsum("ij,njc->nic", rotation, output_vectors.detach()),
        rtol=3.0e-5,
        atol=5.0e-6,
    )
    (output_scalars.square().sum() + output_vectors.square().sum()).backward()
    assert layer.conv.scale.grad is not None
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert any(
        parameter.grad is not None for parameter in layer.global_layer.parameters()
    )


def test_equivariant_local_global_rejects_unknown_coupling_mode():
    with pytest.raises(ValueError, match="parallel.*sequential"):
        _layer("unknown")
