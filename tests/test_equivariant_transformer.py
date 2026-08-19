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

from hydragnn.globalAtt.equivariant_transformer import (
    EquivariantRMSNorm,
    EquivariantTransformerLayer,
)


def _sample(dtype=torch.float64):
    irreps = o3.Irreps("2x0e + 2x1o")
    features = torch.randn(6, irreps.dim, dtype=dtype)
    positions = torch.randn(6, 3, dtype=dtype)
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    return irreps, features, positions, batch


def pytest_equivariant_rms_norm_is_rotation_equivariant():
    irreps, features, _, _ = _sample()
    norm = EquivariantRMSNorm(irreps).double()
    rotation = o3.rand_matrix(dtype=torch.float64)
    representation = irreps.D_from_matrix(rotation)

    transformed = norm(features @ representation.T)
    expected = norm(features) @ representation.T

    torch.testing.assert_close(transformed, expected, rtol=2.0e-5, atol=3.0e-6)
    torch.testing.assert_close(
        norm(features).square().mean(dim=-1),
        torch.ones(features.shape[0], dtype=torch.float64),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def pytest_equivariant_transformer_layer_preserves_se3_equivariance():
    torch.manual_seed(21)
    irreps, features, positions, batch = _sample()
    layer = EquivariantTransformerLayer(
        irreps, heads=2, lmax=1, feedforward_multiplier=2
    ).double()
    rotation = o3.rand_matrix(dtype=torch.float64)
    translation = torch.randn(1, 3, dtype=torch.float64)
    representation = irreps.D_from_matrix(rotation)

    output = layer(features, positions, batch)
    transformed_output = layer(
        features @ representation.T,
        positions @ rotation.T + translation,
        batch,
    )

    torch.testing.assert_close(
        transformed_output,
        output @ representation.T,
        rtol=3.0e-5,
        atol=5.0e-6,
    )


def pytest_equivariant_transformer_layer_is_permutation_equivariant():
    torch.manual_seed(22)
    irreps, features, positions, batch = _sample()
    layer = EquivariantTransformerLayer(irreps, heads=2, lmax=1).double()
    permutation = torch.tensor([3, 0, 2, 1, 5, 4])

    output = layer(features, positions, batch)
    permuted_output = layer(
        features[permutation], positions[permutation], batch[permutation]
    )

    torch.testing.assert_close(permuted_output, output[permutation])


def pytest_equivariant_transformer_layer_handles_singleton_graphs_and_backward():
    torch.manual_seed(23)
    irreps = o3.Irreps("2x0e + 2x1o")
    features = torch.randn(2, irreps.dim, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 1])
    layer = EquivariantTransformerLayer(irreps, heads=2, lmax=1).double()

    output = layer(features, positions, batch)
    output.square().sum().backward()

    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    # With no geometric pairs, the layer is correctly independent of position.
    assert positions.grad is None


@pytest.mark.parametrize(
    ("edge_index", "message"),
    [
        (torch.tensor([[0], [0]]), "self-pairs"),
        (torch.tensor([[0], [2]]), "cross-graph"),
    ],
)
def pytest_equivariant_transformer_layer_rejects_invalid_explicit_pairs(
    edge_index, message
):
    irreps = o3.Irreps("1x0e + 1x1o")
    layer = EquivariantTransformerLayer(irreps)
    features = torch.randn(3, irreps.dim)
    positions = torch.randn(3, 3)
    batch = torch.tensor([0, 0, 1])

    with pytest.raises(ValueError, match=message):
        layer(features, positions, batch, edge_index=edge_index)
