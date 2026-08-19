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

import torch
from e3nn import o3

from hydragnn.globalAtt.complete_graph import complete_graph_edge_index
from hydragnn.globalAtt.equivariant_attention import EquivariantAllToAllAttention


def _sample(dtype=torch.float64):
    irreps = o3.Irreps("2x0e + 2x1o")
    features = torch.randn(5, irreps.dim, dtype=dtype)
    positions = torch.randn(5, 3, dtype=dtype)
    batch = torch.tensor([0, 0, 0, 1, 1])
    return irreps, features, positions, batch


def pytest_equivariant_attention_is_rotation_equivariant_and_translation_invariant():
    torch.manual_seed(11)
    irreps, features, positions, batch = _sample()
    module = EquivariantAllToAllAttention(irreps, heads=2, lmax=1).double()
    edges = complete_graph_edge_index(batch)
    rotation = o3.rand_matrix(dtype=torch.float64)
    translation = torch.randn(1, 3, dtype=torch.float64)
    representation = irreps.D_from_matrix(rotation)

    output = module(features, positions, edges)
    transformed_output = module(
        features @ representation.T,
        positions @ rotation.T + translation,
        edges,
    )

    torch.testing.assert_close(
        transformed_output,
        output @ representation.T,
        rtol=2.0e-5,
        atol=3.0e-6,
    )


def pytest_equivariant_attention_is_permutation_equivariant():
    torch.manual_seed(12)
    irreps, features, positions, batch = _sample()
    module = EquivariantAllToAllAttention(irreps, heads=2, lmax=1).double()
    edges = complete_graph_edge_index(batch)
    permutation = torch.tensor([2, 0, 1, 4, 3])

    output = module(features, positions, edges)
    permuted_output = module(
        features[permutation],
        positions[permutation],
        complete_graph_edge_index(batch[permutation]),
    )

    torch.testing.assert_close(permuted_output, output[permutation])


def pytest_equivariant_attention_isolates_graphs_and_normalizes_each_target():
    torch.manual_seed(13)
    irreps, features, positions, batch = _sample()
    module = EquivariantAllToAllAttention(irreps, heads=3, lmax=1).double()
    edges = complete_graph_edge_index(batch)

    combined, attention = module(
        features, positions, edges, return_attention_weights=True
    )
    first = module(
        features[:3],
        positions[:3],
        complete_graph_edge_index(torch.zeros(3, dtype=torch.long)),
    )
    second = module(
        features[3:],
        positions[3:],
        complete_graph_edge_index(torch.zeros(2, dtype=torch.long)),
    )

    torch.testing.assert_close(combined, torch.cat((first, second)))
    for target in range(features.shape[0]):
        target_weights = attention[edges[1] == target]
        torch.testing.assert_close(
            target_weights.sum(dim=0), torch.ones(module.heads, dtype=torch.float64)
        )


def pytest_equivariant_attention_supports_backward_propagation():
    torch.manual_seed(14)
    irreps, features, positions, batch = _sample()
    features.requires_grad_()
    positions.requires_grad_()
    module = EquivariantAllToAllAttention(irreps, heads=2, lmax=1).double()

    output = module(features, positions, complete_graph_edge_index(batch))
    output.square().sum().backward()

    assert features.grad is not None
    assert positions.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(positions.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
