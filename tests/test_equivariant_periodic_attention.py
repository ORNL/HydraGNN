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

from hydragnn.globalAtt.equivariant_transformer import EquivariantTransformerLayer
from hydragnn.globalAtt.periodic_supercell import (
    build_periodic_attention_sources,
)
from hydragnn.utils.input_config_parsing.config_utils import (
    validate_equivariant_transformer_config,
)


def _periodic_inputs(dtype=torch.double):
    features = torch.tensor(
        [[0.5, -0.2, 0.1, 0.3], [-0.4, 0.7, -0.5, 0.2]], dtype=dtype
    )
    positions = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.6, 0.4]], dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    pbc = torch.ones((1, 3), dtype=torch.bool)
    return features, positions, batch, cell, pbc


def test_periodic_sources_are_explicit_and_not_minimum_image_wrapped():
    features, positions, batch, cell, pbc = _periodic_inputs()
    sources = build_periodic_attention_sources(
        features, positions, batch, cell, pbc, replication=1
    )

    assert sources.features.shape == (2 * 27, 4)
    atom_one_images = sources.positions[sources.base_index == 1]
    assert any(
        torch.allclose(value, torch.tensor([-0.1, 0.6, 0.4]).double())
        for value in atom_one_images
    )
    assert any(
        torch.allclose(value, torch.tensor([0.9, 0.6, 0.4]).double())
        for value in atom_one_images
    )
    assert any(
        torch.allclose(value, torch.tensor([1.9, 0.6, 0.4]).double())
        for value in atom_one_images
    )


def test_periodic_attention_preserves_rotation_equivariance_and_cell_gradients():
    torch.manual_seed(19)
    irreps = o3.Irreps("1x0e + 1x1o")
    layer = EquivariantTransformerLayer(
        irreps, heads=2, lmax=1, num_radial=4, chunk_size=1
    ).double()
    features, positions, batch, cell, pbc = _periodic_inputs()
    positions.requires_grad_()
    cell.requires_grad_()

    output = layer.forward_feedforward(
        layer.forward_periodic_attention(
            features, positions, batch, cell, pbc, replication=1
        )
    )
    rotation = o3.rand_matrix(dtype=torch.double)
    representation = irreps.D_from_matrix(rotation)
    rotated_output = layer.forward_feedforward(
        layer.forward_periodic_attention(
            features @ representation.T,
            positions.detach() @ rotation.T,
            batch,
            cell.detach() @ rotation.T,
            pbc,
            replication=1,
        )
    )
    torch.testing.assert_close(
        rotated_output, output.detach() @ representation.T, atol=2e-6, rtol=2e-6
    )

    output.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert cell.grad is not None and torch.isfinite(cell.grad).all()
    assert cell.grad.abs().sum() > 0


def test_periodic_attention_returns_only_central_queries_and_extent_is_configurable():
    torch.manual_seed(23)
    layer = EquivariantTransformerLayer(
        "4x0e",
        heads=2,
        lmax=1,
        num_radial=4,
        require_tensor_coupling=False,
        chunk_size=1,
    ).double()
    features, positions, batch, cell, pbc = _periodic_inputs()

    extent_one = layer.forward_periodic_attention(
        features, positions, batch, cell, pbc, replication=1
    )
    extent_two = layer.forward_periodic_attention(
        features, positions, batch, cell, pbc, replication=[2, 1, 1]
    )

    assert extent_one.shape == features.shape
    assert extent_two.shape == features.shape
    assert not torch.allclose(extent_one, extent_two)


def test_periodic_attention_keeps_batched_graphs_independent():
    torch.manual_seed(29)
    layer = EquivariantTransformerLayer(
        "2x0e",
        heads=1,
        require_tensor_coupling=False,
        chunk_size=1,
    ).double()
    features = torch.tensor([[0.2, 0.7], [-0.4, 0.3]], dtype=torch.double)
    positions = torch.tensor([[0.1, 0.0, 0.0], [0.6, 0.2, 0.0]], dtype=torch.double)
    batch = torch.tensor([0, 1], dtype=torch.long)
    cells = torch.stack((torch.eye(3), 2.0 * torch.eye(3))).double()
    pbc = torch.tensor([[True, False, False], [True, False, False]])

    together = layer.forward_periodic_attention(
        features, positions, batch, cells, pbc, replication=1
    )
    separate = torch.cat(
        [
            layer.forward_periodic_attention(
                features[index : index + 1],
                positions[index : index + 1],
                torch.zeros(1, dtype=torch.long),
                cells[index : index + 1],
                pbc[index : index + 1],
                replication=1,
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(together, separate)


def test_periodic_configuration_validates_replication_convention():
    config = {
        "global_attn_engine": "EquivariantTransformer",
        "mpnn_type": "PAINN",
        "global_attn_heads": 2,
        "equivariant_attn_lmax": 1,
        "equivariant_attn_num_radial": 8,
        "equivariant_attn_feedforward_multiplier": 2,
        "equivariant_attn_allow_scalar_only": False,
        "equivariant_attn_require_tensor_coupling": True,
        "equivariant_attn_periodic": True,
        "equivariant_attn_periodic_replication": [1, 1, 1],
    }
    validate_equivariant_transformer_config(config)

    config["equivariant_attn_periodic_replication"] = [1, -1, 1]
    with pytest.raises(ValueError, match="replication"):
        validate_equivariant_transformer_config(config)
