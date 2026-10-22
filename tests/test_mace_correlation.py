##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory
# All rights reserved.
#
# This file is part of HydraGNN and is distributed under a BSD 3-clause
# license. For the licensing terms see the LICENSE file in the top-level
# directory.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

import pytest

from hydragnn.models.create import create_model
from hydragnn.models.backends.mace_utils.modules.blocks import (
    EquivariantProductBasisBlock,
)
from hydragnn.utils.model.model import update_multibranch_heads


def _create_mace(correlation):
    output_heads = {
        "node": {
            "num_headlayers": 3,
            "dim_headlayers": [8, 4, 1],
            "type": "mlp",
        }
    }

    return create_model(
        mpnn_type="MACE",
        input_dim=1,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads=update_multibranch_heads(output_heads),
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=3,
        num_nodes=4,
        num_radial=4,
        radial_type="bessel",
        distance_transform="None",
        envelope_exponent=5,
        radius=3.0,
        equivariance=True,
        correlation=correlation,
        max_ell=1,
        node_max_ell=1,
        avg_num_neighbors=4,
        use_gpu=False,
    )


def test_mace_uses_layer_specific_correlation():
    expected = [1, 2, 1]
    model = _create_mace(expected)

    product_blocks = [
        module
        for module in model.modules()
        if isinstance(module, EquivariantProductBasisBlock)
    ]
    actual = [
        block.symmetric_contractions.contractions[0].correlation
        for block in product_blocks
    ]

    assert actual == expected


def test_mace_rejects_invalid_correlation_length():
    with pytest.raises(ValueError, match="one value or one value per interaction"):
        _create_mace([1, 2])
