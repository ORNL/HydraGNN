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

from unittest.mock import Mock, patch

from hydragnn.architecture_defaults import MODEL_SPECIFIC_ARCHITECTURE_DEFAULTS
from hydragnn.models.create import create_model_config


def test_create_model_config_accepts_legacy_model_specific_keys():
    """Older configs need not contain options added for newer model families."""
    architecture = {
        "mpnn_type": "PNA",
        "input_dim": 1,
        "hidden_dim": 8,
        "output_dim": [1],
        "pe_dim": 0,
        "global_attn_engine": None,
        "global_attn_type": None,
        "global_attn_heads": 1,
        "output_type": ["graph"],
        "output_heads": {},
        "activation_function": "relu",
        "task_weights": [1.0],
        "num_conv_layers": 2,
        "freeze_conv_layers": False,
        "initial_bias": None,
        "num_nodes": None,
        "max_neighbours": 16,
        "edge_dim": None,
        "pna_deg": None,
        "num_before_skip": None,
        "num_after_skip": None,
        "num_radial": None,
        "radial_type": None,
        "distance_transform": None,
        "basis_emb_size": None,
        "int_emb_size": None,
        "out_emb_size": None,
        "envelope_exponent": None,
        "num_spherical": None,
        "num_gaussians": None,
        "num_filters": None,
        "radius": None,
        "equivariance": False,
        "correlation": None,
        "max_ell": None,
        "node_max_ell": None,
        "avg_num_neighbors": None,
    }
    config = {
        "Architecture": architecture,
        "Training": {
            "loss_function_type": "mse",
            "conv_checkpointing": False,
            "precision": "fp32",
        },
    }

    model = Mock()
    with patch(
        "hydragnn.models.create.create_model", return_value=model
    ) as create_model:
        result = create_model_config(config, use_gpu=False)

    assert result is model.to.return_value
    kwargs = create_model.call_args.kwargs
    expected_defaults = MODEL_SPECIFIC_ARCHITECTURE_DEFAULTS
    assert {key: kwargs[key] for key in expected_defaults} == expected_defaults
