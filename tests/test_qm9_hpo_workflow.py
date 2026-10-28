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
from examples.qm9_hpo import qm9_deephyper_multi
from examples.qm9_hpo.workflow import (
    QM9_HPO_CACHE_VERSION,
    configure_trial,
    load_base_config,
)


def test_qm9_hpo_trial_configuration_is_isolated():
    base = load_base_config()
    trial = configure_trial(
        base,
        {
            "mpnn_type": "PNA",
            "hidden_dim": 16,
            "global_attn_heads": 4,
            "num_conv_layers": 3,
            "num_headlayers": 2,
            "dim_headlayers": 12,
        },
    )

    architecture = trial["NeuralNetwork"]["Architecture"]
    assert architecture["mpnn_type"] == "PNA"
    assert architecture["hidden_dim"] == 64
    assert architecture["num_conv_layers"] == 3
    assert architecture["output_heads"]["graph"]["dim_headlayers"] == [12, 12]
    assert base["NeuralNetwork"]["Architecture"]["mpnn_type"] == "SchNet"
    assert base["NeuralNetwork"]["Architecture"]["hidden_dim"] == 64


def test_qm9_hpo_uses_primary_subset_cache_format():
    assert QM9_HPO_CACHE_VERSION.endswith(":subset-1000")


def test_qm9_multi_parses_validation_loss_for_deephyper_maximization():
    stdout = b"training output\nValidation Loss: 1.25e-2\n"
    assert qm9_deephyper_multi._parse_results(stdout) == -0.0125
