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
import json
from pathlib import Path

from examples.ising_model.ising_preprocess import prepare_ising_dataset


def test_ising_generator_compiles_named_attributes():
    config_path = (
        Path(__file__).parents[1] / "examples" / "ising_model" / "ising_model.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = prepare_ising_dataset(2, 1, config, seed=7, sampling=0.2)

    assert dataset
    sample = dataset[0]
    assert sample.node_features.shape == (8, 1)
    assert sample.spin.shape == (8, 1)
    assert sample.total_energy.shape == (1, 1)
    assert sample.x.shape == (8, 1)
    assert sample.y_loc.shape == (1, 3)
    assert sample.edge_index.shape[0] == 2
