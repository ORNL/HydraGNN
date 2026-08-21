##############################################################################
# Copyright (c) 2022, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os, json
import pytest

import hydragnn


def test_input_config_parsing_is_exposed_through_utils():
    assert hydragnn.utils.input_config_parsing.update_config is not None


@pytest.mark.parametrize("config_file", ["lsms/lsms.json"])
@pytest.mark.mpi_skip()
def test_config(config_file):

    config_file = os.path.join("examples", config_file)
    with open(config_file, "r") as f:
        config = json.load(f)

    expected = {
        "Dataset": [
            "name",
            "path",
            "format",
            "num_nodes",
        ],
        "NeuralNetwork": ["Architecture", "Training"],
        "Variables": ["inputs", "outputs"],
    }

    for category in expected.keys():
        assert category in config, "Missing required input category"

        for input in category:
            assert input in category, "Missing required input"
