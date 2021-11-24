##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
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


@pytest.mark.parametrize("config_file", ["lsms.json"])
@pytest.mark.mpi_skip()
def pytest_config(config_file):

    config_file = os.path.join("examples/lsms", config_file)
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    expected = {
        "Dataset": [
            "name",
            "path",
            "format",
            "num_nodes",
            "node_features",
            "graph_features",
        ],
        "NeuralNetwork": ["Architecture", "Variables_of_interest", "Training"],
    }

    for category in expected.keys():
        assert category in config, "Missing required input category"

        for input in category:
            assert input in category, "Missing required input"
