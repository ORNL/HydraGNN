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

import sys, os, json
import pytest

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Distance, NormalizeRotation
from hydragnn.preprocess.utils import (
    get_radius_graph_config,
)

from hydragnn.preprocess.utils import (
    check_data_samples_equivalence,
)

torch.set_default_tensor_type(torch.DoubleTensor)


def unittest_rotational_invariance():
    config_file = "./tests/inputs/ci_rotational_invariance.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)
    compute_edges = get_radius_graph_config(config["Architecture"], loop=False)
    compute_edge_lengths = Distance(norm=False, cat=True)
    rotation = NormalizeRotation(max_points=-1, sort=False)

    # Create BCT lattice structure with 32 nodes
    data = Data()

    data.pos = torch.tensor(
        [
            [0.000000000000000, 0.000000000000000, 0.000000000000000],
            [2.609000000000000, 2.609000000000000, 3.529000000000000],
            [5.218000000000000, 0.000000000000000, 0.000000000000000],
            [7.827000000000000, 2.609000000000000, 3.529000000000000],
            [0.000000000000000, 5.218000000000000, 0.000000000000000],
            [2.609000000000000, 7.827000000000000, 3.529000000000000],
            [5.218000000000000, 5.218000000000000, 0.000000000000000],
            [7.827000000000000, 7.827000000000000, 3.529000000000000],
            [0.000000000000000, 0.000000000000000, 7.058000000000000],
            [2.609000000000000, 2.609000000000000, 10.587000000000000],
            [5.218000000000000, 0.000000000000000, 7.058000000000000],
            [7.827000000000000, 2.609000000000000, 10.587000000000000],
            [0.000000000000000, 5.218000000000000, 7.058000000000000],
            [2.609000000000000, 7.827000000000000, 10.587000000000000],
            [5.218000000000000, 5.218000000000000, 7.058000000000000],
            [7.827000000000000, 7.827000000000000, 10.587000000000000],
            [10.436000000000000, 0.000000000000000, 0.000000000000000],
            [13.045000000000000, 2.609000000000000, 3.529000000000000],
            [15.654000000000000, 0.000000000000000, 0.000000000000000],
            [18.262999999999998, 2.609000000000000, 3.529000000000000],
            [10.436000000000000, 5.218000000000000, 0.000000000000000],
            [13.045000000000000, 7.827000000000000, 3.529000000000000],
            [15.654000000000000, 5.218000000000000, 0.000000000000000],
            [18.262999999999998, 7.827000000000000, 3.529000000000000],
            [10.436000000000000, 0.000000000000000, 7.058000000000000],
            [13.045000000000000, 2.609000000000000, 10.587000000000000],
            [15.654000000000000, 0.000000000000000, 7.058000000000000],
            [18.262999999999998, 2.609000000000000, 10.587000000000000],
            [10.436000000000000, 5.218000000000000, 7.058000000000000],
            [13.045000000000000, 7.827000000000000, 10.587000000000000],
            [15.654000000000000, 5.218000000000000, 7.058000000000000],
            [18.262999999999998, 7.827000000000000, 10.587000000000000],
        ]
    )

    # Create random nodal features
    data.x = torch.randn(32, 1)

    # Create arbitrary global feature
    data.y = torch.tensor([[99]])

    # Create a copy of the same data sample that is going to be rotated
    data_rotated = data.clone()

    data = compute_edges(data)
    data = compute_edge_lengths(data)

    # Rotate clone data
    data_rotated = rotation(data_rotated)

    data_rotated = compute_edges(data_rotated)
    data_rotated = compute_edge_lengths(data_rotated)

    assert check_data_samples_equivalence(data, data_rotated, 1e-14)

    # Repeat the same check on 10 graphs randomly generated
    for num_random_graphs in range(10):
        data = Data()

        # The position of the nodes in the graph is random
        data.pos = 3 * torch.randn(10, 3)

        # Create random nodal features
        data.x = torch.randn_like(data.pos)

        # Create arbitrary global feature
        data.y = torch.randn(1, 1)

        # Create a copy of the same data sample that is going to be rotated
        data_rotated = data.clone()

        data = compute_edges(data)
        data = compute_edge_lengths(data)

        # Rotate clone data
        data_rotated = rotation(data_rotated)

        data_rotated = compute_edges(data_rotated)
        data_rotated = compute_edge_lengths(data_rotated)

        assert check_data_samples_equivalence(data, data_rotated, 1e-14)


def pytest_train_model():
    unittest_rotational_invariance()
