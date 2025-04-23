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

import json
import pytest

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Distance, NormalizeRotation
from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_config

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    check_data_samples_equivalence,
)


def create_bct_sample():
    # Create BCT lattice structure with 32 nodes
    uc_x = 4
    uc_y = 2
    uc_z = 2
    lxy = 5.218
    lz = 7.058
    count = 0
    number_nodes = 2 * uc_x * uc_y * uc_z
    positions = torch.zeros(number_nodes, 3)
    for x in range(uc_x):
        for y in range(uc_y):
            for z in range(uc_z):
                positions[count][0] = x * lxy
                positions[count][1] = y * lxy
                positions[count][2] = z * lz
                count += 1
                positions[count][0] = (x + 0.5) * lxy
                positions[count][1] = (y + 0.5) * lxy
                positions[count][2] = (z + 0.5) * lz
                count += 1

    data = Data()
    data.pos = positions
    return data


def check_rotational_invariance(
    data, compute_edges, compute_edge_lengths, compute_rotation, tolerance
):
    # Create a copy of the same data sample that is going to be rotated
    data_rotated = data.clone()

    data = compute_edges(data)
    data = compute_edge_lengths(data)

    # Rotate clone data
    data_rotated = compute_rotation(data_rotated)

    data_rotated = compute_edges(data_rotated)
    data_rotated = compute_edge_lengths(data_rotated)

    assert check_data_samples_equivalence(data, data_rotated, tolerance)


def unittest_rotational_invariance(tol=1e-14):
    config_file = "./tests/inputs/ci_rotational_invariance.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    compute_edges = get_radius_graph_config(config["Architecture"], loop=False)
    compute_edge_lengths = Distance(norm=False, cat=True)
    compute_rotation = NormalizeRotation(max_points=-1, sort=False)

    # Run first with single BCT structure.
    data = create_bct_sample()

    # Create random nodal features
    data.x = torch.randn(32, 1)

    # Create arbitrary global feature
    data.y = torch.tensor([[99]])

    check_rotational_invariance(
        data, compute_edges, compute_edge_lengths, compute_rotation, tol
    )

    # Repeat the same check on 10 graphs randomly generated
    for num_random_graphs in range(10):
        data = Data()

        # The position of the nodes in the graph is random
        data.pos = 3 * torch.randn(10, 3)

        # Create random nodal features
        data.x = torch.randn_like(data.pos)

        # Create arbitrary global feature
        data.y = torch.randn(1, 1)

        check_rotational_invariance(
            data, compute_edges, compute_edge_lengths, compute_rotation, tol
        )


@pytest.mark.mpi_skip()
def pytest_rotational_invariance():
    # Test with (default) single precision
    unittest_rotational_invariance(tol=1e-4)

    # Test with double precision
    torch.set_default_tensor_type(torch.DoubleTensor)
    unittest_rotational_invariance()
