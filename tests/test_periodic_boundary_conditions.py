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
from hydragnn.preprocess.utils import (
    get_radius_graph_config,
    get_radius_graph_pbc_config,
)


def unittest_periodic_boundary_conditions():
    config_file = "./tests/inputs/ci_periodic.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    compute_edges = get_radius_graph_config(config["Architecture"], loop=False)
    compute_edges_pbc_no_self_loops = get_radius_graph_pbc_config(
        config["Architecture"], loop=False
    )
    compute_edges_pbc_with_self_loops = get_radius_graph_pbc_config(
        config["Architecture"], loop=True
    )

    # Create two nodes with arbitrary node features
    data = Data()
    data.supercell_size = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    data.atom_types = [1, 1]  # Hydrogen molecule (H2)
    data.pos = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    data.x = torch.tensor([[3, 5, 7], [9, 11, 13]])

    # Create arbitrary global feature
    data.y = torch.tensor([[99]])

    data_periodic_no_self_loops = data.clone()
    data_periodic_with_self_loops = data.clone()

    data = compute_edges(data)

    # check that there are two edges without self loops
    assert data.edge_index.size(1)

    data_periodic_no_self_loops = compute_edges_pbc_no_self_loops(
        data_periodic_no_self_loops
    )
    data_periodic_with_self_loops = compute_edges_pbc_with_self_loops(
        data_periodic_with_self_loops
    )

    # Check that there's still two nodes.
    assert data_periodic_no_self_loops.edge_index.size(0) == 2
    assert data_periodic_with_self_loops.edge_index.size(0) == 2

    # check that the number of edges with periodic boundary conditions does not change if self loops are excluded
    assert data_periodic_no_self_loops.edge_index.size(1) == data.edge_index.size(1)

    # check that the periodic boundary condition introduces additional edges if self loops are included
    assert data.edge_index.size(1) < data_periodic_with_self_loops.edge_index.size(1)
    # Check that there's one "real" bond and 26 ghost bonds (for both nodes).
    assert data_periodic_with_self_loops.edge_index.size(1) == 4

    # Check the nodes were not modified.
    for i in range(2):
        for d in range(3):
            assert data_periodic_no_self_loops.pos[i][d] == data.pos[i][d]
            assert data_periodic_with_self_loops.pos[i][d] == data.pos[i][d]
        assert data_periodic_no_self_loops.x[i][0] == data.x[i][0]
        assert data_periodic_with_self_loops.x[i][0] == data.x[i][0]
    assert data_periodic_no_self_loops.y == data.y
    assert data_periodic_with_self_loops.y == data.y


def pytest_train_model():
    unittest_periodic_boundary_conditions()
