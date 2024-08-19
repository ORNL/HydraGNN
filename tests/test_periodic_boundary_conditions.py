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

import json, numpy as np

import torch
from torch_geometric.data import Data
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    get_radius_graph_config,
    get_radius_graph_pbc_config,
)

from ase import build


def unittest_periodic_boundary_conditions(
    config, data, expected_neighbors, expected_neighbors_self_loops
):
    compute_edges = get_radius_graph_config(config["Architecture"], loop=False)
    compute_edges_pbc_no_self_loops = get_radius_graph_pbc_config(
        config["Architecture"], loop=False
    )
    compute_edges_pbc_with_self_loops = get_radius_graph_pbc_config(
        config["Architecture"], loop=True
    )
    num_nodes = data.pos.size(0)

    data_periodic_no_self_loops = data.clone()
    data_periodic_with_self_loops = data.clone()

    data = compute_edges(data)

    data_periodic_no_self_loops = compute_edges_pbc_no_self_loops(
        data_periodic_no_self_loops
    )
    data_periodic_with_self_loops = compute_edges_pbc_with_self_loops(
        data_periodic_with_self_loops
    )

    # Check number of nodes is unchanged.
    assert data_periodic_no_self_loops.pos.size(0) == num_nodes
    assert data_periodic_with_self_loops.pos.size(0) == num_nodes

    # Check that each node has the expected number of neighbors (with or without self interactions)
    assert (
        data_periodic_no_self_loops.edge_index.size(1) == expected_neighbors * num_nodes
    )
    assert (
        data_periodic_with_self_loops.edge_index.size(1)
        == expected_neighbors_self_loops * num_nodes
    )

    # Check the nodes were not modified.
    for i in range(num_nodes):
        for d in range(3):
            assert data_periodic_no_self_loops.pos[i][d] == data.pos[i][d]
            assert data_periodic_with_self_loops.pos[i][d] == data.pos[i][d]
        assert data_periodic_no_self_loops.x[i][0] == data.x[i][0]
        assert data_periodic_with_self_loops.x[i][0] == data.x[i][0]
    assert data_periodic_no_self_loops.y == data.y
    assert data_periodic_with_self_loops.y == data.y

    # FIXME: check lengths are at least reasonable
    for n in range(expected_neighbors * num_nodes):
        assert data_periodic_no_self_loops.edge_attr[n] < 5.0
    # Test could also check each periodic shift


def pytest_periodic_h2():
    config_file = "./tests/inputs/ci_periodic.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Create # Hydrogen molecule (H2) with arbitrary node features
    data = Data()
    data.supercell_size = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    )
    data.atom_types = [1, 1]
    data.pos = torch.tensor([[1.0, 1.0, 1.0], [1.43, 1.43, 1.43]])
    data.x = torch.tensor([[3, 5, 7], [9, 11, 13]])
    # Create arbitrary global feature
    data.y = torch.tensor([[99]])

    # Without self loops only 1 bond per atom; with self loops each interacts with itself and its neighbor.
    unittest_periodic_boundary_conditions(config, data, 1, 2)


def pytest_periodic_bcc_large():
    config_file = "./tests/inputs/ci_periodic.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Create BCC Chromium lattice
    config["Architecture"]["radius"] = 5.0
    unitcell = build.bulk("Cr", crystalstructure="bcc", a=3.6, orthorhombic=True)
    supercell = build.make_supercell(unitcell, np.identity(3) * 5)
    # io.write("tmp.xyz", supercell)

    # Convert to PyG
    data2 = Data()
    data2.supercell_size = torch.tensor(supercell.cell[:])
    data2.atom_types = np.ones(len(supercell)) * 27
    data2.pos = torch.tensor(supercell.positions)
    data2.x = torch.randn(data2.pos.size(0), 1)

    # Create arbitrary global feature
    data2.y = torch.tensor([[99]])

    # With this radius and BCC structure, we should get first and second shell neighbors.
    neigh_per_atom = 14
    unittest_periodic_boundary_conditions(
        config, data2, neigh_per_atom, neigh_per_atom + 1
    )
