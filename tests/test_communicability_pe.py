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

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from hydragnn.preprocess.positional_encodings import (
    AddCommunicabilityPE,
    add_relative_pe,
    create_positional_encoder,
)


def _path_graph_data():
    # Undirected path with 3 nodes represented as bidirectional edges.
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    return Data(edge_index=edge_index, num_nodes=3)


def pytest_create_positional_encoder_defaults_to_laplacian():
    encoder = create_positional_encoder({"pe_dim": 2})
    assert isinstance(encoder, AddLaplacianEigenvectorPE)


def pytest_communicability_encoder_produces_expected_shapes_and_symmetry():
    data = _path_graph_data()
    transform = AddCommunicabilityPE(k=3, method="katz")
    data = transform(data)
    data = add_relative_pe(data)

    assert data.pe.shape == (3, 3)
    assert data.rel_pe.shape == (4, 3)
    assert torch.isfinite(data.pe).all()
    # Endpoints in a symmetric path should have identical node PE.
    assert torch.allclose(data.pe[0], data.pe[2], atol=1.0e-5)


def pytest_factory_supports_communicability_encoder():
    cfg = {
        "pe_dim": 4,
        "pe_encoder": "communicability",
        "communicability_method": "adjacency_powers",
    }
    encoder = create_positional_encoder(cfg)
    data = _path_graph_data()
    data = encoder(data)

    assert data.pe.shape == (3, 4)
    assert torch.isfinite(data.pe).all()