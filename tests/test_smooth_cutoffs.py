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

import pytest
import torch
from torch_geometric.data import Data

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    get_radius_graph,
    get_radius_graph_config,
)
from hydragnn.utils.input_config_parsing.config_utils import (
    validate_neighbor_list_config,
)
from hydragnn.utils.model.cutoffs import septic_cutoff


@pytest.mark.parametrize("boundary", [1.0, 2.0])
def pytest_septic_cutoff_matches_through_third_derivative(boundary):
    direction = 1.0 if boundary == 1.0 else -1.0
    distance = torch.tensor(
        boundary + direction * 1.0e-8, dtype=torch.float64, requires_grad=True
    )
    value = septic_cutoff(distance, onset=1.0, cutoff=2.0)
    derivatives = []
    current = value
    for _ in range(3):
        (current,) = torch.autograd.grad(current, distance, create_graph=True)
        derivatives.append(current)

    expected_value = 1.0 if boundary == 1.0 else 0.0
    assert value.item() == pytest.approx(expected_value, abs=1.0e-12)
    assert all(abs(item.item()) < 1.0e-4 for item in derivatives)


def pytest_radius_graph_rejects_neighbor_overflow():
    data = Data(
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
    )

    with pytest.raises(RuntimeError, match="silent truncation would break smoothness"):
        get_radius_graph(1.0, max_neighbours=2, overflow="error")(data)


def pytest_buffered_radius_requires_native_cutoff_model():
    with pytest.raises(ValueError, match="native physical cutoff"):
        validate_neighbor_list_config(
            {
                "mpnn_type": "SAGE",
                "radius": 4.0,
                "neighbor_list_radius": 4.5,
                "neighbor_overflow": "error",
            }
        )


def pytest_buffered_radius_accepts_native_cutoff_model():
    validate_neighbor_list_config(
        {
            "mpnn_type": "MACE",
            "radius": 4.0,
            "neighbor_list_radius": 4.5,
            "neighbor_overflow": "error",
        }
    )


def pytest_buffered_radius_builds_candidates_beyond_physical_cutoff():
    data = Data(pos=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))
    graph = get_radius_graph_config(
        {
            "mpnn_type": "MACE",
            "radius": 1.0,
            "neighbor_list_radius": 1.5,
            "max_neighbours": 4,
            "neighbor_overflow": "error",
        }
    )(data)

    assert graph.edge_index.shape[1] == 2
