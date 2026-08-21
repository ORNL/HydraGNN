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

from hydragnn.preprocess import graph_dataset


def pytest_stratified_sampling_ignores_absent_species_bins(monkeypatch):
    observed = {}

    class RecordingSplitter:
        def __init__(self, **kwargs):
            pass

        def split(self, dataset, categories):
            observed["categories"] = categories
            yield [0], [1]

    monkeypatch.setattr(graph_dataset, "StratifiedShuffleSplit", RecordingSplitter)
    monkeypatch.setattr(
        graph_dataset, "iterate_tqdm", lambda dataset, verbosity: dataset
    )
    dataset = [
        Data(x=torch.tensor([[0.0], [0.0], [2.0]])),
        Data(x=torch.tensor([[0.0], [2.0], [2.0]])),
    ]

    sampled = graph_dataset._stratified_sample(dataset, 0.5, verbosity=0)

    assert sampled == [dataset[0]]
    assert observed["categories"] == [201, 201]


def pytest_periodic_edges_require_cell():
    data = Data(pos=torch.tensor([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="require data.cell"):
        graph_dataset._build_edges(data, 2.0, 8, periodic=True)


def pytest_periodic_edges_are_constructed_from_cell():
    data = Data(
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cell=5.0 * torch.eye(3),
    )

    result = graph_dataset._build_edges(data, 1.5, 8, periodic=True)

    assert result is data
    torch.testing.assert_close(data.pbc, torch.tensor([True, True, True]))
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0
