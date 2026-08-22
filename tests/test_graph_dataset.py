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

import pickle

import pytest
import torch
from torch_geometric.data import Data

from hydragnn.preprocess.graph_dataset import load_prepared_graph_dataset


def pytest_prepared_pickle_is_loaded_without_modifying_graphs(tmp_path):
    graph = Data(
        x=torch.tensor([[1.0], [2.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[7.5]]),
        y=torch.tensor([[3.0]]),
        custom_descriptor=torch.tensor([[9.0, 8.0]]),
    )
    dataset_path = tmp_path / "prepared.pkl"
    with open(dataset_path, "wb") as stream:
        pickle.dump("node metadata", stream)
        pickle.dump("graph metadata", stream)
        pickle.dump([graph], stream)

    loaded = load_prepared_graph_dataset(dataset_path)

    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0].x, graph.x)
    torch.testing.assert_close(loaded[0].edge_index, graph.edge_index)
    torch.testing.assert_close(loaded[0].edge_attr, graph.edge_attr)
    torch.testing.assert_close(loaded[0].y, graph.y)
    torch.testing.assert_close(
        loaded[0].custom_descriptor, graph.custom_descriptor
    )


def pytest_prepared_pt_directory_is_loaded_without_modifying_graphs(tmp_path):
    first = Data(x=torch.tensor([[1.0]]), marker=torch.tensor([[4.0]]))
    second = Data(x=torch.tensor([[2.0]]), marker=torch.tensor([[5.0]]))
    torch.save(second, tmp_path / "sample-1.pt")
    torch.save(first, tmp_path / "sample-0.pt")

    loaded = load_prepared_graph_dataset(tmp_path)

    assert len(loaded) == 2
    torch.testing.assert_close(loaded[0].marker, first.marker)
    torch.testing.assert_close(loaded[1].marker, second.marker)


def pytest_prepared_dataset_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No serialized PyG .pt samples"):
        load_prepared_graph_dataset(tmp_path)
