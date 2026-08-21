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

from hydragnn.globalAtt.complete_graph import complete_graph_edge_index


def pytest_complete_graph_constructs_ordered_non_self_pairs():
    edge_index = complete_graph_edge_index(torch.tensor([0, 0, 0]))

    expected = torch.tensor(
        [
            [1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2],
        ]
    )
    assert torch.equal(edge_index, expected)


def pytest_complete_graph_keeps_batched_graphs_isolated():
    batch = torch.tensor([3, 3, 8, 8, 8, 11])
    edge_index = complete_graph_edge_index(batch)
    source, target = edge_index

    assert edge_index.shape == (2, 8)
    assert torch.all(source != target)
    assert torch.equal(batch[source], batch[target])

    actual_pairs = set(zip(source.tolist(), target.tolist()))
    expected_pairs = {
        (0, 1),
        (1, 0),
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 4),
        (4, 2),
        (4, 3),
    }
    assert actual_pairs == expected_pairs


def pytest_complete_graph_handles_empty_and_singleton_batches():
    empty = complete_graph_edge_index(torch.empty(0, dtype=torch.long))
    singletons = complete_graph_edge_index(torch.tensor([0, 2, 7]))

    assert empty.shape == (2, 0)
    assert singletons.shape == (2, 0)
    assert empty.dtype == torch.long


@pytest.mark.parametrize(
    ("batch", "error", "message"),
    [
        (torch.tensor([[0, 0]]), ValueError, "one-dimensional"),
        (torch.tensor([0.0, 0.0]), TypeError, "torch.long"),
        (torch.tensor([0, -1]), ValueError, "nonnegative"),
    ],
)
def pytest_complete_graph_rejects_invalid_batches(batch, error, message):
    with pytest.raises(error, match=message):
        complete_graph_edge_index(batch)
