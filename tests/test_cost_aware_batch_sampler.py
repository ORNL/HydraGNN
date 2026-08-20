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

from hydragnn.preprocess.batch_sampler import CostAwareBatchSampler
from hydragnn.preprocess.load_data import create_dataloaders


def _dataset(costs):
    return [Data(x=torch.zeros(cost, 2)) for cost in costs]


def pytest_cost_sampler_respects_node_and_graph_budgets():
    sampler = CostAwareBatchSampler(
        _dataset([2, 3, 4, 1, 2]),
        max_cost=6,
        max_graphs=2,
        shuffle=False,
    )
    batches = list(sampler)

    assert batches == [[0, 1], [2, 3], [4]]
    assert all(sum(sampler.costs[i] for i in batch) <= 6 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def pytest_cost_sampler_shuffle_is_deterministic_per_epoch():
    dataset = _dataset([1] * 12)
    first = CostAwareBatchSampler(dataset, 3, seed=17)
    second = CostAwareBatchSampler(dataset, 3, seed=17)

    assert list(first) == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != list(CostAwareBatchSampler(dataset, 3, seed=17))


@pytest.mark.parametrize("policy", ["error", "single", "skip"])
def pytest_cost_sampler_has_explicit_oversized_policy(policy):
    sampler = CostAwareBatchSampler(
        _dataset([2, 8, 3]), 5, shuffle=False, oversized_sample=policy
    )
    if policy == "error":
        with pytest.raises(ValueError, match="sample 1.*exceeding"):
            list(sampler)
    elif policy == "single":
        assert list(sampler) == [[0], [1], [2]]
    else:
        assert list(sampler) == [[0, 2]]
        assert sampler.statistics().skipped_samples == 1


def pytest_cost_sampler_uses_precomputed_costs_without_loading_dataset():
    class MetadataOnlyDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            raise AssertionError("samples should not be loaded")

    sampler = CostAwareBatchSampler(
        MetadataOnlyDataset(), 4, costs=[1, 2, 3], shuffle=False
    )
    assert list(sampler) == [[0, 1], [2]]


def pytest_node_budget_configuration_builds_variable_graph_batches():
    dataset = _dataset([2, 3, 4, 1])
    loaders = create_dataloaders(
        dataset,
        dataset,
        dataset,
        batch_size=99,
        train_sampler_shuffle=False,
        val_sampler_shuffle=False,
        test_sampler_shuffle=False,
        batching={"mode": "node_budget", "max_nodes": 5},
    )

    assert [[batch.num_graphs, batch.num_nodes] for batch in loaders[0]] == [
        [2, 5],
        [2, 5],
    ]
