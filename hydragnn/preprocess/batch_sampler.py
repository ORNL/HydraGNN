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

"""Cost-aware batching for variable-size graph datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import math

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class BatchStatistics:
    """Summary of the batches constructed for one sampler epoch."""

    num_batches: int
    num_samples: int
    skipped_samples: int
    min_cost: int
    max_cost: int
    mean_cost: float


def graph_node_cost(sample) -> int:
    """Return the number of nodes in a graph-like sample."""
    value = getattr(sample, "num_nodes", None)
    if value is None:
        features = getattr(sample, "x", None)
        if features is None:
            raise ValueError("sample has neither num_nodes nor node features x")
        value = len(features)
    return int(value)


class CostAwareBatchSampler(Sampler[list[int]]):
    """Pack dataset samples into batches bounded by an additive cost.

    Sample order is shuffled deterministically from ``seed + epoch``. Batches
    are then constructed greedily, retaining dataset order when shuffling is
    disabled. Costs are computed once and cached for the sampler lifetime.

    Args:
        dataset: Indexable dataset.
        max_cost: Maximum summed cost of an ordinary batch.
        cost_fn: Function mapping a dataset sample to a positive integer cost.
        costs: Optional precomputed costs, avoiding sample materialization.
        max_graphs: Optional maximum number of samples in a batch.
        shuffle: Whether to shuffle samples before packing.
        seed: Base seed used for deterministic epoch shuffling.
        oversized_sample: Policy for a sample whose cost exceeds ``max_cost``:
            ``"error"``, ``"single"``, or ``"skip"``.
        drop_last: Drop the final batch when its cost is below ``max_cost``.
    """

    def __init__(
        self,
        dataset,
        max_cost: int,
        *,
        cost_fn: Callable[[object], int] = graph_node_cost,
        costs: Sequence[int] | None = None,
        max_graphs: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        oversized_sample: str = "error",
        drop_last: bool = False,
    ) -> None:
        if max_cost <= 0:
            raise ValueError("max_cost must be positive")
        if max_graphs is not None and max_graphs <= 0:
            raise ValueError("max_graphs must be positive when provided")
        if oversized_sample not in {"error", "single", "skip"}:
            raise ValueError("oversized_sample must be one of: error, single, skip")

        self.dataset = dataset
        self.max_cost = int(max_cost)
        self.max_graphs = max_graphs
        self.shuffle = shuffle
        self.seed = int(seed)
        self.oversized_sample = oversized_sample
        self.drop_last = drop_last
        self.epoch = 0

        if costs is None:
            costs = [cost_fn(dataset[index]) for index in range(len(dataset))]
        if len(costs) != len(dataset):
            raise ValueError("costs must contain one value per dataset sample")
        self.costs = tuple(int(cost) for cost in costs)
        if any(cost <= 0 for cost in self.costs):
            raise ValueError("all sample costs must be positive")

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic ordering for an epoch."""
        self.epoch = int(epoch)

    def _ordered_indices(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def _batches(self) -> tuple[list[list[int]], int]:
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_cost = 0
        skipped = 0

        for index in self._ordered_indices():
            cost = self.costs[index]
            if cost > self.max_cost:
                if self.oversized_sample == "error":
                    raise ValueError(
                        f"sample {index} has cost {cost}, exceeding max_cost "
                        f"{self.max_cost}"
                    )
                if self.oversized_sample == "skip":
                    skipped += 1
                    continue
                if batch:
                    batches.append(batch)
                    batch, batch_cost = [], 0
                batches.append([index])
                continue

            exceeds_cost = batch_cost + cost > self.max_cost
            exceeds_graphs = (
                self.max_graphs is not None and len(batch) >= self.max_graphs
            )
            if batch and (exceeds_cost or exceeds_graphs):
                batches.append(batch)
                batch, batch_cost = [], 0
            batch.append(index)
            batch_cost += cost

        if batch and not (self.drop_last and batch_cost < self.max_cost):
            batches.append(batch)
        return batches, skipped

    def __iter__(self) -> Iterator[list[int]]:
        batches, _ = self._batches()
        return iter(batches)

    def __len__(self) -> int:
        batches, _ = self._batches()
        return len(batches)

    def statistics(self) -> BatchStatistics:
        """Return packing diagnostics for the currently selected epoch."""
        batches, skipped = self._batches()
        totals = [sum(self.costs[index] for index in batch) for batch in batches]
        return BatchStatistics(
            num_batches=len(batches),
            num_samples=sum(map(len, batches)),
            skipped_samples=skipped,
            min_cost=min(totals, default=0),
            max_cost=max(totals, default=0),
            mean_cost=math.fsum(totals) / len(totals) if totals else 0.0,
        )
