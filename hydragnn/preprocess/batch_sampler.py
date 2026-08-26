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

from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import math
import warnings

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


@dataclass(frozen=True)
class StreamingBatchStatistics:
    """Diagnostics for the most recently started fixed-step training epoch."""

    training_epoch: int
    configured_steps: int
    emitted_steps: int
    emitted_samples: int
    emitted_nodes: int
    min_nodes: int
    max_nodes: int
    mean_nodes: float
    min_utilization: float
    max_utilization: float
    mean_utilization: float
    start_traversal: int
    end_traversal: int
    traversal_boundaries: int
    deferred_samples: int
    oversized_samples: int
    skipped_samples: int
    metadata_requests: int
    metadata_cache_hits: int


_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_PERMUTATION_ROUNDS = 6


def _mix_uint64(value: int) -> int:
    mixed = (int(value) + _SPLITMIX_INCREMENT) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * _SPLITMIX_MULTIPLIER_1) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * _SPLITMIX_MULTIPLIER_2) & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) & _UINT64_MASK


def _permutation_key(seed: int, traversal: int) -> int:
    return _mix_uint64((int(seed) & _UINT64_MASK) ^ _mix_uint64(traversal))


def _permute_power_of_two(value: int, bit_width: int, key: int) -> int:
    mask = (1 << bit_width) - 1
    mixed = int(value) & mask
    for round_index in range(_PERMUTATION_ROUNDS):
        round_key = _mix_uint64(key + round_index * _SPLITMIX_INCREMENT)
        mixed = (mixed + (round_key & mask)) & mask
        multiplier = ((round_key >> 17) | 1) & mask
        mixed = (mixed * (multiplier or 1)) & mask
        shift = 1 + ((round_key >> 41) % bit_width)
        mixed = (mixed ^ (mixed >> shift)) & mask
    return mixed


def stateless_permute_index(
    index: int, dataset_size: int, seed: int, traversal: int
) -> int:
    """Map one index through a traversal-keyed bijection without dense state."""
    size = int(dataset_size)
    index = int(index)
    if size <= 0:
        raise ValueError("dataset_size must be positive")
    if not 0 <= index < size:
        raise ValueError(f"index must be in [0, {size}), got {index}")
    if size == 1:
        return 0

    bit_width = (size - 1).bit_length()
    key = _permutation_key(seed, traversal)
    permuted = _permute_power_of_two(index, bit_width, key)
    while permuted >= size:
        permuted = _permute_power_of_two(permuted, bit_width, key)
    return permuted


class NodeCountProvider:
    """Bounded, backend-neutral access to graph node counts."""

    def __init__(
        self,
        dataset,
        *,
        cost_fn: Callable[[object], int] = None,
        costs: Sequence[int] | None = None,
        total_node_count: int | None = None,
        cache_size: int = 256,
    ) -> None:
        self.dataset = dataset
        self.cost_fn = graph_node_cost if cost_fn is None else cost_fn
        self.costs = costs
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[int, int] = OrderedDict()
        self.requests = 0
        self.cache_hits = 0

        reader = getattr(dataset, "read_node_counts", None)
        self._reader = reader if callable(reader) else None
        range_reader = getattr(dataset, "read_node_counts_range", None)
        self._range_reader = range_reader if callable(range_reader) else None

        if (
            self.costs is None
            and self._reader is None
            and self._range_reader is None
        ):
            getter = getattr(dataset, "get_node_counts", None)
            if callable(getter):
                self.costs = getter()

        if self.costs is None:
            variable_count = getattr(dataset, "variable_count", None)
            variable_dim = getattr(dataset, "variable_dim", None)
            if isinstance(variable_count, dict) and isinstance(variable_dim, dict):
                for key in ("x", "pos"):
                    candidate = variable_count.get(key)
                    if (
                        candidate is not None
                        and variable_dim.get(key) == 0
                        and len(candidate) == len(dataset)
                    ):
                        self.costs = candidate
                        break

        if total_node_count is None:
            total_node_count = getattr(dataset, "total_node_count", None)
        if total_node_count is None:
            getter = getattr(dataset, "get_total_node_count", None)
            if callable(getter):
                total_node_count = getter()
        self.total_node_count = (
            None if total_node_count is None else int(total_node_count)
        )

        if (
            self._reader is None
            and self._range_reader is None
            and self.costs is None
        ):
            warnings.warn(
                "streaming node-budget batching must load dataset samples to "
                "discover node counts; provide read_node_counts(indices) or "
                "resident node-count metadata for out-of-core use",
                RuntimeWarning,
                stacklevel=2,
            )

    def _remember(self, index: int, value: int) -> None:
        if self.cache_size == 0:
            return
        self._cache[index] = value
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def read_node_counts(self, indices: Sequence[int]) -> list[int]:
        normalized = [int(index) for index in indices]
        result: list[int | None] = [None] * len(normalized)
        missing_positions: list[int] = []
        missing_indices: list[int] = []

        for position, index in enumerate(normalized):
            if not 0 <= index < len(self.dataset):
                raise IndexError(index)
            if index in self._cache:
                self.cache_hits += 1
                value = self._cache[index]
                self._cache.move_to_end(index)
                result[position] = value
            else:
                missing_positions.append(position)
                missing_indices.append(index)

        if missing_indices:
            if self._range_reader is not None:
                unique = sorted(set(missing_indices))
                counts_by_index = {}
                run_start = unique[0]
                run_stop = run_start + 1
                runs = []
                for index in unique[1:]:
                    if index == run_stop:
                        run_stop += 1
                    else:
                        runs.append((run_start, run_stop))
                        run_start, run_stop = index, index + 1
                runs.append((run_start, run_stop))
                self.requests += len(runs)
                for start, stop in runs:
                    values = list(self._range_reader(start, stop - start))
                    if len(values) != stop - start:
                        raise RuntimeError(
                            "node-count range reader returned a different number "
                            "of values than requested"
                        )
                    counts_by_index.update(
                        (start + offset, value)
                        for offset, value in enumerate(values)
                    )
                values = [counts_by_index[index] for index in missing_indices]
            elif self._reader is not None:
                self.requests += 1
                values = list(self._reader(missing_indices))
            elif self.costs is not None:
                self.requests += 1
                values = [self.costs[index] for index in missing_indices]
            else:
                self.requests += 1
                values = [
                    self.cost_fn(self.dataset[index]) for index in missing_indices
                ]
            if len(values) != len(missing_indices):
                raise RuntimeError(
                    "node-count provider returned a different number of values "
                    "than requested"
                )
            for position, index, raw_value in zip(
                missing_positions, missing_indices, values
            ):
                value = int(raw_value)
                result[position] = value
                self._remember(index, value)

        return [int(value) for value in result]

    def close(self) -> None:
        close = getattr(self.dataset, "close_node_count_reader", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class _StreamingItem:
    logical_position: int
    physical_index: int
    node_count: int
    traversal: int


def graph_node_cost(sample) -> int:
    """Return the number of nodes in a graph-like sample."""
    value = getattr(sample, "num_nodes", None)
    if value is None:
        features = getattr(sample, "x", None)
        if features is None:
            raise ValueError("sample has neither num_nodes nor node features x")
        value = len(features)
    return int(value)


def graph_node_costs(dataset) -> Sequence[int] | None:
    """Return node counts from dataset metadata without loading samples.

    Dataset implementations may expose a ``get_node_counts`` method. The
    fallback recognizes HydraGNN datasets carrying global variable-shape
    metadata, including DDStore- and ADIOS-backed datasets.
    """
    getter = getattr(dataset, "get_node_counts", None)
    if callable(getter):
        costs = getter()
        if costs is not None:
            return costs

    variable_count = getattr(dataset, "variable_count", None)
    variable_dim = getattr(dataset, "variable_dim", None)
    if isinstance(variable_count, dict) and isinstance(variable_dim, dict):
        for key in ("x", "pos"):
            if key in variable_count and variable_dim.get(key) == 0:
                costs = variable_count[key]
                if len(costs) == len(dataset):
                    return costs
    return None


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
        drop_last: Drop the final batch unless it reaches ``max_cost`` or,
            when configured, ``max_graphs``.
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
        self._batch_cache: tuple[list[list[int]], int] | None = None

        if costs is None:
            costs = graph_node_costs(dataset)
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
        self._batch_cache = None

    def _ordered_indices(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def _batches(self) -> tuple[list[list[int]], int]:
        if self._batch_cache is not None:
            return self._batch_cache

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

        if batch:
            reached_cost_limit = batch_cost == self.max_cost
            reached_graph_limit = (
                self.max_graphs is not None and len(batch) == self.max_graphs
            )
            if not self.drop_last or reached_cost_limit or reached_graph_limit:
                batches.append(batch)

        self._batch_cache = (batches, skipped)
        return self._batch_cache

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


class DistributedCostAwareBatchSampler(CostAwareBatchSampler):
    """Distribute cost-bounded batches with equal steps on every rank.

    Complete batches are created identically on every rank. They are ordered by
    cost and grouped into distributed steps so batches executed concurrently
    have similar estimated costs. Step order and the rank-to-batch mapping are
    shuffled deterministically each epoch.
    """

    def __init__(
        self,
        dataset,
        max_cost: int,
        *,
        num_replicas: int,
        rank: int,
        pad_batches: bool = True,
        **kwargs,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.pad_batches = pad_batches
        super().__init__(dataset, max_cost, **kwargs)

    def _distributed_batches(self) -> list[list[int]]:
        packed_batches, _ = self._batches()
        # Distribution sorts and may pad the global plan. Keep the cached core
        # batches immutable from the perspective of this operation.
        batches = [batch.copy() for batch in packed_batches]
        if not batches:
            return []

        remainder = len(batches) % self.num_replicas
        if remainder:
            if self.pad_batches:
                needed = self.num_replicas - remainder
                batches.extend(batches[index % len(batches)] for index in range(needed))
            else:
                batches = batches[: len(batches) - remainder]
        if not batches:
            return []

        def batch_cost(batch):
            return sum(self.costs[index] for index in batch)

        batches.sort(key=batch_cost)
        steps = [
            batches[start : start + self.num_replicas]
            for start in range(0, len(batches), self.num_replicas)
        ]

        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            step_order = torch.randperm(len(steps), generator=generator).tolist()
            steps = [steps[index] for index in step_order]

        local_batches = []
        for step, candidates in enumerate(steps):
            # Rotation prevents one rank from consistently receiving the most
            # expensive member of each similarly sized group.
            local_index = (self.rank + step) % self.num_replicas
            local_batches.append(candidates[local_index])
        return local_batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._distributed_batches())

    def __len__(self) -> int:
        batches, _ = self._batches()
        if self.pad_batches:
            return math.ceil(len(batches) / self.num_replicas)
        return len(batches) // self.num_replicas


class StreamingNodeBudgetBatchSampler(Sampler[list[int]]):
    """Yield a fixed number of batches from a continuous shuffled stream.

    Dataset traversals and training epochs are intentionally independent. Each
    rank owns a disjoint logical range within a traversal and advances to its
    next traversal as soon as that range is exhausted. Sampler state is kept in
    memory across ``__iter__`` calls but is deliberately not checkpointed.
    """

    is_streaming_node_budget = True

    def __init__(
        self,
        dataset,
        max_nodes: int,
        *,
        steps_per_epoch: int | None = None,
        target_nodes: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        max_graphs: int | None = None,
        metadata_chunk_size: int = 32,
        forward_window: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        oversized_sample: str = "error",
        provider: NodeCountProvider | None = None,
        costs: Sequence[int] | None = None,
        total_node_count: int | None = None,
        cost_fn: Callable[[object], int] = graph_node_cost,
        metadata_cache_size: int | None = None,
    ) -> None:
        if len(dataset) <= 0:
            raise ValueError("streaming node-budget dataset cannot be empty")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        target_nodes = max_nodes if target_nodes is None else int(target_nodes)
        if target_nodes <= 0:
            raise ValueError("target_nodes must be positive")
        if target_nodes > max_nodes:
            raise ValueError("target_nodes cannot exceed max_nodes")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if len(dataset) < num_replicas:
            raise ValueError("dataset size must be at least num_replicas")
        if max_graphs is not None and max_graphs <= 0:
            raise ValueError("max_graphs must be positive when provided")
        if metadata_chunk_size <= 0:
            raise ValueError("metadata_chunk_size must be positive")
        if forward_window < 0:
            raise ValueError("forward_window cannot be negative")
        if oversized_sample not in {"error", "single", "skip"}:
            raise ValueError("oversized_sample must be one of: error, single, skip")

        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.max_nodes = int(max_nodes)
        self.target_nodes = target_nodes
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.max_graphs = max_graphs
        self.metadata_chunk_size = int(metadata_chunk_size)
        self.forward_window = int(forward_window)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.oversized_sample = oversized_sample
        cache_size = (
            8 * self.metadata_chunk_size
            if metadata_cache_size is None
            else int(metadata_cache_size)
        )
        self.provider = provider or NodeCountProvider(
            dataset,
            cost_fn=cost_fn,
            costs=costs,
            total_node_count=total_node_count,
            cache_size=cache_size,
        )

        if steps_per_epoch is None:
            total = self.provider.total_node_count
            if total is None:
                raise ValueError(
                    "streaming_node_budget requires steps_per_epoch when the "
                    "dataset has no exact total_node_count metadata"
                )
            steps_per_epoch = math.ceil(
                total / (self.target_nodes * self.num_replicas)
            )
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.steps_per_epoch = int(steps_per_epoch)

        self.training_epoch = 0
        self.traversal = 0
        self._started = False
        self._initial_epoch_set = False
        self._iterator_active = False
        self._pending: deque[_StreamingItem] = deque()
        self._range_start = 0
        self._range_stop = 0
        self._next_position = 0
        self._batches_in_traversal = 0
        self._last_batch_traversal = 0
        self._reset_epoch_statistics()
        self._start_traversal(0)

    @property
    def last_batch_traversal(self) -> int:
        """Traversal identity of the most recently emitted batch."""
        return self._last_batch_traversal

    @property
    def logical_cursor(self) -> tuple[int, int]:
        """Current traversal and next unread logical position."""
        return self.traversal, self._next_position

    def _quota(self, rank: int) -> int:
        base, remainder = divmod(self.dataset_size, self.num_replicas)
        return base + int(rank < remainder)

    def _start_traversal(self, traversal: int) -> None:
        self.traversal = int(traversal)
        self._range_start = sum(self._quota(rank) for rank in range(self.rank))
        self._range_stop = self._range_start + self._quota(self.rank)
        self._next_position = self._range_start
        self._pending.clear()
        self._batches_in_traversal = 0

    def _reset_epoch_statistics(self) -> None:
        self._stats_start_traversal = getattr(self, "traversal", 0)
        self._stats_boundaries = 0
        self._stats_steps = 0
        self._stats_samples = 0
        self._stats_nodes = 0
        self._stats_min_nodes = None
        self._stats_max_nodes = 0
        self._stats_deferred = 0
        self._stats_oversized = 0
        self._stats_skipped = 0
        self._stats_provider_requests = getattr(
            getattr(self, "provider", None), "requests", 0
        )
        self._stats_provider_hits = getattr(
            getattr(self, "provider", None), "cache_hits", 0
        )

    def set_epoch(self, epoch: int) -> None:
        """Record a fixed-step training epoch without resetting the stream."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if self._iterator_active:
            raise RuntimeError("cannot change epoch while a sampler iterator is active")
        if not self._started and not self._initial_epoch_set:
            self._start_traversal(epoch)
        self._initial_epoch_set = True
        self.training_epoch = epoch
        self._reset_epoch_statistics()

    def _physical_index(self, logical_position: int) -> int:
        if not self.shuffle:
            return logical_position
        return stateless_permute_index(
            logical_position,
            self.dataset_size,
            seed=self.seed,
            traversal=self.traversal,
        )

    def _take_chunk(self) -> list[_StreamingItem]:
        chunk: list[_StreamingItem] = []
        while self._pending and len(chunk) < self.metadata_chunk_size:
            chunk.append(self._pending.popleft())

        count = min(
            self.metadata_chunk_size - len(chunk),
            self._range_stop - self._next_position,
        )
        if count <= 0:
            return chunk

        logical_positions = list(
            range(self._next_position, self._next_position + count)
        )
        physical_indices = [
            self._physical_index(position) for position in logical_positions
        ]
        node_counts = self.provider.read_node_counts(physical_indices)
        chunk.extend(
            _StreamingItem(position, index, nodes, self.traversal)
            for position, index, nodes in zip(
                logical_positions, physical_indices, node_counts
            )
        )
        self._next_position += count
        return chunk

    def _restore_deferred(self, items: Sequence[_StreamingItem]) -> None:
        self._stats_deferred += len(items)
        for item in reversed(sorted(items, key=lambda value: value.logical_position)):
            self._pending.appendleft(item)

    def _consider_add_or_swap(
        self,
        item: _StreamingItem,
        selected: list[_StreamingItem],
        deferred: list[_StreamingItem],
        total: int,
    ) -> tuple[int, bool]:
        current_distance = abs(total - self.target_nodes)
        moves: list[tuple[tuple[int, int, int, int], int | None, int]] = []

        added_total = total + item.node_count
        if (
            (self.max_graphs is None or len(selected) < self.max_graphs)
            and added_total <= self.max_nodes
            and abs(added_total - self.target_nodes) < current_distance
        ):
            moves.append(
                (
                    (
                        abs(added_total - self.target_nodes),
                        added_total,
                        0,
                        item.logical_position,
                    ),
                    None,
                    added_total,
                )
            )

        for index, old_item in enumerate(selected):
            swapped_total = total - old_item.node_count + item.node_count
            distance = abs(swapped_total - self.target_nodes)
            if 0 < swapped_total <= self.max_nodes and distance < current_distance:
                moves.append(
                    (
                        (distance, swapped_total, 1, old_item.logical_position),
                        index,
                        swapped_total,
                    )
                )

        if not moves:
            deferred.append(item)
            return total, True

        _, replacement_index, new_total = min(moves, key=lambda move: move[0])
        if replacement_index is None:
            selected.append(item)
        else:
            deferred.append(selected[replacement_index])
            selected[replacement_index] = item
        return new_total, False

    def _build_batch(self) -> list[_StreamingItem]:
        selected: list[_StreamingItem] = []
        deferred: list[_StreamingItem] = []
        total = 0
        target_reached = False
        forward_chunks_remaining = self.forward_window

        while True:
            chunk = self._take_chunk()
            if not chunk:
                break
            chunk_rejected = False

            for item_index, item in enumerate(chunk):
                if item.node_count <= 0:
                    self._restore_deferred(deferred + chunk[item_index + 1 :])
                    raise ValueError(
                        f"node count must be positive; sample "
                        f"{item.physical_index} has {item.node_count}"
                    )
                if item.node_count > self.max_nodes:
                    self._stats_oversized += 1
                    if self.oversized_sample == "error":
                        self._restore_deferred(deferred + chunk[item_index + 1 :])
                        raise ValueError(
                            f"sample {item.physical_index} has cost "
                            f"{item.node_count}, exceeding max_nodes {self.max_nodes}"
                        )
                    if self.oversized_sample == "skip":
                        self._stats_skipped += 1
                        continue
                    if selected:
                        deferred.append(item)
                        deferred.extend(chunk[item_index + 1 :])
                        self._restore_deferred(deferred)
                        return selected
                    deferred.extend(chunk[item_index + 1 :])
                    self._restore_deferred(deferred)
                    return [item]

                if not selected:
                    selected.append(item)
                    total = item.node_count
                    target_reached = total >= self.target_nodes
                elif self.max_graphs is not None and len(selected) >= self.max_graphs:
                    deferred.append(item)
                    deferred.extend(chunk[item_index + 1 :])
                    self._restore_deferred(deferred)
                    return selected
                elif target_reached:
                    total, rejected = self._consider_add_or_swap(
                        item, selected, deferred, total
                    )
                    chunk_rejected = chunk_rejected or rejected
                else:
                    candidate_total = total + item.node_count
                    improves = (
                        candidate_total <= self.max_nodes
                        and abs(candidate_total - self.target_nodes)
                        < abs(total - self.target_nodes)
                    )
                    if improves:
                        selected.append(item)
                        total = candidate_total
                        target_reached = total >= self.target_nodes
                    else:
                        deferred.append(item)
                        chunk_rejected = True

                if total == self.target_nodes:
                    deferred.extend(chunk[item_index + 1 :])
                    self._restore_deferred(deferred)
                    return selected

            if target_reached:
                if forward_chunks_remaining == 0:
                    break
                forward_chunks_remaining -= 1
            elif chunk_rejected:
                break

        self._restore_deferred(deferred)
        return selected

    def _next_batch(self) -> list[_StreamingItem]:
        while True:
            batch = self._build_batch()
            if batch:
                self._batches_in_traversal += 1
                return sorted(batch, key=lambda item: item.logical_position)

            if self._pending or self._next_position < self._range_stop:
                raise RuntimeError("streaming packer made no progress")
            if self._batches_in_traversal == 0:
                raise RuntimeError(
                    f"rank {self.rank} could not emit a valid batch from "
                    f"traversal {self.traversal}"
                )
            self._stats_boundaries += 1
            self._start_traversal(self.traversal + 1)

    def __iter__(self) -> Iterator[list[int]]:
        return self._iterate()

    def _iterate(self) -> Iterator[list[int]]:
        if self._iterator_active:
            raise RuntimeError(
                "only one active streaming sampler iterator is supported"
            )
        self._iterator_active = True
        self._started = True
        try:
            for _ in range(self.steps_per_epoch):
                batch = self._next_batch()
                self._last_batch_traversal = batch[0].traversal
                node_total = sum(item.node_count for item in batch)
                self._stats_steps += 1
                self._stats_samples += len(batch)
                self._stats_nodes += node_total
                self._stats_min_nodes = (
                    node_total
                    if self._stats_min_nodes is None
                    else min(self._stats_min_nodes, node_total)
                )
                self._stats_max_nodes = max(self._stats_max_nodes, node_total)
                yield [item.physical_index for item in batch]
        finally:
            self._iterator_active = False

    def __len__(self) -> int:
        return self.steps_per_epoch

    def statistics(self) -> StreamingBatchStatistics:
        min_nodes = 0 if self._stats_min_nodes is None else self._stats_min_nodes
        mean_nodes = (
            self._stats_nodes / self._stats_steps if self._stats_steps else 0.0
        )
        return StreamingBatchStatistics(
            training_epoch=self.training_epoch,
            configured_steps=self.steps_per_epoch,
            emitted_steps=self._stats_steps,
            emitted_samples=self._stats_samples,
            emitted_nodes=self._stats_nodes,
            min_nodes=min_nodes,
            max_nodes=self._stats_max_nodes,
            mean_nodes=mean_nodes,
            min_utilization=min_nodes / self.max_nodes,
            max_utilization=self._stats_max_nodes / self.max_nodes,
            mean_utilization=mean_nodes / self.max_nodes,
            start_traversal=self._stats_start_traversal,
            end_traversal=self.traversal,
            traversal_boundaries=self._stats_boundaries,
            deferred_samples=self._stats_deferred,
            oversized_samples=self._stats_oversized,
            skipped_samples=self._stats_skipped,
            metadata_requests=self.provider.requests - self._stats_provider_requests,
            metadata_cache_hits=self.provider.cache_hits - self._stats_provider_hits,
        )

    def close(self) -> None:
        self.provider.close()
