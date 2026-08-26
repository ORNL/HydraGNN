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

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from hydragnn.preprocess.batch_sampler import (
    NodeCountProvider,
    StreamingNodeBudgetBatchSampler,
    stateless_permute_index,
)
from hydragnn.preprocess.load_data import HydraDataLoader, create_dataloaders


class _MetadataDataset:
    def __init__(self, counts, *, total=True):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.reads = []
        self.total_node_count = int(self.counts.sum()) if total else None

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, _index):
        raise AssertionError("graph payload must not be read by the sampler")

    def read_node_counts(self, indices):
        indices = [int(index) for index in indices]
        self.reads.append(tuple(indices))
        return self.counts[indices]


class _RangeMetadataDataset(_MetadataDataset):
    def read_node_counts_range(self, start, count):
        self.reads.append((int(start), int(count)))
        return self.counts[start : start + count]


class _GraphDataset(list):
    @property
    def total_node_count(self):
        return sum(graph.num_nodes for graph in self)

    def read_node_counts(self, indices):
        return [self[int(index)].num_nodes for index in indices]


def _graphs(counts):
    return _GraphDataset(
        Data(x=torch.zeros(count, 2), y=torch.zeros(1)) for count in counts
    )


def pytest_streaming_stateless_mapping_is_bijective_and_deterministic():
    for size in (1, 2, 3, 16, 17, 257):
        first = [stateless_permute_index(i, size, 31, 4) for i in range(size)]
        repeated = [stateless_permute_index(i, size, 31, 4) for i in range(size)]
        assert sorted(first) == list(range(size))
        assert repeated == first
    assert [stateless_permute_index(i, 257, 31, 4) for i in range(257)] != [
        stateless_permute_index(i, 257, 31, 5) for i in range(257)
    ]


def pytest_streaming_rank_ranges_cover_each_traversal_once():
    dataset = _MetadataDataset([1] * 102)
    samplers = [
        StreamingNodeBudgetBatchSampler(
            dataset,
            1,
            steps_per_epoch=34,
            num_replicas=3,
            rank=rank,
            seed=19,
        )
        for rank in range(3)
    ]
    emitted = [index for sampler in samplers for batch in sampler for index in batch]
    assert sorted(emitted) == list(range(102))


def pytest_streaming_fixed_steps_cross_independent_96_104_traversals():
    counts = [5] * 192 + [10] * 16 + [5] * 176
    dataset = _MetadataDataset(counts)
    rank0 = StreamingNodeBudgetBatchSampler(
        dataset,
        10,
        target_nodes=10,
        steps_per_epoch=100,
        num_replicas=2,
        rank=0,
        shuffle=False,
        metadata_chunk_size=8,
    )
    rank1 = StreamingNodeBudgetBatchSampler(
        dataset,
        10,
        target_nodes=10,
        steps_per_epoch=100,
        num_replicas=2,
        rank=1,
        shuffle=False,
        metadata_chunk_size=8,
    )

    rank0_traversals = []
    rank1_traversals = []
    for _batch in rank0:
        rank0_traversals.append(rank0.last_batch_traversal)
    for _batch in rank1:
        rank1_traversals.append(rank1.last_batch_traversal)

    assert rank0_traversals == [0] * 96 + [1] * 4
    assert rank1_traversals == [0] * 100
    next_rank1 = []
    for _batch in rank1:
        next_rank1.append(rank1.last_batch_traversal)
    assert next_rank1[:4] == [0] * 4
    assert next_rank1[4:] == [1] * 96
    assert len(rank0) == len(rank1) == 100


def pytest_streaming_emits_final_partial_before_next_traversal():
    sampler = StreamingNodeBudgetBatchSampler(
        _MetadataDataset([1] * 5),
        2,
        steps_per_epoch=4,
        shuffle=False,
        metadata_chunk_size=2,
    )
    batches = []
    traversals = []
    for batch in sampler:
        batches.append(batch)
        traversals.append(sampler.last_batch_traversal)
    assert batches == [[0, 1], [2, 3], [4], [0, 1]]
    assert traversals == [0, 0, 0, 1]


def pytest_streaming_set_epoch_does_not_reset_an_active_stream():
    sampler = StreamingNodeBudgetBatchSampler(
        _MetadataDataset([1] * 8),
        2,
        steps_per_epoch=3,
        shuffle=False,
    )
    assert list(sampler) == [[0, 1], [2, 3], [4, 5]]
    sampler.set_epoch(1)
    assert list(sampler)[:2] == [[6, 7], [0, 1]]
    assert sampler.statistics().start_traversal == 0
    assert sampler.statistics().end_traversal == 1


def pytest_streaming_fresh_nonzero_epoch_selects_initial_shuffle():
    dataset = _MetadataDataset([1] * 17)
    sampler = StreamingNodeBudgetBatchSampler(
        dataset, 1, steps_per_epoch=3, seed=7
    )
    sampler.set_epoch(9)
    sampler.set_epoch(10)
    batches = list(sampler)
    assert [batch[0] for batch in batches] == [
        stateless_permute_index(index, 17, 7, 9) for index in range(3)
    ]


def pytest_streaming_derives_steps_only_from_exact_aggregate():
    derived = StreamingNodeBudgetBatchSampler(
        _MetadataDataset([2] * 10), 5, target_nodes=5, num_replicas=2
    )
    assert len(derived) == 2

    with pytest.raises(ValueError, match="requires steps_per_epoch"):
        StreamingNodeBudgetBatchSampler(
            _MetadataDataset([2] * 10, total=False),
            5,
            target_nodes=5,
            num_replicas=2,
        )

    class ResidentCountsWithoutAggregate:
        def __len__(self):
            return 10

        def get_node_counts(self):
            return [2] * 10

    with pytest.raises(ValueError, match="requires steps_per_epoch"):
        StreamingNodeBudgetBatchSampler(
            ResidentCountsWithoutAggregate(),
            5,
            target_nodes=5,
            num_replicas=2,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_nodes": 0}, "max_nodes must be positive"),
        ({"target_nodes": 0}, "target_nodes must be positive"),
        ({"target_nodes": 6}, "cannot exceed max_nodes"),
        ({"steps_per_epoch": 0}, "steps_per_epoch must be positive"),
        ({"metadata_chunk_size": 0}, "metadata_chunk_size must be positive"),
        ({"forward_window": -1}, "forward_window cannot be negative"),
        ({"max_graphs": 0}, "max_graphs must be positive"),
    ],
)
def pytest_streaming_rejects_nonpositive_and_inconsistent_values(
    overrides, message
):
    options = {"max_nodes": 5, "steps_per_epoch": 1}
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        StreamingNodeBudgetBatchSampler(_MetadataDataset([1, 2]), **options)


def pytest_streaming_provider_coalesces_ranges_and_preserves_request_order():
    dataset = _RangeMetadataDataset(np.arange(12) + 1)
    provider = NodeCountProvider(dataset, cache_size=16)
    assert provider.read_node_counts([5, 2, 3, 8, 8]) == [6, 3, 4, 9, 9]
    assert dataset.reads == [(2, 2), (5, 1), (8, 1)]
    assert provider.read_node_counts([8, 3]) == [9, 4]
    assert provider.cache_hits == 2


def pytest_streaming_generic_provider_warns_and_closes_backend():
    class ClosingDataset(_MetadataDataset):
        def __init__(self):
            super().__init__([1, 2])
            self.closed = False

        def close_node_count_reader(self):
            self.closed = True

    dataset = ClosingDataset()
    provider = NodeCountProvider(dataset)
    provider.close()
    assert dataset.closed

    plain_graphs = [Data(x=torch.zeros(2, 1)), Data(x=torch.zeros(3, 1))]
    with pytest.warns(RuntimeWarning, match="must load dataset samples"):
        fallback = NodeCountProvider(plain_graphs)
    assert fallback.read_node_counts([1, 0]) == [3, 2]


def pytest_streaming_large_dataset_keeps_bounded_state(monkeypatch):
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda *_args, **_kwargs: pytest.fail("dense permutation was allocated"),
    )

    class HugeDataset:
        total_node_count = 100_000_000

        def __init__(self):
            self.requested = 0

        def __len__(self):
            return 100_000_000

        def read_node_counts(self, indices):
            self.requested += len(indices)
            return [1] * len(indices)

    dataset = HugeDataset()
    sampler = StreamingNodeBudgetBatchSampler(
        dataset,
        4,
        steps_per_epoch=2,
        metadata_chunk_size=4,
        metadata_cache_size=8,
    )
    assert len(list(sampler)) == 2
    assert len(sampler._pending) <= sampler.metadata_chunk_size
    assert len(sampler.provider._cache) <= 8
    assert dataset.requested <= 2 * sampler.metadata_chunk_size


def pytest_streaming_respects_limits_and_oversized_policies():
    dataset = _MetadataDataset([6, 4, 3, 12])
    sampler = StreamingNodeBudgetBatchSampler(
        dataset,
        10,
        target_nodes=8,
        steps_per_epoch=2,
        metadata_chunk_size=2,
        forward_window=1,
        max_graphs=2,
        shuffle=False,
        oversized_sample="skip",
    )
    batches = list(sampler)
    assert all(sum(dataset.counts[index] for index in batch) <= 10 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)

    with pytest.raises(ValueError, match="exceeding max_nodes"):
        list(
            StreamingNodeBudgetBatchSampler(
                _MetadataDataset([12]),
                10,
                steps_per_epoch=1,
                shuffle=False,
                oversized_sample="error",
            )
        )

    swap_sampler = StreamingNodeBudgetBatchSampler(
        _MetadataDataset([6, 3, 2]),
        10,
        target_nodes=8,
        steps_per_epoch=1,
        metadata_chunk_size=2,
        forward_window=1,
        shuffle=False,
    )
    assert list(swap_sampler) == [[0, 2]]

    single = StreamingNodeBudgetBatchSampler(
        _MetadataDataset([12]),
        10,
        steps_per_epoch=1,
        shuffle=False,
        oversized_sample="single",
    )
    assert list(single) == [[0]]

    with pytest.raises(RuntimeError, match="could not emit a valid batch"):
        list(
            StreamingNodeBudgetBatchSampler(
                _MetadataDataset([12, 13]),
                10,
                steps_per_epoch=1,
                shuffle=False,
                oversized_sample="skip",
            )
        )


def pytest_streaming_dataloader_continues_and_evaluation_stays_finite():
    train = _graphs([2, 3, 4, 1, 2, 3])
    evaluation = _graphs([2, 3, 4, 1])
    train_loader, val_loader, test_loader = create_dataloaders(
        train,
        evaluation,
        evaluation,
        batch_size=99,
        train_sampler_shuffle=False,
        val_sampler_shuffle=False,
        test_sampler_shuffle=False,
        batching={
            "mode": "streaming_node_budget",
            "max_nodes": 5,
            "steps_per_epoch": 4,
            "metadata_chunk_size": 2,
        },
    )
    first = [batch.num_nodes for batch in train_loader]
    second = [batch.num_nodes for batch in train_loader]
    assert len(first) == len(second) == 4
    assert all(nodes <= 5 for nodes in first + second)
    assert len(list(val_loader)) == len(list(test_loader)) == 2


def pytest_streaming_custom_loader_prefetch_is_bounded(monkeypatch):
    monkeypatch.setattr(
        HydraDataLoader, "worker_init", staticmethod(lambda _counter: 0)
    )
    dataset = _graphs([1] * 20)
    sampler = StreamingNodeBudgetBatchSampler(
        dataset, 2, steps_per_epoch=8, shuffle=False
    )
    loader = HydraDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        prefetch_batches=3,
    )
    assert len(list(loader)) == 8
    assert loader.max_prefetch_depth <= 3


def pytest_streaming_standard_multiworker_loader_is_finite():
    dataset = _graphs([1] * 20)
    sampler = StreamingNodeBudgetBatchSampler(
        dataset, 2, steps_per_epoch=8, shuffle=False
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    assert len(list(loader)) == 8


def pytest_streaming_early_break_has_bounded_custom_prefetch(monkeypatch):
    monkeypatch.setattr(HydraDataLoader, "worker_init", staticmethod(lambda _: 0))
    dataset = _graphs([1] * 40)
    sampler = StreamingNodeBudgetBatchSampler(
        dataset, 2, steps_per_epoch=10, shuffle=False
    )
    loader = HydraDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        prefetch_batches=3,
    )
    iterator = iter(loader)
    next(iterator)
    yielded_ahead = sampler.statistics().emitted_steps - 1
    loader._shutdown_executor()
    assert 0 <= yielded_ahead <= loader.prefetch_batches


def pytest_streaming_loader_rejects_misleading_options():
    dataset = _graphs([1, 1])
    config = {
        "mode": "streaming_node_budget",
        "max_nodes": 2,
        "steps_per_epoch": 1,
        "drop_last": True,
    }
    with pytest.raises(ValueError, match="does not accept drop_last"):
        create_dataloaders(dataset, dataset, dataset, 8, batching=config)

    config.pop("drop_last")
    with pytest.raises(ValueError, match="oversampling"):
        create_dataloaders(
            dataset, dataset, dataset, 8, batching=config, oversampling=True
        )


def pytest_streaming_adios_stored_total_and_indexed_counts(tmp_path):
    pytest.importorskip("adios2")
    from mpi4py import MPI

    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, AdiosWriter

    path = str(tmp_path / "streaming-counts.bp")
    writer = AdiosWriter(path, MPI.COMM_SELF)
    writer.add("trainset", _graphs([2, 5, 3]))
    writer.save()

    dataset = AdiosDataset(path, "trainset", MPI.COMM_SELF)
    try:
        assert dataset.get_total_node_count() == 10
        assert dataset.read_node_counts([2, 0]) == [3, 2]
        sampler = StreamingNodeBudgetBatchSampler(
            dataset,
            5,
            target_nodes=5,
            shuffle=False,
        )
        assert len(sampler) == 2
    finally:
        dataset.close_node_count_reader()
        assert dataset._node_count_reader is None
        dataset.f.close()


def _gloo_streaming_worker(rank, world_size, init_file, result_queue):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    counts = [5] * 192 + [10] * 16 + [5] * 176
    sampler = StreamingNodeBudgetBatchSampler(
        _MetadataDataset(counts),
        10,
        steps_per_epoch=100,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        metadata_chunk_size=8,
    )
    traversals = []
    for _batch in sampler:
        traversals.append(sampler.last_batch_traversal)
    result_queue.put((rank, len(traversals), traversals.count(0)))
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def pytest_streaming_two_rank_gloo_has_equal_fixed_steps(tmp_path):
    world_size = 2
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_gloo_streaming_worker,
            args=(rank, world_size, init_file, queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert sorted(queue.get() for _ in range(world_size)) == [
        (0, 100, 96),
        (1, 100, 100),
    ]


@pytest.mark.gpu
def pytest_streaming_two_rank_nccl_optimizer_smoke():
    from mpi4py import MPI
    from torch.nn.parallel import DistributedDataParallel

    from hydragnn.utils.distributed import setup_ddp

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        pytest.skip("run this smoke test under two MPI ranks")
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for the NCCL smoke test")
    assert dist.is_nccl_available()

    world_size, rank = setup_ddp()
    assert world_size == 2
    device_index = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    counts = [5, 1, 1, 1, 1, 2, 3, 4, 1, 2] * 4
    dataset = _graphs(counts)
    sampler = StreamingNodeBudgetBatchSampler(
        dataset,
        5,
        target_nodes=5,
        steps_per_epoch=7,
        num_replicas=world_size,
        rank=rank,
        max_graphs=4,
        seed=29,
        metadata_chunk_size=4,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    model = DistributedDataParallel(torch.nn.Linear(1, 1).to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    local_graph_counts = []
    optimizer_steps = 0

    try:
        for epoch in range(2):
            sampler.set_epoch(epoch)
            for batch in loader:
                assert batch.num_nodes <= 5
                local_graph_counts.append(batch.num_graphs)
                value = torch.tensor(
                    [[float(batch.num_nodes)]], dtype=torch.float32, device=device
                )
                optimizer.zero_grad()
                model(value).square().mean().backward()
                optimizer.step()
                optimizer_steps += 1

        step_tensor = torch.tensor(optimizer_steps, device=device)
        minimum = step_tensor.clone()
        maximum = step_tensor.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert minimum.item() == maximum.item() == 14
        all_graph_counts = comm.allgather(local_graph_counts)
        assert len({value for values in all_graph_counts for value in values}) > 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
