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

import os
import socket
from collections import deque

import random

import torch
import torch.distributed as dist
from mpi4py import MPI

# FIXME: deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from hydragnn.preprocess.graph_dataset import (
    load_and_prepare_graph_dataset,
)
from hydragnn.preprocess.batch_sampler import (
    CostAwareBatchSampler,
    DistributedCostAwareBatchSampler,
    StreamingNodeBudgetBatchSampler,
)
from hydragnn.preprocess.lsms_raw_dataset_loader import LSMS_RawDataLoader
from hydragnn.preprocess.cfg_raw_dataset_loader import CFG_RawDataLoader
from hydragnn.utils.datasets.compositional_data_splitting import (
    compositional_stratified_splitting,
)
from hydragnn.utils.distributed import get_comm_size_and_rank
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
import pickle

from hydragnn.utils.print.print_utils import log

from torch_geometric.data import Batch
from torch.utils.data.dataloader import _DatasetKind

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import re


def parse_omp_places(envstr):
    """
    Parse OMP_PLACES env string to get list of places
    Usage example:
        parse_omp_places(os.environ["OMP_PLACES"])
    Input examples:
        "{0:4},{4:4},{8:4},{12:4},{16:4},{20:4},{24:4}"
    """
    plist = list()
    for block in re.findall(r"({[\d,:]+})", envstr):
        start, cnt = list(map(int, re.findall(r"\d+", block)))
        for i in range(start, start + cnt):
            plist.append(i)
    return plist


class SimpleDataLoader(DataLoader):
    """
    A naive implementation of a custom dataloader
    """

    def __init__(self, dataset, **kwargs):
        super(HydraDataLoader, self).__init__(dataset, **kwargs)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self.dataset,
            self._auto_collation,
            self.collate_fn,
            self.drop_last,
        )

        log("num_workers:", self.num_workers)
        log("len:", len(self._index_sampler))

    def __iter__(self):
        self._num_yielded = 0
        self._sampler_iter = iter(self._index_sampler)
        return self

    def __next__(self):
        self._num_yielded += 1
        index = next(self._sampler_iter)
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


class HydraDataLoader(DataLoader):
    """
    A custom data loader with multi-threading on a HPC system.
    This is to overcome a few problems (affinity, hanging, crashing, etc)
    with Pytorch's multi-threaded DataLoader on Summit and Perlmutter.
    (2022/08) jyc: This is a work-in-progress version. Performance is not verified.
    """

    def __init__(self, dataset, **kwargs):
        self.prefetch_batches = int(
            kwargs.pop(
                "prefetch_batches",
                max(1, 2 * int(kwargs.get("num_workers", 0))),
            )
        )
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        super(HydraDataLoader, self).__init__(dataset, **kwargs)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self.dataset,
            self._auto_collation,
            self.collate_fn,
            self.drop_last,
        )

        self._futures = deque()
        self._executor = None
        self._submit_index = 0
        self.max_prefetch_depth = 0

        log("num_workers:", self.num_workers)
        log("len:", len(self._index_sampler))

    @staticmethod
    def worker_init(counter):
        core_width = 2
        if os.getenv("HYDRAGNN_AFFINITY_WIDTH") is not None:
            core_width = int(os.environ["HYDRAGNN_AFFINITY_WIDTH"])

        core_offset = 0
        if os.getenv("HYDRAGNN_AFFINITY_OFFSET") is not None:
            core_offset = int(os.environ["HYDRAGNN_AFFINITY_OFFSET"])

        with counter.get_lock():
            wid = counter.value
            counter.value += 1

        affinity = None
        if hasattr(os, "sched_getaffinity"):
            affinity_check = os.getenv("HYDRAGNN_AFFINITY")
            if affinity_check == "OMP":
                affinity = parse_omp_places(os.getenv("OMP_PLACES"))
            else:
                affinity = list(os.sched_getaffinity(0))

            affinity_mask = set(
                affinity[
                    core_width * wid
                    + core_offset : core_width * (wid + 1)
                    + core_offset
                ]
            )
            os.sched_setaffinity(0, affinity_mask)
            affinity = os.sched_getaffinity(0)

        hostname = socket.gethostname()
        log(
            f"Worker: pid={os.getpid()} hostname={hostname} ID={wid} affinity={affinity}"
        )
        return 0

    @staticmethod
    def fetch(dataset, ibatch, index, pin_memory=False):
        batch = [dataset[i] for i in index]
        # hostname = socket.gethostname()
        # log (f"Worker done: pid={os.getpid()} hostname={hostname} ibatch={ibatch}")
        data = Batch.from_data_list(batch)
        if pin_memory:
            if torch.cuda.is_available():
                data = torch.utils.data._utils.pin_memory.pin_memory(data)
        return (ibatch, data)

    def __iter__(self):
        log("Iterator reset")
        self._shutdown_executor()
        self._num_yielded = 0
        self._sampler_iter = iter(self._index_sampler)
        counter = mp.Value("i", 0)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.num_workers),
            initializer=self.worker_init,
            initargs=(counter,),
        )
        self._submit_index = 0
        self._futures.clear()
        for _ in range(self.prefetch_batches):
            if not self._submit_one():
                break
        return self

    def _submit_one(self):
        try:
            index = next(self._sampler_iter)
        except StopIteration:
            return False
        future = self._executor.submit(
            self.fetch,
            self.dataset,
            self._submit_index,
            index,
            pin_memory=self.pin_memory,
        )
        self._submit_index += 1
        self._futures.append(future)
        self.max_prefetch_depth = max(self.max_prefetch_depth, len(self._futures))
        return True

    def _shutdown_executor(self):
        while self._futures:
            self._futures.popleft().cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __next__(self):
        if not self._futures:
            self._shutdown_executor()
            raise StopIteration
        future = self._futures.popleft()
        try:
            _, data = future.result()
        except BaseException:
            self._shutdown_executor()
            raise
        self._num_yielded += 1
        self._submit_one()
        if not self._futures and self._submit_index == self._num_yielded:
            self._shutdown_executor()
        return data

    def __del__(self):
        try:
            self._shutdown_executor()
        except Exception:
            pass


def dataset_loading_and_splitting(config: {}):
    ##check if serialized pickle files or folders for raw files provided
    if not list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
        transform_raw_data_to_serialized(config["Dataset"])

    ##if total datasets is provided, split the datasets and save them to pkl files and update config with pkl file locations
    if "total" in config["Dataset"]["path"].keys():
        total_to_train_val_test_pkls(config)

    trainset, valset, testset = load_train_val_test_sets(config)

    return create_dataloaders(
        trainset,
        valset,
        testset,
        batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
        batching=config["NeuralNetwork"]["Training"].get("Batching"),
    )


def create_dataloaders(
    trainset,
    valset,
    testset,
    batch_size,
    train_sampler_shuffle=True,
    val_sampler_shuffle=True,
    test_sampler_shuffle=True,
    group=None,
    oversampling=False,
    num_samples=None,  ## tuple of number of samples (train, val, test)
    batching=None,
):
    if batching and batching.get("mode", "fixed") == "streaming_node_budget":
        if "max_nodes" not in batching:
            raise ValueError("streaming_node_budget batching requires max_nodes")
        if oversampling:
            raise ValueError(
                "streaming node-budget batching cannot be combined with oversampling"
            )
        if "drop_last" in batching:
            raise ValueError("streaming_node_budget does not accept drop_last")
        if int(batching.get("prefetch_batches", 1)) <= 0:
            raise ValueError("prefetch_batches must be positive")

        group_size = 1
        group_rank = 0
        if dist.is_initialized():
            if group is None:
                group = dist.group.WORLD
            if isinstance(group, dist.ProcessGroup):
                group_size = dist.get_world_size(group=group)
                group_rank = dist.get_rank(group=group)
            elif isinstance(group, MPI.Comm):
                group_size = group.Get_size()
                group_rank = group.Get_rank()
            else:
                raise ValueError("Unsupported group type for distributed sampling")

        num_workers = int(os.getenv("HYDRAGNN_NUM_WORKERS", "0"))
        use_custom_loader = dist.is_initialized() and int(
            os.getenv("HYDRAGNN_CUSTOM_DATALOADER", "0")
        )

        train_sampler = StreamingNodeBudgetBatchSampler(
            trainset,
            max_nodes=batching["max_nodes"],
            target_nodes=batching.get("target_nodes"),
            steps_per_epoch=batching.get("steps_per_epoch"),
            num_replicas=group_size,
            rank=group_rank,
            max_graphs=batching.get("max_graphs"),
            metadata_chunk_size=batching.get("metadata_chunk_size", 32),
            metadata_cache_size=batching.get("metadata_cache_size"),
            forward_window=batching.get("forward_window", 1),
            shuffle=batching.get("shuffle", train_sampler_shuffle),
            seed=batching.get("seed", 0),
            oversized_sample=batching.get("oversized_sample", "error"),
        )

        train_loader_type = HydraDataLoader if use_custom_loader else DataLoader
        train_loader_kwargs = {
            "batch_sampler": train_sampler,
            "num_workers": num_workers,
            "pin_memory": dist.is_initialized(),
            "persistent_workers": False,
        }
        if use_custom_loader:
            train_loader_kwargs["prefetch_batches"] = batching.get(
                "prefetch_batches", max(1, 2 * num_workers)
            )
        train_loader = train_loader_type(trainset, **train_loader_kwargs)

        finite_sampler_type = CostAwareBatchSampler
        finite_distributed_options = {}
        if dist.is_initialized():
            finite_sampler_type = DistributedCostAwareBatchSampler
            finite_distributed_options = {
                "num_replicas": group_size,
                "rank": group_rank,
                "pad_batches": True,
            }

        def make_finite_loader(dataset, shuffle):
            sampler = finite_sampler_type(
                dataset,
                max_cost=batching["max_nodes"],
                max_graphs=batching.get("max_graphs"),
                shuffle=shuffle,
                seed=batching.get("seed", 0),
                oversized_sample=batching.get("oversized_sample", "error"),
                **finite_distributed_options,
            )
            return DataLoader(dataset, batch_sampler=sampler)

        return (
            train_loader,
            make_finite_loader(valset, val_sampler_shuffle),
            make_finite_loader(testset, test_sampler_shuffle),
        )

    if batching and batching.get("mode", "fixed") != "fixed":
        if oversampling:
            raise ValueError("cost-aware batching cannot be combined with oversampling")
        if batching["mode"] != "node_budget":
            raise ValueError(f"unsupported batching mode: {batching['mode']}")

        sampler_type = CostAwareBatchSampler
        distributed_options = {}
        if dist.is_initialized():
            if group is None:
                group = dist.group.WORLD
            if isinstance(group, dist.ProcessGroup):
                group_size = dist.get_world_size(group=group)
                group_rank = dist.get_rank(group=group)
            elif isinstance(group, MPI.Comm):
                group_size = group.Get_size()
                group_rank = group.Get_rank()
            else:
                raise ValueError("Unsupported group type for distributed sampling")
            sampler_type = DistributedCostAwareBatchSampler
            distributed_options = {
                "num_replicas": group_size,
                "rank": group_rank,
                "pad_batches": not batching.get("drop_last", False),
            }

        def make_loader(dataset, shuffle):
            sampler_drop_last = (
                batching.get("drop_last", False)
                if sampler_type is CostAwareBatchSampler
                else False
            )
            batch_sampler = sampler_type(
                dataset,
                max_cost=batching["max_nodes"],
                max_graphs=batching.get("max_graphs"),
                shuffle=batching.get("shuffle", shuffle),
                seed=batching.get("seed", 0),
                oversized_sample=batching.get("oversized_sample", "error"),
                drop_last=sampler_drop_last,
                **distributed_options,
            )
            loader_type = DataLoader
            if dist.is_initialized() and int(
                os.getenv("HYDRAGNN_CUSTOM_DATALOADER", "0")
            ):
                loader_type = HydraDataLoader
            return loader_type(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=int(os.getenv("HYDRAGNN_NUM_WORKERS", "0")),
                pin_memory=dist.is_initialized(),
                persistent_workers=False,
            )

        return (
            make_loader(trainset, train_sampler_shuffle),
            make_loader(valset, val_sampler_shuffle),
            make_loader(testset, test_sampler_shuffle),
        )

    if dist.is_initialized():
        if oversampling:
            assert num_samples is not None
            train_sampler = torch.utils.data.RandomSampler(
                trainset, replacement=False, num_samples=num_samples[0]
            )
            val_sampler = torch.utils.data.RandomSampler(
                valset, replacement=False, num_samples=num_samples[1]
            )
            test_sampler = torch.utils.data.RandomSampler(
                testset, replacement=False, num_samples=num_samples[2]
            )
        else:

            if group is None:
                group = dist.group.WORLD

            if isinstance(group, dist.ProcessGroup):
                group_size = dist.get_world_size(group=group)
                group_rank = dist.get_rank(group=group)
            elif isinstance(group, MPI.Comm):
                group_size = group.Get_size()
                group_rank = group.Get_rank()
            else:
                raise ValueError("Unsupported group type for distributed sampling")

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset,
                num_replicas=group_size,
                rank=group_rank,
                shuffle=train_sampler_shuffle,
            )

            val_sampler = torch.utils.data.distributed.DistributedSampler(
                valset,
                num_replicas=group_size,
                rank=group_rank,
                shuffle=val_sampler_shuffle,
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset,
                num_replicas=group_size,
                rank=group_rank,
                shuffle=test_sampler_shuffle,
            )

        pin_memory = True
        persistent_workers = False
        num_workers = 0
        if os.getenv("HYDRAGNN_NUM_WORKERS") is not None:
            num_workers = int(os.environ["HYDRAGNN_NUM_WORKERS"])

        use_custom_dataloader = 0
        if os.getenv("HYDRAGNN_CUSTOM_DATALOADER") is not None:
            use_custom_dataloader = int(os.environ["HYDRAGNN_CUSTOM_DATALOADER"])

        if use_custom_dataloader == 1:
            train_loader = HydraDataLoader(
                trainset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        else:
            train_loader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        val_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False, sampler=val_sampler
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=test_sampler
        )

    else:

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True,
        )

    return train_loader, val_loader, test_loader


def split_dataset(
    dataset: [],
    perc_train: float,
    stratify_splitting: bool,
):
    if not stratify_splitting:
        perc_val = (1 - perc_train) / 2
        dataset = list(dataset)
        data_size = len(dataset)
        random.shuffle(dataset)
        trainset = dataset[: int(data_size * perc_train)]
        valset = dataset[
            int(data_size * perc_train) : int(data_size * (perc_train + perc_val))
        ]
        testset = dataset[int(data_size * (perc_train + perc_val)) :]
    else:
        trainset, valset, testset = compositional_stratified_splitting(
            dataset, perc_train
        )

    return trainset, valset, testset


def load_train_val_test_sets(config, isdist=False):
    timer = Timer("load_data")
    timer.start()

    dataset_list = []
    datasetname_list = []

    for dataset_name, raw_data_path in config["Dataset"]["path"].items():
        if raw_data_path.endswith(".pkl"):
            files_dir = raw_data_path
        else:
            serialized_data_path = os.environ.get("SERIALIZED_DATA_PATH", os.getcwd())
            files_dir = f"{serialized_data_path}/serialized_dataset/{config['Dataset']['name']}_{dataset_name}.pkl"
        # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
        dataset = load_and_prepare_graph_dataset(files_dir, config, dist=isdist)

        dataset_list.append(dataset)
        datasetname_list.append(dataset_name)

    trainset = dataset_list[datasetname_list.index("train")]
    valset = dataset_list[datasetname_list.index("validate")]
    testset = dataset_list[datasetname_list.index("test")]

    timer.stop()

    return trainset, valset, testset


def transform_raw_data_to_serialized(config):
    _, rank = get_comm_size_and_rank()

    if rank == 0:
        if config["format"] == "LSMS" or config["format"] == "unit_test":
            loader = LSMS_RawDataLoader(config)
        elif config["format"] == "CFG":
            loader = CFG_RawDataLoader(config)
        else:
            raise NameError("Data format not recognized for raw data loader")

        loader.load_raw_data()

    if dist.is_initialized():
        dist.barrier()


def total_to_train_val_test_pkls(config, isdist=False):
    _, rank = get_comm_size_and_rank()

    if list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
        file_dir = config["Dataset"]["path"]["total"]
    else:
        file_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
    # if "total" raw datasets is provided, generate train/val/test pkl files and update config dict.
    with open(file_dir, "rb") as f:
        minmax_node_feature = pickle.load(f)
        minmax_graph_feature = pickle.load(f)
        dataset_total = pickle.load(f)

    trainset, valset, testset = split_dataset(
        dataset=dataset_total,
        perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
        stratify_splitting=config["Dataset"]["compositional_stratified_splitting"],
    )
    serialized_dir = os.path.dirname(file_dir)
    config["Dataset"]["path"] = {}
    for dataset_type, dataset in zip(
        ["train", "validate", "test"], [trainset, valset, testset]
    ):
        serial_data_name = config["Dataset"]["name"] + "_" + dataset_type + ".pkl"
        config["Dataset"]["path"][dataset_type] = (
            serialized_dir + "/" + serial_data_name
        )
        if (not isdist) and (rank == 0):
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(minmax_node_feature, f)
                pickle.dump(minmax_graph_feature, f)
                pickle.dump(dataset, f)
        elif isdist:
            ## This is for the ising example.
            ## Each process writes own pickle data. config["Dataset"]["name"] contains rank info.
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(minmax_node_feature, f)
                pickle.dump(minmax_graph_feature, f)
                pickle.dump(dataset, f)

    if dist.is_initialized():
        dist.barrier()
