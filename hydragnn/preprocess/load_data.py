##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
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

import random

import torch
import torch.distributed as dist

# FIXME: deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from hydragnn.preprocess.serialized_dataset_loader import SerializedDataLoader
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
import queue
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
        super(HydraDataLoader, self).__init__(dataset, **kwargs)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self.dataset,
            self._auto_collation,
            self.collate_fn,
            self.drop_last,
        )

        ## List of threads job (futures)
        self.fs = queue.Queue()

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
            data = torch.utils.data._utils.pin_memory.pin_memory(data)
        return (ibatch, data)

    def __iter__(self):
        log("Iterator reset")
        ## Check previous futures
        if self.fs.qsize() > 0:
            log("Clearn previous futures:", self.fs.qsize())
            for future in iter(self.fs.get, None):
                future.cancel()

        ## Resetting
        self._num_yielded = 0
        self._sampler_iter = iter(self._index_sampler)
        self.fs_iter = iter(self.fs.get, None)
        counter = mp.Value("i", 0)
        executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            initializer=self.worker_init,
            initargs=(counter,),
        )
        for i in range(len(self._index_sampler)):
            index = next(self._sampler_iter)
            future = executor.submit(
                self.fetch,
                self.dataset,
                i,
                index,
                pin_memory=self.pin_memory,
            )
            self.fs.put(future)
        self.fs.put(None)
        # log ("Submit all done.")
        return self

    def __next__(self):
        # log ("Getting next", self._num_yielded)
        future = next(self.fs_iter)
        ibatch, data = future.result()
        # log (f"Future done: ibatch={ibatch}", data.num_graphs)
        self._num_yielded += 1
        return data


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
    )


def create_dataloaders(
    trainset,
    valset,
    testset,
    batch_size,
    train_sampler_shuffle=True,
    val_sampler_shuffle=True,
    test_sampler_shuffle=True,
):
    if dist.is_initialized():

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, shuffle=train_sampler_shuffle
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset, shuffle=val_sampler_shuffle
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset, shuffle=test_sampler_shuffle
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
            files_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}_{dataset_name}.pkl"
        # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
        loader = SerializedDataLoader(config, dist=isdist)
        dataset = loader.load_serialized_data(dataset_path=files_dir)

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
