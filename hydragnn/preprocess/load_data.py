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

import numpy as np
import torch
import torch.distributed as dist

# FIXME: deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from hydragnn.preprocess.serialized_dataset_loader import SerializedDataLoader
from hydragnn.preprocess.raw_dataset_loader import RawDataLoader
from hydragnn.utils.distributed import get_comm_size_and_rank
from hydragnn.utils.time_utils import Timer


def dataset_loading_and_splitting(
    config: {},
    chosen_dataset_option,
):

    dataset_chosen, dataset_names = load_data(chosen_dataset_option, config)
    return split_dataset(
        dataset_list=dataset_chosen,
        dataset_names=dataset_names,
        batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
        perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
    )


def create_dataloaders(trainset, valset, testset, batch_size):

    if dist.is_initialized():

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler
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
    dataset_list: [],
    dataset_names: [],
    batch_size: int,
    perc_train: float,
):

    if len(dataset_names) == 1 and dataset_names[0] == "total":
        dataset = dataset_list[0]
        perc_val = (1 - perc_train) / 2
        data_size = len(dataset)
        trainset = dataset[: int(data_size * perc_train)]
        valset = dataset[
            int(data_size * perc_train) : int(data_size * (perc_train + perc_val))
        ]
        testset = dataset[int(data_size * (perc_train + perc_val)) :]
    elif len(dataset_names) == 3:
        trainset = dataset_list[dataset_names.index("train")]
        valset = dataset_list[dataset_names.index("validate")]
        testset = dataset_list[dataset_names.index("test")]
    else:
        raise ValueError(
            'Must provide "total" OR "train", "test", "validate" data paths: ',
            dataset_names,
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, batch_size
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [],
    dataset2: [],
    batch_size: int,
    perc_train: float,
):

    data_size = len(dataset1)

    trainset = dataset1[: int(data_size * perc_train)]
    valset = dataset1[int(data_size * perc_train) :]
    testset = dataset2

    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, batch_size
    )

    return train_loader, val_loader, test_loader


def load_data(dataset_option, config):

    timer = Timer("load_data")
    timer.start()

    dataset_list = []
    datasetname_list = []

    ##check if serialized pickle files or folders for raw files provided
    pkl_input = False
    if list(config["Dataset"]["path"]["raw"].values())[0].endswith(".pkl"):
        pkl_input = True
    if not pkl_input:
        transform_raw_data_to_serialized(config["Dataset"])
    for dataset_name, raw_data_path in config["Dataset"]["path"]["raw"].items():
        if pkl_input:
            files_dir = raw_data_path
        else:
            if dataset_name == "total":
                files_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}.pkl"
            else:
                files_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}_{dataset_name}.pkl"

        # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
        loader = SerializedDataLoader(config["Verbosity"]["level"])
        dataset = loader.load_serialized_data(
            dataset_path=files_dir,
            config=config["NeuralNetwork"],
        )
        dataset_list.append(dataset)
        datasetname_list.append(dataset_name)

    timer.stop()

    return dataset_list, datasetname_list


def transform_raw_data_to_serialized(config):

    _, rank = get_comm_size_and_rank()

    if rank == 0:
        loader = RawDataLoader(config)
        loader.load_raw_data()

    if dist.is_initialized():
        dist.barrier()
