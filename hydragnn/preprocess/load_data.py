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
import pickle


def dataset_loading_and_splitting(config: {}):

    ##check if serialized pickle files or folders for raw files provided
    if not list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
        transform_raw_data_to_serialized(config["Dataset"])

    ##if total dataset is provided, split the dataset and save them to pkl files and update config with pkl file locations
    if "total" in config["Dataset"]["path"].keys():
        total_to_train_val_test_pkls(config)

    trainset, valset, testset = load_train_val_test_sets(config)

    return create_dataloaders(
        trainset,
        valset,
        testset,
        batch_size=config["NeuralNetwork"]["Training"]["batch_size"],
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


def split_dataset(dataset: [], perc_train: float):

    perc_val = (1 - perc_train) / 2
    data_size = len(dataset)
    trainset = dataset[: int(data_size * perc_train)]
    valset = dataset[
        int(data_size * perc_train) : int(data_size * (perc_train + perc_val))
    ]
    testset = dataset[int(data_size * (perc_train + perc_val)) :]

    return trainset, valset, testset


def load_train_val_test_sets(config):

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
        loader = SerializedDataLoader(config["Verbosity"]["level"])
        dataset = loader.load_serialized_data(
            dataset_path=files_dir,
            config=config["NeuralNetwork"],
        )
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
        loader = RawDataLoader(config)
        loader.load_raw_data()

    if dist.is_initialized():
        dist.barrier()


def total_to_train_val_test_pkls(config):
    _, rank = get_comm_size_and_rank()

    if list(config["Dataset"]["path"].values())[0].endswith(".pkl"):
        file_dir = config["Dataset"]["path"]["total"]
    else:
        file_dir = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
    # if "total" raw dataset is provided, generate train/val/test pkl files and update config dict.
    with open(file_dir, "rb") as f:
        minmax_node_feature = pickle.load(f)
        minmax_graph_feature = pickle.load(f)
        dataset_total = pickle.load(f)

    trainset, valset, testset = split_dataset(
        dataset=dataset_total,
        perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
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
        if rank == 0:
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(minmax_node_feature, f)
                pickle.dump(minmax_graph_feature, f)
                pickle.dump(dataset, f)

    if dist.is_initialized():
        dist.barrier()
