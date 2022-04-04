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

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.utils import degree
from .print_utils import print_master

from hydragnn.utils.distributed import (
    get_comm_size_and_rank,
    get_device,
    is_model_distributed,
)
from collections import OrderedDict


def get_model_or_module(model):
    if is_model_distributed(model):
        return model.module
    else:
        return model


def save_model(model, optimizer, name, path="./logs/"):
    """Save both model and optimizer state in a single checkpoint file"""
    _, world_rank = get_comm_size_and_rank()
    if world_rank == 0:
        model = get_model_or_module(model)
        path_name = os.path.join(path, name, name + ".pk")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path_name,
        )


def get_summary_writer(name, path="./logs/"):
    _, world_rank = get_comm_size_and_rank()
    if world_rank == 0:
        path_name = os.path.join(path, name)
        writer = SummaryWriter(path_name)


def load_existing_model_config(model, config, path="./logs/", optimizer=None):
    if "continue" in config and config["continue"]:
        model_name = config["startfrom"]
        load_existing_model(model, model_name, path, optimizer)


def load_existing_model(model, model_name, path="./logs/", optimizer=None):
    """Load both model and optimizer state from a single checkpoint file"""
    _, world_rank = get_comm_size_and_rank()
    path_name = os.path.join(path, model_name, model_name + ".pk")
    map_location = {"cuda:%d" % 0: "cuda:%d" % world_rank}
    checkpoint = torch.load(path_name, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]
    if is_model_distributed(model):
        ddp_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = "module." + k
            ddp_state_dict[k] = v
        state_dict = ddp_state_dict
    model.load_state_dict(state_dict)
    if (optimizer is not None) and ("optimizer_state_dict" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def calculate_PNA_degree(dataset: [Data], max_neighbours):
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long).to(get_device())
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def print_model(model):
    """print model's parameter size layer by layer"""
    num_params = 0
    for k, v in model.state_dict().items():
        print_master("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
        num_params += v.numel()
    print_master("-" * 50)
    print_master("%50s\t%20s\t%10d" % ("Total", "", num_params))
    print_master("All (total, MB): %d %g" % (num_params, num_params * 4 / 1024 / 1024))
