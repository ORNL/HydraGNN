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
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.utils import degree
from .print_utils import print_master

from hydragnn.utils.distributed import (
    get_comm_size_and_rank,
    get_device,
    get_device_name,
)
from collections import OrderedDict


def loss_function_selection(loss_function_string: str):
    if loss_function_string == "mse":
        return torch.nn.functional.mse_loss
    elif loss_function_string == "mae":
        return torch.nn.functional.l1_loss
    elif loss_function_string == "smooth_l1":
        return torch.nn.SmoothL1Loss
    elif loss_function_string == "rmse":
        return lambda x, y: torch.sqrt(torch.nn.functional.mse_loss(x, y))


def save_model(model, optimizer, name, path="./logs/"):
    """Save both model and optimizer state in a single checkpoint file"""
    _, world_rank = get_comm_size_and_rank()
    if world_rank == 0:
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
    path_name = os.path.join(path, model_name, model_name + ".pk")
    map_location = {"cuda:%d" % 0: get_device_name()}
    checkpoint = torch.load(path_name, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]
    ## To be compatible with old checkpoint which was not written as a ddp model
    if not next(iter(state_dict)).startswith("module"):
        ddp_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = "module." + k
            ddp_state_dict[k] = v
        state_dict = ddp_state_dict
    model.load_state_dict(state_dict)
    if (optimizer is not None) and ("optimizer_state_dict" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


## This function may cause OOM if dataset is too large
## to fit in a single GPU (i.e., with DDP). Use with caution.
## Recommend to use calculate_PNA_degree_dist
def calculate_PNA_degree(dataset: [Data], max_neighbours):
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        if dist.get_backend() == "nccl":
            d = d.to(device)
        deg += torch.bincount(d, minlength=deg.numel())

    ## Allreduce from everytone
    dist.all_reduce(deg, op=dist.ReduceOp.SUM)
    return deg


## MPI causing problem with num_workers>0 which using torch.multiprocessing
def calculate_PNA_degree_mpi(loader, max_neighbours):
    assert MPI.Is_initialized()
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long)
    for i, data in enumerate(loader):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    _deg = deg.detach().cpu().numpy()
    _deg = MPI.COMM_WORLD.allreduce(_deg, op=MPI.SUM)
    deg = torch.from_numpy(_deg)
    return deg


def calculate_PNA_degree_dist(loader, max_neighbours):
    assert torch.distributed.is_initialized()
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long).to(get_device())
    for data in loader:
        data = data.to(get_device())
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    ## Allreduce from everytone
    dist.all_reduce(deg, op=dist.ReduceOp.SUM)
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
