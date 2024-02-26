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
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.utils import degree
from .print_utils import print_master, iterate_tqdm

from hydragnn.utils.distributed import (
    get_comm_size_and_rank,
    get_device,
    get_device_name,
)
from collections import OrderedDict


def activation_function_selection(activation_function_string: str):
    if activation_function_string == "relu":
        return torch.nn.ReLU()
    elif activation_function_string == "selu":
        return torch.nn.SELU()
    elif activation_function_string == "prelu":
        return torch.nn.PReLU()
    elif activation_function_string == "elu":
        return torch.nn.ELU()
    elif activation_function_string == "lrelu_01":
        return torch.nn.LeakyReLU(0.1)
    elif activation_function_string == "lrelu_025":
        return torch.nn.LeakyReLU(0.25)
    elif activation_function_string == "lrelu_05":
        return torch.nn.LeakyReLU(0.5)


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
    if hasattr(optimizer, "consolidate_state_dict"):
        optimizer.consolidate_state_dict()
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
    print_master("Load existing model:", path_name)
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
def calculate_PNA_degree(loader, max_neighbours):
    backend = os.getenv("HYDRAGNN_AGGR_BACKEND", "torch")
    if backend == "torch":
        return calculate_PNA_degree_dist(loader, max_neighbours)
    elif backend == "mpi":
        return calculate_PNA_degree_mpi(loader, max_neighbours)
    else:
        deg = torch.zeros(max_neighbours + 1, dtype=torch.long)
        for data in iterate_tqdm(loader, 2, desc="Calculate PNA degree"):
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())[: max_neighbours + 1]
        return deg


def calculate_PNA_degree_dist(loader, max_neighbours):
    assert dist.is_initialized()
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long)
    for data in iterate_tqdm(loader, 2, desc="Calculate PNA degree"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())[: max_neighbours + 1]
    deg = deg.to(get_device())
    dist.all_reduce(deg, op=dist.ReduceOp.SUM)
    deg = deg.detach().cpu()
    return deg


def calculate_PNA_degree_mpi(loader, max_neighbours):
    assert dist.is_initialized()
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long)
    for data in iterate_tqdm(loader, 2, desc="Calculate PNA degree"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())[: max_neighbours + 1]
    from mpi4py import MPI

    deg = MPI.COMM_WORLD.allreduce(deg.numpy(), op=MPI.SUM)
    return torch.tensor(deg)


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def print_model(model):
    """print model's parameter size layer by layer"""
    num_params = 0
    for k, v in model.state_dict().items():
        print_master("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
        num_params += v.numel()
    print_master("-" * 50)
    print_master("%50s\t%20s\t%10d" % ("Total number of model params:", "", num_params))
    print_master("All (total, MB): %d %g" % (num_params, num_params * 4 / 1024 / 1024))


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))


# early stop based on validation loss
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.val_loss_min = float("inf")
        self.count = 0

    def __call__(self, val_loss):
        if val_loss > self.val_loss_min + self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                return True
        else:
            self.val_loss_min = val_loss
            self.count = 0
        return False


class Checkpoint:
    """
    Checkpoints the model and optimizer when:
        + The performance metric is smaller than the stored performance metric
    Args
      warmup: (int) Number of epochs to warmup prior to checkpointing.
      path: (str) Path for checkpointing
      name: (str) Model name for the directory and the file to save.
    """

    def __init__(
        self,
        name: str,
        warmup: int = 0,
        path: str = "./logs/",
    ):
        self.count = 1
        self.warmup = warmup
        self.path = path
        self.name = name
        self.min_perf_metric = float("inf")
        self.min_delta = 0

    def __call__(self, model, optimizer, perf_metric):

        if (perf_metric > self.min_perf_metric + self.min_delta) or (
            self.count < self.warmup
        ):
            self.count += 1
            return False
        else:
            self.min_perf_metric = perf_metric
            save_model(model, optimizer, name=self.name, path=self.path)
            return True
