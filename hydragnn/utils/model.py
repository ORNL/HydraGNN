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


def save_model(model, name, path="./logs/"):
    _, world_rank = get_comm_size_and_rank()
    if world_rank == 0:
        model = get_model_or_module(model)
        path_name = os.path.join(path, name, name + ".pk")
        torch.save(model.state_dict(), path_name)


def get_summary_writer(name, path="./logs/"):
    _, world_rank = get_comm_size_and_rank()
    if world_rank == 0:
        path_name = os.path.join(path, name)
        writer = SummaryWriter(path_name)


def load_existing_model_config(model, config, path="./logs/"):
    if "continue" in config and config["continue"]:
        model_name = config["startfrom"]
        load_existing_model(model, model_name, path)


def load_existing_model(model, model_name, path="./logs/"):
    path_name = os.path.join(path, model_name, model_name + ".pk")
    state_dict = torch.load(path_name, map_location="cpu")

    if is_model_distributed(model):
        ddp_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = "module." + k
            ddp_state_dict[k] = v
        state_dict = ddp_state_dict

    model.load_state_dict(state_dict)


def calculate_PNA_degree(dataset: [Data], max_neighbours):
    deg = torch.zeros(max_neighbours + 1, dtype=torch.long).to(get_device())
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
