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
from torch_geometric.data import Data
from torch_geometric.utils import degree

from hydragnn.models.GINStack import GINStack
from hydragnn.models.PNAStack import PNAStack
from hydragnn.models.GATStack import GATStack
from hydragnn.models.MFCStack import MFCStack
from hydragnn.models.CGCNNStack import CGCNNStack

from hydragnn.utils.device import get_device
from hydragnn.utils.print_utils import print_distributed


def create(
    model_type: str,
    input_dim: int,
    dataset: [Data],
    config: dict,
    verbosity_level: int,
    use_gpu: bool = True,
    use_distributed: bool = False,
):
    torch.manual_seed(0)

    _, device = get_device(use_gpu, verbosity_level=verbosity_level)

    num_atoms = dataset[0].num_nodes  # FIXME: assumes constant number of atoms

    if model_type == "GIN":
        model = GINStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            num_nodes=num_atoms,
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)

    elif model_type == "PNA":
        deg = torch.zeros(config["max_neighbours"] + 1, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNAStack(
            deg=deg,
            input_dim=input_dim,
            output_dim=config["output_dim"],
            num_nodes=num_atoms,
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)

    elif model_type == "GAT":
        # heads = int(input("Enter the number of multi-head-attentions(default 1): "))
        # negative_slope = float(
        #     input("Enter LeakyReLU angle of the negative slope(default 0.2): ")
        # )
        # dropout = float(
        #     input(
        #         "Enter dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training(default 0): "
        #     )
        # )

        model = GATStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            num_nodes=num_atoms,
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)

    elif model_type == "MFC":
        model = MFCStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            num_nodes=num_atoms,
            hidden_dim=config["hidden_dim"],
            max_degree=config["max_neighbours"],
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)
    elif model_type == "CGCNN":
        model = CGCNNStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            num_nodes=num_atoms,
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)
    else:
        raise ValueError("Unknown model_type: {0}".format(model_type))

    return model
