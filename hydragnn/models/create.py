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

from hydragnn.models.GINStack import GINStack
from hydragnn.models.PNAStack import PNAStack
from hydragnn.models.GATStack import GATStack
from hydragnn.models.MFCStack import MFCStack
from hydragnn.models.CGCNNStack import CGCNNStack

from hydragnn.utils.distributed import get_device
from hydragnn.utils.print_utils import print_distributed
from hydragnn.utils.time_utils import Timer


def create_model_config(
    config: dict,
    max_neighbours: int = None,
    num_nodes: int = None,
    pna_deg: torch.tensor = None,
    verbosity: int = 0,
    use_gpu: bool = True,
    use_distributed: bool = False,
):
    edge_dim = None
    edge_models = ["PNA", "CGCNN"]
    if "edge_features" in config and config["edge_features"]:
        assert (
            config["model_type"] in edge_models
        ), "Edge features can only be used with PNA and CGCNN."
        edge_dim = len(config["edge_features"])
    elif config["model_type"] == "CGCNN":
        # CG always needs an integer edge_dim
        # PNA would fail with integer edge_dim without edge_attr
        edge_dim = 0

    return create_model(
        config["model_type"],
        config["input_dim"],
        config["output_dim"],
        config["hidden_dim"],
        config["num_conv_layers"],
        config["output_type"],
        config["output_heads"],
        config["task_weights"],
        config["max_neighbours"],
        num_nodes,
        edge_dim,
        pna_deg,
        verbosity,
        use_gpu,
        use_distributed,
    )


def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_conv_layers: int,
    output_type: str,
    output_heads: dict,
    task_weights: list,
    max_neighbours: int = None,
    num_nodes: int = None,
    edge_dim: int = None,
    pna_deg: torch.tensor = None,
    verbosity: int = 0,
    use_gpu: bool = True,
    use_distributed: bool = False,
):
    timer = Timer("create_model")
    timer.start()
    torch.manual_seed(0)

    device = get_device(use_gpu, verbosity_level=verbosity)

    if model_type == "GIN":
        model = GINStack(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_conv_layers=num_conv_layers,
            output_type=output_type,
            config_heads=output_heads,
            loss_weights=task_weights,
        )

    elif model_type == "PNA":
        assert pna_deg is not None, "PNA requires degree input."
        model = PNAStack(
            deg=pna_deg,
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            output_type=output_type,
            config_heads=output_heads,
            loss_weights=task_weights,
            edge_dim=edge_dim,
        )

    elif model_type == "GAT":
        model = GATStack(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_conv_layers=num_conv_layers,
            output_type=output_type,
            config_heads=output_heads,
            loss_weights=task_weights,
        )

    elif model_type == "MFC":
        assert max_neighbours is not None, "MFC requires max_neighbours input."
        model = MFCStack(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            max_degree=max_neighbours,
            num_conv_layers=num_conv_layers,
            output_type=output_type,
            config_heads=output_heads,
            loss_weights=task_weights,
        )

    elif model_type == "CGCNN":
        model = CGCNNStack(
            input_dim=input_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            num_conv_layers=num_conv_layers,
            output_type=output_type,
            config_heads=output_heads,
            loss_weights=task_weights,
            edge_dim=edge_dim,
        )

    else:
        raise ValueError("Unknown model_type: {0}".format(model_type))

    timer.stop()

    return model.to(device)
