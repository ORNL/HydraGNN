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

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
from .Base import Base


class PNAStack(Base):
    def __init__(
        self,
        deg: torch.Tensor,
        input_dim: int,
        output_dim: list,
        output_type: list,
        num_nodes: int,
        hidden_dim: int,
        config_heads: {},
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        ilossweights_hyperp: int = 1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        loss_weights: list = [1.0, 1.0, 1.0],  # weights for losses of different tasks
        ilossweights_nll: int = 0,  # if =1, using the scalar uncertainty as weights, as in paper
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.input_dim = input_dim
        self.head_dims = output_dim
        self.head_type = output_type
        self.config_heads = config_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            PNAConv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )
        self.batch_norms.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            conv = PNAConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))
        self.__conv_node_features__(aggregators, scalers, deg)
        super()._multihead(
            num_nodes, ilossweights_hyperp, loss_weights, ilossweights_nll
        )

    def __conv_node_features__(self, aggregators, scalers, deg):
        # convolutional layers for node level predictions
        # two ways to implement node features from here:
        # 1. one graph for all node features
        # 2. one graph for one node features (currently implemented)
        self.convs_node_hidden = ModuleList()
        self.batch_norms_node_hidden = ModuleList()
        self.convs_node_output = ModuleList()
        self.batch_norms_node_output = ModuleList()

        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return

        self.num_conv_layers_node = self.config_heads["node"]["num_headlayers"]
        self.hidden_dim_node = self.config_heads["node"]["dim_headlayers"]
        # In this part, each head has same number of convolutional layers, but can have different output dimension
        self.convs_node_hidden.append(
            PNAConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim_node[0],
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )
        self.batch_norms_node_hidden.append(BatchNorm(self.hidden_dim_node[0]))
        for ilayer in range(self.num_conv_layers_node - 1):
            self.convs_node_hidden.append(
                PNAConv(
                    in_channels=self.hidden_dim_node[ilayer],
                    out_channels=self.hidden_dim_node[ilayer + 1],
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            )
            self.batch_norms_node_hidden.append(
                BatchNorm(self.hidden_dim_node[ilayer + 1])
            )

        for ihead in node_feature_ind:
            self.convs_node_output.append(
                PNAConv(
                    in_channels=self.hidden_dim_node[-1],
                    out_channels=self.head_dims[ihead],
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            )
            self.batch_norms_node_output.append(BatchNorm(self.head_dims[ihead]))

    def __str__(self):
        return "PNAStack"
