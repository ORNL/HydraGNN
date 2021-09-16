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
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import GATv2Conv, BatchNorm

from .Base import Base


class GATStack(Base):
    def __init__(
        self,
        input_dim: int,
        output_dim: list,
        output_type: list,
        num_nodes: int,
        hidden_dim: int,
        config_heads: {},
        heads: int = 6,
        negative_slope: float = 0.05,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        ilossweights_hyperp: int = 1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        loss_weights: list = [1.0, 1.0, 1.0],  # weights for losses of different tasks
        ilossweights_nll: int = 0,  # if =1, using the scalar uncertainty as weights, as in paper# https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    ):
        super().__init__()

        self.input_dim = input_dim
        self.head_dims = output_dim
        self.head_type = output_type
        self.config_heads = config_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        # note that self.heads is a parameter in GATConv, not the num_heads in the output part
        self.heads = heads
        self.negative_slope = negative_slope
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                heads=self.heads,
                negative_slope=self.negative_slope,
                dropout=self.dropout,
                add_self_loops=True,
            )
        )
        self.batch_norms.append(BatchNorm(self.hidden_dim * self.heads))
        for _ in range(self.num_conv_layers - 2):
            conv = GATv2Conv(
                in_channels=self.hidden_dim * self.heads,
                out_channels=self.hidden_dim,
                heads=self.heads,
                negative_slope=self.negative_slope,
                dropout=self.dropout,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim * self.heads))
        conv = GATv2Conv(
            in_channels=self.hidden_dim * self.heads,
            out_channels=self.hidden_dim,
            heads=self.heads,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=True,
            concat=False,
        )
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(self.hidden_dim))
        self.__conv_node_features__()
        super()._multihead(
            num_nodes, ilossweights_hyperp, loss_weights, ilossweights_nll
        )

    def __conv_node_features__(self):
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
            GATv2Conv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim_node[0],
                heads=self.heads,
                negative_slope=self.negative_slope,
                dropout=self.dropout,
                add_self_loops=True,
            )
        )
        self.batch_norms_node_hidden.append(
            BatchNorm(self.hidden_dim_node[0] * self.heads)
        )
        for ilayer in range(self.num_conv_layers_node - 1):
            self.convs_node_hidden.append(
                GATv2Conv(
                    in_channels=self.hidden_dim_node[ilayer] * self.heads,
                    out_channels=self.hidden_dim_node[ilayer + 1],
                    heads=self.heads,
                    negative_slope=self.negative_slope,
                    dropout=self.dropout,
                    add_self_loops=True,
                )
            )
            self.batch_norms_node_hidden.append(
                BatchNorm(self.hidden_dim_node[ilayer + 1] * self.heads)
            )

        for ihead in node_feature_ind:
            self.convs_node_output.append(
                GATv2Conv(
                    in_channels=self.hidden_dim_node[-1] * self.heads,
                    out_channels=self.head_dims[ihead],
                    heads=self.heads,
                    negative_slope=self.negative_slope,
                    dropout=self.dropout,
                    add_self_loops=True,
                    concat=False,
                )
            )
            self.batch_norms_node_output.append(BatchNorm(self.head_dims[ihead]))

    def __str__(self):
        return "GATStack"
