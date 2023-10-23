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
from torch.nn import ReLU, Linear
from torch_geometric.nn import GATv2Conv, BatchNorm, Sequential

from .Base import Base


class GATStack(Base):
    def __init__(
        self,
        heads: int,
        negative_slope: float,
        *args,
        **kwargs,
    ):
        # note that self.heads is a parameter in GATConv, not the num_heads in the output part
        self.heads = heads
        self.negative_slope = negative_slope

        super().__init__(*args, **kwargs)

    def _init_conv(self):
        """Here this function overwrites _init_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        self.graph_convs.append(self.get_conv(self.input_dim, self.hidden_dim, True))
        self.feature_layers.append(BatchNorm(self.hidden_dim * self.heads))
        for _ in range(self.num_conv_layers - 2):
            conv = self.get_conv(self.hidden_dim * self.heads, self.hidden_dim, True)
            self.graph_convs.append(conv)
            self.feature_layers.append(BatchNorm(self.hidden_dim * self.heads))
        conv = self.get_conv(self.hidden_dim * self.heads, self.hidden_dim, False)
        self.graph_convs.append(conv)
        self.feature_layers.append(BatchNorm(self.hidden_dim))

    def _init_node_conv(self):
        """Here this function overwrites _init_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        # *******convolutional layers for node level predictions*******#
        # two ways to implement node features from here:
        # 1. one graph for all node features
        # 2. one graph for one node features (currently implemented)
        if (
            "node" not in self.config_heads
            or self.config_heads["node"]["type"] != "conv"
        ):
            return
        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return
        # In this part, each head has same number of convolutional layers, but can have different output dimension
        self.convs_node_hidden.append(
            self.get_conv(self.hidden_dim, self.hidden_dim_node[0], True)
        )
        self.batch_norms_node_hidden.append(
            BatchNorm(self.hidden_dim_node[0] * self.heads)
        )
        for ilayer in range(self.num_conv_layers_node - 1):
            self.convs_node_hidden.append(
                self.get_conv(
                    self.hidden_dim_node[ilayer] * self.heads,
                    self.hidden_dim_node[ilayer + 1],
                    True,
                )
            )
            self.batch_norms_node_hidden.append(
                BatchNorm(self.hidden_dim_node[ilayer + 1] * self.heads)
            )
        for ihead in node_feature_ind:
            self.convs_node_output.append(
                self.get_conv(
                    self.hidden_dim_node[-1] * self.heads, self.head_dims[ihead], False
                )
            )
            self.batch_norms_node_output.append(BatchNorm(self.head_dims[ihead]))

    def get_conv(self, input_dim, output_dim, concat):
        gat = GATv2Conv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=self.heads,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=True,
            concat=concat,
        )

        input_args = "x, pos, edge_index"
        conv_args = "x, edge_index"

        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        return Sequential(
            input_args,
            [
                (gat, conv_args + " -> x"),
                (lambda x, pos: [x, pos], "x, pos -> x, pos"),
            ],
        )

    def __str__(self):
        return "GATStack"
