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
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        # note that self.heads is a parameter in GATConv, not the num_heads in the output part
        self.heads = heads
        self.negative_slope = negative_slope
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        self.__init_model()

        super()._multihead(
            output_dim,
            num_nodes,
            output_type,
            config_heads,
            ilossweights_hyperp,
            loss_weights,
            ilossweights_nll,
        )

    def __init_model(self):
        self.convs.append(self.get_conv(self.input_dim, True))
        self.batch_norms.append(BatchNorm(self.hidden_dim * self.heads))
        for _ in range(self.num_conv_layers - 2):
            conv = self.get_conv(self.hidden_dim * self.heads, True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim * self.heads))
        conv = self.get_conv(self.hidden_dim * self.heads, False)
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(self.hidden_dim))

    def get_conv(self, in_channels, concat):
        return GATv2Conv(
            in_channels=in_channels,
            out_channels=self.hidden_dim,
            heads=self.heads,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=True,
            concat=concat,
        )

    def __str__(self):
        return "GATStack"
