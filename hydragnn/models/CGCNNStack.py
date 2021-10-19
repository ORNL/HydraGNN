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
from torch_geometric.nn import CGConv, BatchNorm, global_mean_pool
from .Base import Base


class CGCNNStack(Base):
    def __init__(
        self,
        input_dim: int,
        output_dim: list,
        output_type: list,
        num_nodes: int,
        config_heads: {},
        edge_dim: int = 0,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        ilossweights_hyperp: int = 1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        loss_weights: list = [1.0, 1.0, 1.0],  # weights for losses of different tasks
        ilossweights_nll: int = 0,  # if =1, using the scalar uncertainty as weights, as in paper
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    ):
        super().__init__()
        self.input_dim = input_dim
        self.head_dims = output_dim
        self.head_type = output_type
        self.config_heads = config_heads
        # self.hidden_dim = hidden_dim
        self.hidden_dim = input_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            CGConv(
                channels=input_dim,
                dim=self.edge_dim,
                aggr="add",
                batch_norm=False,
                bias=True,
            )
        )
        self.batch_norms.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            conv = CGConv(
                channels=self.hidden_dim,
                dim=self.edge_dim,
                aggr="add",
                batch_norm=False,
                bias=True,
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

        print(
            "Warning: conv for node features decoder part not ready yet! Switch to shared mlp for prediction"
        )
        self.config_heads["node"]["type"] = "mlp"
        self.config_heads["node"]["share_mlp"] = True

        # fixme: CGConv layer alone will present the same out dimension with the input, instead of having different "in_channels" and "out_channels" as in the other conv layers;
        # so to predict output node features with different dimensions from the input node feature's, CGConv can be
        # combined with, e.g.,mlp
        return

    def __str__(self):
        return "CGCNNStack"
