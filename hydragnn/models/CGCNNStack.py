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
        self.edge_dim = edge_dim

        # CGCNN does not change embedding dimensions
        # We use input dimension (first argument of constructor) also as hidden dimension (second argument of constructor)
        super().__init__(
            input_dim,
            input_dim,
            output_dim,
            output_type,
            config_heads,
            num_nodes,
            ilossweights_hyperp,
            loss_weights,
            ilossweights_nll,
            dropout,
            num_conv_layers,
        )
        super()._multihead()

    def get_conv(self, input_dim, _):
        return CGConv(
            channels=input_dim,
            dim=self.edge_dim,
            aggr="add",
            batch_norm=False,
            bias=True,
        )

    def _init_node_conv(self):
        # *******convolutional layers for node level predictions*******#
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
        # fixme: CGConv layer alone will present the same out dimension with the input, instead of having different "in_channels" and "out_channels" as in the other conv layers;
        # so to predict output node features with different dimensions from the input node feature's, CGConv can be
        # combined with, e.g.,mlp
        self.config_heads["node"]["type"] = "mlp"
        self.config_heads["node"]["share_mlp"] = True
        return

    def __str__(self):
        return "CGCNNStack"
