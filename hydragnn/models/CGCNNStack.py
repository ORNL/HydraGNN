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
from torch_geometric.nn import CGConv, BatchNorm, global_mean_pool, Sequential
from .Base import Base


class CGCNNStack(Base):
    def __init__(
        self,
        edge_dim: int,
        input_dim,
        output_dim,
        *args,
        **kwargs,
    ):
        self.edge_dim = edge_dim

        # CGCNN does not change embedding dimensions
        # We use input dimension (first argument of base constructor)
        #    also as hidden dimension (second argument of base constructor)
        # We therefore pass all required args explicitly.
        super().__init__(
            input_dim,
            input_dim,
            output_dim,
            *args,
            **kwargs,
        )

    def get_conv(self, input_dim, _):
        cgcnn = CGConv(
            channels=input_dim,
            dim=self.edge_dim,
            aggr="add",
            batch_norm=False,
            bias=True,
        )

        input_args = "x, pos, edge_index"
        conv_args = "x, edge_index"

        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        return Sequential(
            input_args,
            [
                (cgcnn, conv_args + " -> x"),
                (lambda x, pos: [x, pos], "x, pos -> x, pos"),
            ],
        )

    def _init_node_conv(self):
        """It overwrites _init_node_conv() in Base since purely convolutional layers in _init_node_conv() is not implemented yet.
        Here it serves as a temporary place holder. Purely cgcnn conv is not feasible for node feature predictions with
        arbitrary output dimensions, unless we combine it with mlp"""
        # *******convolutional layers for node level predictions******* #
        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return
        self.num_conv_layers_node = self.config_heads["node"]["num_headlayers"]
        self.hidden_dim_node = self.config_heads["node"]["dim_headlayers"]
        # fixme: CGConv layer alone will present the same out dimension with the input, instead of having different "in_channels" and "out_channels" as in the other conv layers;
        # so to predict output node features with different dimensions from the input node feature's, CGConv can be
        # combined with, e.g.,mlp
        for ihead in range(self.num_heads):
            if (
                self.head_type[ihead] == "node"
                and self.config_heads["node"]["type"] == "conv"
            ):
                raise ValueError(
                    '"conv" for node features decoder part in CGCNN is not ready yet. Please set config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["type"] to be "mlp" or "mlp_per_node" in input file.'
                )

    def __str__(self):
        return "CGCNNStack"
