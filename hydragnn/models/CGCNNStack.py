##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import pdb
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import CGConv, BatchNorm, global_mean_pool, Sequential
from .Base import Base


class CGCNNStack(Base):
    def __init__(
        self,
        input_args,
        conv_args,
        edge_dim: int,
        input_dim,
        hidden_dim,
        output_dim,
        *args,
        **kwargs,
    ):
        self.edge_dim = edge_dim
        self.is_edge_model = True  # specify that mpnn can handle edge features
        # CGCNN does not change embedding dimensions
        # We use input dimension (first argument of base constructor)
        # also as hidden dimension (second argument of base constructor)
        # We therefore pass all required args explicitly.
        # Unless we use GPS, in which case hidden dimension is user defined and
        # typically different from input dim.
        super().__init__(
            input_args,
            conv_args,
            input_dim,
            hidden_dim,
            output_dim,
            *args,
            **kwargs,
        )

        if self.use_edge_attr or (
            self.use_global_attn and self.is_edge_model
        ):  # check if gps is being used and mpnn can handle edge feats
            assert (
                self.input_args
                == "inv_node_feat, equiv_node_feat, edge_index, edge_attr"
            )
            assert self.conv_args == "inv_node_feat, edge_index, edge_attr"
        else:
            assert self.input_args == "inv_node_feat, equiv_node_feat, edge_index"
            assert self.conv_args == "inv_node_feat, edge_index"

    def get_conv(self, input_dim, _, edge_dim=None):
        cgcnn = CGConv(
            channels=input_dim,
            dim=edge_dim,
            aggr="add",
            batch_norm=False,
            bias=True,
        )

        return Sequential(
            self.input_args,
            [
                (cgcnn, self.conv_args + " -> inv_node_feat"),
                (
                    lambda inv_node_feat, equiv_node_feat: [
                        inv_node_feat,
                        equiv_node_feat,
                    ],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
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
        nodeconfiglist = self.config_heads["node"]
        for branchdict in nodeconfiglist:
            if branchdict["architecture"]["type"] != "conv":
                return
        self.num_conv_layers_node = nodeconfiglist[0]["num_headlayers"]
        self.hidden_dim_node = nodeconfiglist[0]["dim_headlayers"]
        # fixme: CGConv layer alone will present the same out dimension with the input, instead of having different "in_channels" and "out_channels" as in the other conv layers;
        # so to predict output node features with different dimensions from the input node feature's, CGConv can be
        # combined with, e.g.,mlp
        for ihead in range(self.num_heads):
            for branchdict in nodeconfiglist:
                assert self.num_conv_layers_node == branchdict["num_headlayers"]
                assert self.hidden_dim_node == branchdict["dim_headlayers"]
                if self.head_type[ihead] == "node" and branchdict["type"] == "conv":
                    raise ValueError(
                        '"conv" for node features decoder part in CGCNN is not ready yet. Please set config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["type"] to be "mlp" or "mlp_per_node" in input file.'
                    )

    def __str__(self):
        return "CGCNNStack"
