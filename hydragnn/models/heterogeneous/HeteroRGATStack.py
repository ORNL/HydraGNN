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

from torch.nn import ModuleDict, ModuleList
from torch_geometric.nn import BatchNorm, GATConv, HeteroConv

from .HeteroBase import HeteroBase


class HeteroRGATStack(HeteroBase):
    def __init__(
        self,
        heads: int,
        negative_slope: float,
        edge_dim: int,
        *args,
        **kwargs,
    ):
        self.heads = heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim
        self.is_edge_model = True
        super().__init__(*args, **kwargs)

    def _build_hetero_rgat_conv(self, input_dim: int, output_dim: int, concat: bool):
        conv_dict = {}
        shared_conv = None
        for edge_type in self._metadata[1]:
            if self.share_relation_weights:
                if shared_conv is None:
                    shared_conv = GATConv(
                        in_channels=input_dim,
                        out_channels=output_dim,
                        heads=self.heads,
                        negative_slope=self.negative_slope,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=self.edge_dim,
                        concat=concat,
                    )
                conv_dict[edge_type] = shared_conv
            else:
                conv_dict[edge_type] = GATConv(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    heads=self.heads,
                    negative_slope=self.negative_slope,
                    dropout=self.dropout,
                    add_self_loops=False,
                    edge_dim=self.edge_dim,
                    concat=concat,
                )
        return HeteroConv(conv_dict, aggr="sum")

    def _init_conv(self):
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()

        # First layer: concat=True -> hidden_dim * heads
        self.graph_convs.append(
            self._build_hetero_rgat_conv(self.hidden_dim, self.hidden_dim, True)
        )
        node_norms = ModuleDict({})
        for node_type in self._metadata[0]:
            node_norms[node_type] = BatchNorm(self.hidden_dim * self.heads)
        self.feature_layers.append(node_norms)

        # Middle layers: concat=True -> hidden_dim * heads
        for _ in range(self.num_conv_layers - 2):
            self.graph_convs.append(
                self._build_hetero_rgat_conv(
                    self.hidden_dim * self.heads, self.hidden_dim, True
                )
            )
            node_norms = ModuleDict({})
            for node_type in self._metadata[0]:
                node_norms[node_type] = BatchNorm(self.hidden_dim * self.heads)
            self.feature_layers.append(node_norms)

        # Final layer: concat=False -> hidden_dim
        if self.num_conv_layers > 1:
            self.graph_convs.append(
                self._build_hetero_rgat_conv(
                    self.hidden_dim * self.heads, self.hidden_dim, False
                )
            )
            node_norms = ModuleDict({})
            for node_type in self._metadata[0]:
                node_norms[node_type] = BatchNorm(self.hidden_dim)
            self.feature_layers.append(node_norms)

    def _init_node_conv(self):
        nodeconfiglist = self.config_heads["node"]
        assert (
            self.num_branches == len(nodeconfiglist) or self.num_branches == 1
        ), "assuming node head has the same branches as graph head, if any"
        for branchdict in nodeconfiglist:
            if branchdict["architecture"]["type"] != "conv":
                return

        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return

        for branchdict in nodeconfiglist:
            branchtype = branchdict["type"]
            brancharct = branchdict["architecture"]
            num_conv_layers_node = brancharct["num_headlayers"]
            hidden_dim_node = brancharct["dim_headlayers"]

            convs_node_hidden = ModuleList()
            batch_norms_node_hidden = ModuleList()
            convs_node_output = ModuleList()
            batch_norms_node_output = ModuleList()

            convs_node_hidden.append(
                self._build_hetero_rgat_conv(self.hidden_dim, hidden_dim_node[0], True)
            )
            bn_dict = ModuleDict({})
            for node_type in self._metadata[0]:
                bn_dict[node_type] = BatchNorm(hidden_dim_node[0] * self.heads)
            batch_norms_node_hidden.append(bn_dict)

            for ilayer in range(num_conv_layers_node - 1):
                convs_node_hidden.append(
                    self._build_hetero_rgat_conv(
                        hidden_dim_node[ilayer] * self.heads,
                        hidden_dim_node[ilayer + 1],
                        True,
                    )
                )
                bn_dict = ModuleDict({})
                for node_type in self._metadata[0]:
                    bn_dict[node_type] = BatchNorm(
                        hidden_dim_node[ilayer + 1] * self.heads
                    )
                batch_norms_node_hidden.append(bn_dict)

            for ihead in node_feature_ind:
                convs_node_output.append(
                    self._build_hetero_rgat_conv(
                        hidden_dim_node[-1] * self.heads,
                        self.head_dims[ihead],
                        False,
                    )
                )
                bn_dict = ModuleDict({})
                for node_type in self._metadata[0]:
                    bn_dict[node_type] = BatchNorm(self.head_dims[ihead])
                batch_norms_node_output.append(bn_dict)

            self.convs_node_hidden[branchtype] = convs_node_hidden
            self.batch_norms_node_hidden[branchtype] = batch_norms_node_hidden
            self.convs_node_output[branchtype] = convs_node_output
            self.batch_norms_node_output[branchtype] = batch_norms_node_output

    def __str__(self):
        return "HeteroRGATStack"
