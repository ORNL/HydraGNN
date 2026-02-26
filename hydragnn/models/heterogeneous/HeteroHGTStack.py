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
from torch_geometric.nn import BatchNorm, HGTConv

from .HeteroBase import HeteroBase


class HeteroHGTStack(HeteroBase):
    def __init__(self, num_heads: int, *args, **kwargs):
        self.num_heads = num_heads
        self.is_edge_model = False
        super().__init__(*args, **kwargs)

    def _init_conv(self):
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()

        for _ in range(self.num_conv_layers):
            self.graph_convs.append(
                HGTConv(
                    self.hidden_dim, self.hidden_dim, self._metadata, self.num_heads
                )
            )
            node_norms = ModuleDict({})
            for node_type in self._metadata[0]:
                node_norms[node_type] = BatchNorm(self.hidden_dim)
            self.feature_layers.append(node_norms)
        self._initialized = True

    def _init_node_conv(self):
        nodeconfiglist = self.config_heads.get("node", [])
        for branchdict in nodeconfiglist:
            if branchdict["architecture"]["type"] == "conv":
                raise NotImplementedError(
                    "HeteroHGTStack does not support conv-based node heads. Use 'mlp' or 'mlp_per_node'."
                )

    def forward(self, data):
        self._maybe_init_metadata(data)

        x_dict = data.x_dict
        self._ensure_node_embedders(x_dict)
        x_dict = {
            node_type: self.node_embedders[node_type](x.float())
            for node_type, x in x_dict.items()
        }

        batch_dict = self._get_batch_dict(data, x_dict)

        node_heads = self.config_heads.get("node", [])
        if node_heads and node_heads[0]["architecture"]["type"] == "conv":
            raise NotImplementedError(
                "HeteroHGTStack does not support conv-based node heads. Use 'mlp' or 'mlp_per_node'."
            )

        for conv, node_norms in zip(self.graph_convs, self.feature_layers):
            x_dict = conv(x_dict, data.edge_index_dict)
            for node_type, x in x_dict.items():
                x = self._apply_graph_conditioning(x, batch_dict[node_type], data)
                x = node_norms[node_type](x)
                x = self.activation_function(x)
                x_dict[node_type] = x

        return self._decode_from_x_dict(x_dict, batch_dict, data, edge_attr_dict=None)

    def __str__(self):
        return "HeteroHGTStack"
