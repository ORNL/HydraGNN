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

import torch
from torch.nn import Linear, Module, ModuleDict, ModuleList
from torch_geometric.data import HeteroData
from torch_geometric.nn import BatchNorm, HEATConv

from .HeteroBase import HeteroBase


class _HeteroHEATConvAdapter(Module):
    def __init__(
        self,
        heat_conv,
        node_types,
        edge_types,
        heat_edge_dim: int,
    ):
        super().__init__()
        self.heat_conv = heat_conv
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.heat_edge_dim = heat_edge_dim

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        hdata = HeteroData()
        for node_type in self.node_types:
            hdata[node_type].x = x_dict[node_type]

        for edge_type in self.edge_types:
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
            else:
                device = x_dict[self.node_types[0]].device
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            hdata[edge_type].edge_index = edge_index
            num_edges = edge_index.size(1)
            device = edge_index.device

            if edge_attr_dict is not None and edge_type in edge_attr_dict:
                edge_attr = edge_attr_dict[edge_type]
                if edge_attr.size(0) != num_edges:
                    raise ValueError(
                        f"edge_attr rows ({edge_attr.size(0)}) must match "
                        f"num_edges ({num_edges}) for edge_type={edge_type}"
                    )
                hdata[edge_type].edge_attr = edge_attr
            else:
                hdata[edge_type].edge_attr = torch.zeros(
                    (num_edges, self.heat_edge_dim),
                    dtype=hdata[self.node_types[0]].x.dtype,
                    device=device,
                )

        homo = hdata.to_homogeneous(node_attrs=["x"], edge_attrs=["edge_attr"])
        x = self.heat_conv(
            homo.x,
            homo.edge_index,
            homo.node_type,
            homo.edge_type,
            homo.edge_attr,
        )

        out_dict = {}
        for idx, node_type in enumerate(self.node_types):
            out_dict[node_type] = x[homo.node_type == idx]
        return out_dict

    def reset_parameters(self):
        self.heat_conv.reset_parameters()


class HeteroHEATStack(HeteroBase):
    def __init__(
        self,
        attention_heads: int,
        edge_type_emb_dim: int,
        edge_attr_emb_dim: int,
        *args,
        **kwargs,
    ):
        self.attention_heads = attention_heads
        self.edge_type_emb_dim = edge_type_emb_dim
        self.edge_attr_emb_dim = edge_attr_emb_dim
        self.node_types = None
        self.edge_types = None
        self._heat_edge_dim = None
        self.is_edge_model = True
        super().__init__(*args, **kwargs)
        self.edge_lin_dict = ModuleDict()

    def _init_conv(self):
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()

        self.node_types = list(self._metadata[0])
        self.edge_types = list(self._metadata[1])
        self._heat_edge_dim = self.hidden_dim

        for _ in range(self.num_conv_layers):
            heat_conv = HEATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_node_types=len(self.node_types),
                num_edge_types=len(self.edge_types),
                edge_type_emb_dim=self.edge_type_emb_dim,
                edge_dim=self._heat_edge_dim,
                edge_attr_emb_dim=self.edge_attr_emb_dim,
                heads=self.attention_heads,
                concat=False,
            )
            mpnn = _HeteroHEATConvAdapter(
                heat_conv,
                self.node_types,
                self.edge_types,
                self._heat_edge_dim,
            )
            self.graph_convs.append(self._apply_global_attn(mpnn))
            node_norms = ModuleDict({})
            for node_type in self.node_types:
                node_norms[node_type] = BatchNorm(self.hidden_dim)
            self.feature_layers.append(node_norms)
        self._initialized = True

    def _init_node_conv(self):
        nodeconfiglist = self.config_heads.get("node", [])
        for branchdict in nodeconfiglist:
            if branchdict["architecture"]["type"] == "conv":
                raise NotImplementedError(
                    "HeteroHEATStack does not support conv-based node heads. Use 'mlp' or 'mlp_per_node'."
                )

    def _ensure_edge_projector(self, edge_type, edge_attr_dim: int, device):
        key = str(edge_type)
        if key not in self.edge_lin_dict:
            self.edge_lin_dict[key] = Linear(edge_attr_dim, self.hidden_dim)
        if self.edge_lin_dict[key].weight.device != device:
            self.edge_lin_dict[key] = self.edge_lin_dict[key].to(device)

    def forward(self, data):
        self._maybe_init_metadata(data)

        x_dict = data.x_dict
        self._ensure_node_embedders(x_dict)
        x_dict = {
            node_type: self.node_embedders[node_type](x.float())
            for node_type, x in x_dict.items()
        }

        batch_dict = self._get_batch_dict(data, x_dict)
        edge_attr_dict = self._get_edge_attr_dict(data)

        node_heads = self.config_heads.get("node", [])
        if node_heads and node_heads[0]["architecture"]["type"] == "conv":
            raise NotImplementedError(
                "HeteroHEATStack does not support conv-based node heads. Use 'mlp' or 'mlp_per_node'."
            )

        attention_only_gps = self.use_global_attn and self.attn_only
        projected_edge_attr_dict = {}
        if edge_attr_dict is not None and not attention_only_gps:
            for edge_type, edge_attr in edge_attr_dict.items():
                self._ensure_edge_projector(
                    edge_type, edge_attr.size(-1), edge_attr.device
                )
                projected_edge_attr_dict[edge_type] = self.activation_function(
                    self.edge_lin_dict[str(edge_type)](edge_attr)
                )

        for conv, node_norms in zip(self.graph_convs, self.feature_layers):
            if self.use_global_attn:
                x_dict = conv(
                    x_dict,
                    data.edge_index_dict,
                    batch_dict,
                    edge_attr_dict=projected_edge_attr_dict,
                )
            else:
                x_dict = conv(
                    x_dict,
                    data.edge_index_dict,
                    edge_attr_dict=projected_edge_attr_dict,
                )
            for node_type, x in x_dict.items():
                x = self._apply_graph_conditioning(x, batch_dict[node_type], data)
                x = node_norms[node_type](x)
                x = self.activation_function(x)
                x_dict[node_type] = x

        return self._decode_from_x_dict(x_dict, batch_dict, data, edge_attr_dict=None)

    def __str__(self):
        return "HeteroHEATStack"
