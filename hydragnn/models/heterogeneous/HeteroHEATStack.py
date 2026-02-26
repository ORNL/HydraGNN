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
from torch.nn import Linear, ModuleDict, ModuleList
from torch_geometric.data import HeteroData
from torch_geometric.nn import BatchNorm, HEATConv

from .HeteroBase import HeteroBase


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
        self.edge_lin_dict = ModuleDict()
        self.node_types = None
        self.edge_types = None
        self._heat_edge_dim = None
        self.is_edge_model = True
        super().__init__(*args, **kwargs)

    def _init_conv(self):
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()

        self.node_types = list(self._metadata[0])
        self.edge_types = list(self._metadata[1])
        self._heat_edge_dim = self.hidden_dim

        for _ in range(self.num_conv_layers):
            self.graph_convs.append(
                HEATConv(
                    in_channels=-1,
                    out_channels=self.hidden_dim,
                    num_node_types=len(self.node_types),
                    num_edge_types=len(self.edge_types),
                    edge_type_emb_dim=self.edge_type_emb_dim,
                    edge_dim=self._heat_edge_dim,
                    edge_attr_emb_dim=self.edge_attr_emb_dim,
                    heads=self.attention_heads,
                    concat=False,
                )
            )
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

        projected_edge_attr_dict = {}
        if edge_attr_dict is not None:
            for edge_type, edge_attr in edge_attr_dict.items():
                self._ensure_edge_projector(
                    edge_type, edge_attr.size(-1), edge_attr.device
                )
                projected_edge_attr_dict[edge_type] = self.activation_function(
                    self.edge_lin_dict[str(edge_type)](edge_attr)
                )

        # Build temporary HeteroData for HEATConv -> homogeneous conversion
        hdata = HeteroData()
        for node_type in self.node_types:
            hdata[node_type].x = x_dict[node_type]

        for edge_type in self.edge_types:
            if edge_type in data.edge_index_dict:
                ei = data.edge_index_dict[edge_type]
            else:
                device = x_dict[self.node_types[0]].device
                ei = torch.empty((2, 0), dtype=torch.long, device=device)

            hdata[edge_type].edge_index = ei
            num_edges = ei.size(1)
            device = ei.device

            if edge_type in projected_edge_attr_dict:
                ea = projected_edge_attr_dict[edge_type]
                if ea.size(0) != num_edges:
                    raise ValueError(
                        f"edge_attr rows ({ea.size(0)}) must match num_edges ({num_edges}) for edge_type={edge_type}"
                    )
                hdata[edge_type].edge_attr = ea
            else:
                hdata[edge_type].edge_attr = torch.zeros(
                    (num_edges, self._heat_edge_dim),
                    dtype=hdata[self.node_types[0]].x.dtype,
                    device=device,
                )

        homo = hdata.to_homogeneous(node_attrs=["x"], edge_attrs=["edge_attr"])
        x = homo.x
        edge_index = homo.edge_index
        node_type = homo.node_type
        edge_type = homo.edge_type
        edge_attr = homo.edge_attr

        for conv, node_norms in zip(self.graph_convs, self.feature_layers):
            x = conv(x, edge_index, node_type, edge_type, edge_attr)
            for idx, node_name in enumerate(self.node_types):
                mask = node_type == idx
                if not torch.any(mask):
                    continue
                x_type = x[mask]
                x_type = self._apply_graph_conditioning(
                    x_type, batch_dict[node_name], data
                )
                x_type = node_norms[node_name](x_type)
                x_type = self.activation_function(x_type)
                x[mask] = x_type

        # Reconstruct x_dict from homogeneous representation
        x_dict = {}
        for idx, node_name in enumerate(self.node_types):
            mask = node_type == idx
            x_dict[node_name] = x[mask]

        return self._decode_from_x_dict(x_dict, batch_dict, data, edge_attr_dict=None)

    def __str__(self):
        return "HeteroHEATStack"
