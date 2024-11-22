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

from typing import Optional
from math import pi as PI

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ReLU, Sequential
from torch_geometric.nn import Sequential as PyGSeq
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import (
    GaussianSmearing,
    RadiusInteractionGraph,
    ShiftedSoftplus,
)

from .Base import Base

from hydragnn.utils.model import unsorted_segment_mean
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths


class SCFStack(Base):
    def __init__(
        self,
        input_args,
        conv_args,
        num_filters: int,
        num_gaussians: list,
        radius: float,
        *args,
        max_neighbours: Optional[int] = None,
        **kwargs,
    ):
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        super().__init__(input_args, conv_args, *args, **kwargs)

        pass

    def _init_conv(self):
        self.distance_expansion = GaussianSmearing(0.0, self.radius, self.num_gaussians)
        self.interaction_graph = RadiusInteractionGraph(
            self.radius, self.max_neighbours
        )
        # comment on why equiv avoids last layer
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(self._apply_global_attn(self.get_conv(self.embed_dim, self.hidden_dim, last_layer)))
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            self.graph_convs.append(self._apply_global_attn(self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)))
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer):
        mlp = Sequential(
            Linear(self.num_gaussians, self.num_filters),
            ShiftedSoftplus(),
            Linear(self.num_filters, self.num_filters),
        )

        interaction = CFConv(
            in_channels=input_dim,
            out_channels=output_dim,
            nn=mlp,
            num_filters=self.num_filters,
            cutoff=self.radius,
            equivariant=self.equivariance and not last_layer,
        )

        if self.use_edge_attr:
            return PyGSeq(
                self.input_args,
                [
                    (interaction, self.conv_args + " -> inv_node_feat"),
                    (
                        lambda inv_node_feat, equiv_node_feat: [
                            inv_node_feat,
                            equiv_node_feat,
                        ],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )
        elif self.equivariance and not last_layer:
            return PyGSeq(
                self.input_args,
                [
                    (
                        self.interaction_graph,
                        "equiv_node_feat, batch -> edge_index, edge_weight",
                    ),
                    (self.distance_expansion, "edge_weight -> edge_attr"),
                    (
                        interaction,
                        self.conv_args + " -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )
        else:
            return PyGSeq(
                self.input_args,
                [
                    (
                        self.interaction_graph,
                        "equiv_node_feat, batch -> edge_index, edge_weight",
                    ),
                    (self.distance_expansion, "edge_weight -> edge_attr"),
                    (interaction, self.conv_args + " -> inv_node_feat"),
                    (
                        lambda inv_node_feat, equiv_node_feat: [
                            inv_node_feat,
                            equiv_node_feat,
                        ],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )

    def _embedding(self, data):
        super()._embedding(data)

        if (self.use_edge_attr) and (self.equivariance):
            raise Exception(
                "For SchNet if using edge attributes, then E(3)-equivariance cannot be ensured. Please disable equivariance or edge attributes."
            )
        elif self.use_edge_attr:
            edge_index = data.edge_index
            data.edge_shifts = torch.zeros(
                (data.edge_index.size(1), 3), device=data.edge_index.device
            )  # Override. pbc edge shifts are currently not supported in positional update models
            edge_weight = data.edge_attr.norm(dim=-1)

            conv_args = {
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "edge_attr": self.distance_expansion(edge_weight),
            }
        else:
            conv_args = {
                "batch": data.batch,
            }

        if self.use_global_attn:
            x = self.pos_emb(data.pe)
            e = self.rel_pos_emb(data.rel_pe)
            if self.input_dim:
                x = torch.cat((self.node_emb(data.x.float()), x), 1)
                x = self.node_lin(x)
            if self.use_edge_attr:
                e = torch.cat((self.edge_emb(conv_args['edge_attr']), e), 1 )
                e = self.edge_lin(e)    
            conv_args.update({"edge_attr": e})
            return x, data.pos, conv_args 
        else:
            return data.x, data.pos, conv_args

    def __str__(self):
        return "SCFStack"


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
        equivariant: bool,
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.equivariant = equivariant

        if self.equivariant:

            layer = Linear(num_filters, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(Linear(num_filters, num_filters))
            coord_mlp.append(ReLU())
            coord_mlp.append(layer)
            self.coord_mlp = Sequential(*coord_mlp)

        self.reset_parameters()

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(
            trans, min=-100, max=100
        )  # This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg
        return coord

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)

        if self.equivariant:
            edge_shifts = torch.zeros(
                (edge_index.size(1), 3), device=edge_index.device
            )  # pbc edge shifts are currently not supported in positional update models
            coord_diff, radial = get_edge_vectors_and_lengths(
                pos, edge_index, edge_shifts, normalize=True, eps=1.0
            )
            pos = self.coord_model(pos, edge_index, coord_diff, W)

        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        if self.equivariant:
            return x, pos
        else:
            return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W
