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
    CFConv,
    GaussianSmearing,
    RadiusInteractionGraph,
    ShiftedSoftplus,
)

from .Base import Base

from ..utils import unsorted_segment_mean


class SCFStack(Base):
    def __init__(
        self,
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

        super().__init__(*args, **kwargs)

        pass

    def _init_conv(self):

        self.distance_expansion = GaussianSmearing(0.0, self.radius, self.num_gaussians)
        self.interaction_graph = RadiusInteractionGraph(
            self.radius, self.max_neighbours
        )

        # comment on why equiv avoids last layer
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(
            self.get_conv(self.input_dim, self.hidden_dim, last_layer)
        )
        self.feature_layers.append(Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(Identity())

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

        conv_args = "x, edge_index, edge_weight, edge_attr, pos"
        if self.use_edge_attr:
            input_args = "x, pos, edge_index, edge_weight, edge_attr"
            return PyGSeq(
                input_args,
                [
                    (interaction, conv_args + " -> x"),
                    (lambda x, pos: [x, pos], "x, pos -> x, pos"),
                ],
            )
        elif self.equivariance and not last_layer:
            input_args = "x, pos, batch"
            return PyGSeq(
                input_args,
                [
                    (self.interaction_graph, "pos, batch -> edge_index, edge_weight"),
                    (self.distance_expansion, "edge_weight -> edge_attr"),
                    (interaction, conv_args + " -> x, pos"),
                ],
            )
        else:
            input_args = "x, pos, batch"
            return PyGSeq(
                input_args,
                [
                    (self.interaction_graph, "pos, batch -> edge_index, edge_weight"),
                    (self.distance_expansion, "edge_weight -> edge_attr"),
                    (interaction, conv_args + " -> x"),
                    (lambda x, pos: [x, pos], "x, pos -> x, pos"),
                ],
            )

    def _conv_args(self, data):
        if (self.use_edge_attr) and (self.equivariance):
            raise Exception(
                "For SchNet if using edge attributes, then E(3)-equivariance cannot be ensured. Please disable equivariance or edge attributes."
            )
        elif self.use_edge_attr:
            edge_index = data.edge_index
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

        return conv_args

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
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        pos: Tensor,
    ) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)

        if self.equivariant:
            radial, coord_diff = self.coord2radial(edge_index, pos)
            pos = self.coord_model(pos, edge_index, coord_diff, W)

        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        if self.equivariant:
            return x, pos
        else:
            return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial) + 1
        coord_diff = coord_diff / (norm)

        return radial, coord_diff
