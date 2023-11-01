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

import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
from .Base import Base

from ..utils import unsorted_segment_mean


class EGCLStack(Base):
    def __init__(
        self,
        edge_attr_dim: int,
        *args,
        max_neighbours: Optional[int] = None,
        **kwargs,
    ):

        self.edge_dim = (
            0 if edge_attr_dim is None else edge_attr_dim
        )  # Must be named edge_dim to trigger use by Base
        super().__init__(*args, **kwargs)
        pass

    def _init_conv(self):
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(
            self.get_conv(self.input_dim, self.hidden_dim, last_layer)
        )
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer=False):
        egcl = E_GCL(
            input_channels=input_dim,
            output_channels=output_dim,
            hidden_channels=self.hidden_dim,
            edge_attr_dim=self.edge_dim,
            equivariant=self.equivariance and not last_layer,
        )

        if self.equivariance and not last_layer:
            return Sequential(
                "x, pos, edge_index, edge_attr",
                [
                    (egcl, "x, pos, edge_index, edge_attr -> x, pos"),
                ],
            )
        else:
            return Sequential(
                "x, pos, edge_index, edge_attr",
                [
                    (egcl, "x, pos, edge_index, edge_attr -> x"),
                    (lambda x, pos: [x, pos], "x, pos -> x, pos"),
                ],
            )

    def _conv_args(self, data):
        if self.edge_dim > 0:
            conv_args = {
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr,
            }
        else:
            conv_args = {
                "edge_index": data.edge_index,
                "edge_attr": None,
            }

        return conv_args

    def __str__(self):
        return "EGCLStack"


"""
EGNN
=====

E(n) equivariant graph neural network as
a message passing neural network. The
model uses positional data only to ensure
that the message passing component is
equivariant.

In particular this message passing layer
relies on the angle formed by the triplet
of incomming and outgoing messages.

The three key components of this network are
outlined below. In particular, the convolutional
network that is used for the message passing
the triplet function that generates to/from
information for angular values, and finally
the radial basis embedding that is used to
include radial basis information.

"""


class E_GCL(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels,
        edge_attr_dim=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        recurrent=False,
        coords_weight=1.0,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=True,
        equivariant=False,
    ) -> None:
        super(E_GCL, self).__init__()
        input_edge = input_channels * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        self.equivariant = equivariant
        self.edge_attr_dim = edge_attr_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edge_attr_dim, hidden_channels),
            act_fn,
            nn.Linear(hidden_channels, hidden_channels),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(
                hidden_channels + input_channels + nodes_attr_dim, hidden_channels
            ),
            act_fn,
            nn.Linear(hidden_channels, output_channels),
        )

        self.clamp = clamp

        if self.equivariant:

            layer = nn.Linear(hidden_channels, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_channels, hidden_channels))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
                self.coords_range = nn.Parameter(torch.ones(1)) * 3
            self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())

        self.act_fn = act_fn
        pass

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(
            trans, min=-100, max=100
        )  # This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / (norm)

        return radial, coord_diff

    def forward(self, x, coord, edge_index, edge_attr, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # Message Passing
        edge_feat = self.edge_model(x[row], x[col], radial, edge_attr)
        if self.equivariant:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        x, agg = self.node_model(x, edge_index, edge_feat, node_attr)
        if self.equivariant:
            return x, coord
        else:
            return x


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result
