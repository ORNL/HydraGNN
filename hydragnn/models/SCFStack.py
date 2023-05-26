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
from torch.nn import Linear, Sequential
from torch_geometric.nn.models.schnet import (
    CFConv,
    GaussianSmearing,
    RadiusInteractionGraph,
    ShiftedSoftplus,
)

from .Base import Base


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

        self.distance_expansion = GaussianSmearing(0.0, radius, num_gaussians)
        self.interaction_graph = RadiusInteractionGraph(radius, max_neighbours)

        pass

    def get_conv(self, input_dim, output_dim):
        mlp = Sequential(
            Linear(self.num_gaussians, self.num_filters),
            ShiftedSoftplus(),
            Linear(self.num_filters, self.num_filters),
        )

        return CFConv(
            in_channels=input_dim,
            out_channels=output_dim,
            nn=mlp,
            num_filters=self.num_filters,
            cutoff=self.radius,
        )

    def _conv_args(self, data):
        if (data.edge_attr is not None) and (self.use_edge_attr):
            edge_index = data.edge_index
            edge_weight = data.edge_attr.norm(dim=-1)
        else:
            edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)

        conv_args = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_attr": self.distance_expansion(edge_weight),
        }

        return conv_args

    def __str__(self):
        return "SCFStack"
