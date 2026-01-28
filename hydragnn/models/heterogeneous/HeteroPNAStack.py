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
from torch_geometric.nn import PNAConv
from torch.nn import Module

from .HeteroBase import HeteroBase


class _BipartitePNAWrapper(Module):
    def __init__(self, conv: PNAConv):
        super().__init__()
        self.conv = conv

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, tuple):
            x_src, x_dst = x
            x_cat = torch.cat([x_src, x_dst], dim=0)
            edge_index_cat = torch.stack(
                [edge_index[0], edge_index[1] + x_src.size(0)], dim=0
            )
            out = self.conv(x_cat, edge_index_cat, edge_attr=edge_attr)
            return out[x_src.size(0) :]
        return self.conv(x, edge_index, edge_attr=edge_attr)


class HeteroPNAStack(HeteroBase):
    def __init__(
        self,
        deg: list,
        edge_dim: int,
        *args,
        **kwargs,
    ):
        self.aggregators = ["mean", "min", "max", "std"]
        self.scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.deg = torch.Tensor(deg)
        self.edge_dim = edge_dim
        self.is_edge_model = True
        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim):
        conv = PNAConv(
            in_channels=input_dim,
            out_channels=output_dim,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            edge_dim=self.edge_dim,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        return _BipartitePNAWrapper(conv)

    def __str__(self):
        return "HeteroPNAStack"
