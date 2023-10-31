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

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool, Sequential
from .Base import Base


class PNAStack(Base):
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

        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim):
        pna = PNAConv(
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

        input_args = "x, pos, edge_index"
        conv_args = "x, edge_index"

        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        return Sequential(
            input_args,
            [
                (pna, conv_args + " -> x"),
                (lambda x, pos: [x, pos], "x, pos -> x, pos"),
            ],
        )

    def __str__(self):
        return "PNAStack"
