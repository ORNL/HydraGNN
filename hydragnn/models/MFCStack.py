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
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import ReLU, Linear
from torch_geometric.nn import MFConv, BatchNorm, global_mean_pool, Sequential

from .Base import Base


class MFCStack(Base):
    def __init__(
        self,
        input_args,
        conv_args,
        max_degree: int,
        *args,
        **kwargs,
    ):
        self.max_degree = max_degree
        self.is_edge_model = False  # specify that mpnn cannot handle edge features
        super().__init__(input_args, conv_args, *args, **kwargs)

    def get_conv(self, input_dim, output_dim, edge_dim=None):
        mfc = MFConv(
            in_channels=input_dim,
            out_channels=output_dim,
            max_degree=self.max_degree,
        )

        return Sequential(
            self.input_args,
            [
                (mfc, self.conv_args + " -> inv_node_feat"),
                (
                    lambda x, pos: [x, pos],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
            ],
        )

    def __str__(self):
        return "MFCStack"
