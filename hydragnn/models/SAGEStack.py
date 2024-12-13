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
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv, BatchNorm, Sequential

from .Base import Base


class SAGEStack(Base):
    def __init__(self, *args, **kwargs):
        self.is_edge_model = False  # specify that mpnn cannot handle edge features
        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim, edge_dim=None):
        sage = SAGEConv(
            in_channels=input_dim,
            out_channels=output_dim,
        )

        return Sequential(
            self.input_args,
            [
                (sage, self.conv_args + " -> inv_node_feat"),
                (
                    lambda inv_node_feat, equiv_node_feat: [
                        inv_node_feat,
                        equiv_node_feat,
                    ],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
            ],
        )

    def __str__(self):
        return "SAGEStack"
