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

import torch.nn as nn
from torch_geometric.nn import GINConv

from .HeteroBase import HeteroBase


class HeteroGINStack(HeteroBase):
    def __init__(self, *args, **kwargs):
        self.is_edge_model = False
        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim):
        gin = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            ),
            eps=100.0,
            train_eps=True,
        )
        return gin

    def __str__(self):
        return "HeteroGINStack"
