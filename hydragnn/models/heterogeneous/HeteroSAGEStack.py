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

from torch_geometric.nn import SAGEConv

from .HeteroBase import HeteroBase


class HeteroSAGEStack(HeteroBase):
    def __init__(self, *args, **kwargs):
        self.is_edge_model = False
        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim):
        return SAGEConv(in_channels=input_dim, out_channels=output_dim)

    def __str__(self):
        return "HeteroSAGEStack"
