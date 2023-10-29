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
from torch.nn import ReLU, Linear
from torch_geometric.nn import MFConv, BatchNorm, global_mean_pool, Sequential

from .Base import Base


class MFCStack(Base):
    def __init__(
        self,
        max_degree: int,
        *args,
        **kwargs,
    ):
        self.max_degree = max_degree

        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim):
        mfc = MFConv(
            in_channels=input_dim,
            out_channels=output_dim,
            max_degree=self.max_degree,
        )

        input_args = "x, pos, edge_index"
        conv_args = "x, edge_index"

        return Sequential(
            input_args,
            [
                (mfc, conv_args + " -> x"),
                (lambda x, pos: [x, pos], "x, pos -> x, pos"),
            ],
        )

    def __str__(self):
        return "MFCStack"
