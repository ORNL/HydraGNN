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
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
from .Base import Base


class PNAStack(Base):
    def __init__(
        self,
        deg: torch.Tensor,
        input_dim: int,
        output_dim: list,
        output_type: list,
        num_nodes: int,
        hidden_dim: int,
        config_heads: {},
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        ilossweights_hyperp: int = 1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        loss_weights: list = [1.0, 1.0, 1.0],  # weights for losses of different tasks
        ilossweights_nll: int = 0,  # if =1, using the scalar uncertainty as weights, as in paper
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    ):
        super().__init__(input_dim, hidden_dim, dropout, num_conv_layers)

        self.aggregators = ["mean", "min", "max", "std"]
        self.scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.deg = deg

        super()._init_model()

        super()._multihead(
            output_dim,
            num_nodes,
            output_type,
            config_heads,
            ilossweights_hyperp,
            loss_weights,
            ilossweights_nll,
        )

    def get_conv(self, dim):
        return PNAConv(
            in_channels=dim,
            out_channels=self.hidden_dim,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

    def __str__(self):
        return "PNAStack"
