##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Training-time switches shared by HydraGNN models."""

from hydragnn.utils.print.print_utils import print_master


class ModelControlsMixin:
    """Freezing, output-bias initialization and activation checkpointing.

    Requires ``graph_convs``, ``feature_layers``, ``heads_NN``, ``head_type``
    and ``initial_bias``.
    """

    def _freeze_conv(self):
        for module in [self.graph_convs, self.feature_layers]:
            for layer in module:
                for param in layer.parameters():
                    param.requires_grad = False

    def _set_bias(self):
        # Set the bias of the last linear layer of graph heads to a large value (UQ).
        for head, type in zip(self.heads_NN, self.head_type):
            if type == "graph":
                for branch in head.values():
                    branch[-1].bias.data.fill_(self.initial_bias)

    def enable_conv_checkpointing(self):
        print_master("Enabling checkpointing")
        self.conv_checkpointing = True
