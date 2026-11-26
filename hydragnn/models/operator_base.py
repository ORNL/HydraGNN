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
"""Encode-process-decode base class for PDE operator models.

Exposes the same attributes the HydraGNN trainer reads from ``Base``
(``num_heads``, ``head_dims``, ``head_type``, ``var_output``,
``loss_weights``, ``loss``). The latent state passed between stages is a
dict keyed by entity level (``"node"``, ``"edge"``, ...).
"""

import torch

from hydragnn.models.common.losses import MultiHeadLossMixin


class OperatorBase(MultiHeadLossMixin, torch.nn.Module):
    supported_output_levels = frozenset({"node"})

    def __init__(self, output_dim, output_type, loss_function_type, loss_weights=None):
        super().__init__()
        unsupported = set(output_type) - self.supported_output_levels
        if unsupported:
            raise ValueError(
                f"{type(self).__name__} supports output levels "
                f"{sorted(self.supported_output_levels)}, got {sorted(unsupported)}"
            )
        if len(output_dim) != len(output_type):
            raise ValueError("output_dim and output_type must have the same length")
        self.head_dims = list(output_dim)
        self.head_type = list(output_type)
        self.num_heads = len(self.head_dims)
        self.conv_checkpointing = False
        self._configure_loss(loss_function_type, loss_weights)

    def encode(self, data):
        raise NotImplementedError

    def process(self, state, data):
        raise NotImplementedError

    def decode(self, state, data):
        raise NotImplementedError

    def forward(self, data):
        outputs = self.decode(self.process(self.encode(data), data), data)
        if not self.var_output:
            return outputs
        means, variances = [], []
        for out, dim in zip(outputs, self.head_dims):
            means.append(out[:, :dim])
            variances.append(torch.nn.functional.softplus(out[:, dim:]) + 1e-6)
        return means, variances

    def configure_input_feature_encoders(self, encodings):
        if encodings:
            raise ValueError(
                f"{type(self).__name__} does not support input_node_encodings"
            )

    def enable_conv_checkpointing(self):
        self.conv_checkpointing = True
