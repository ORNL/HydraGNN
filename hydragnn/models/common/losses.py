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
"""Supervised multi-head loss shared by HydraGNN models."""

import torch

from hydragnn.loss_reporting import supervised_loss_report
from hydragnn.models.utils import loss_function_selection


class MultiHeadLossMixin:
    """Weighted sum of per-head losses.

    Requires ``num_heads``. Call :meth:`_configure_loss` from ``__init__``.
    """

    def _configure_loss(self, loss_function_type: str, loss_weights=None):
        # Output variance for Gaussian negative log likelihood loss.
        self.var_output = 1 if loss_function_type == "GaussianNLLLoss" else 0
        self.loss_function_type = loss_function_type
        self.loss_function = loss_function_selection(loss_function_type)
        if loss_weights is None:
            loss_weights = [1.0] * self.num_heads
        if len(loss_weights) != self.num_heads:
            raise ValueError(
                "Inconsistent number of loss weights and tasks: "
                + str(len(loss_weights))
                + " VS "
                + str(self.num_heads)
            )
        weightabssum = sum(abs(number) for number in loss_weights)
        self.loss_weights = [iw / weightabssum for iw in loss_weights]

    def loss(self, pred, value, head_index):
        var = None
        if self.var_output:
            pred, var = pred
        return self.loss_hpweighted(pred, value, head_index, var=var)

    def loss_hpweighted(self, pred, value, head_index, var=None):
        """Sum of per-head losses weighted by the normalized ``loss_weights``.

        Returns ``(total_loss, per_head_losses)``.
        """
        if var is None and self.loss_function_type == "GaussianNLLLoss":
            raise ValueError("Expecting var for GaussianNLLLoss, but got None")
        tot_loss = 0
        tasks_loss = []
        for ihead in range(self.num_heads):
            head_pre = pred[ihead]
            head_val = value[head_index[ihead]]
            if head_pre.shape != head_val.shape:
                head_val = torch.reshape(head_val, head_pre.shape)
            head_val = head_val.to(head_pre.device)
            if var is None:
                head_loss = self.loss_function(head_pre, head_val)
            else:
                head_loss = self.loss_function(head_pre, head_val, var[ihead])
            tot_loss += head_loss * self.loss_weights[ihead]
            tasks_loss.append(head_loss)

        self.last_loss_components = supervised_loss_report(
            self, pred, value, head_index, tasks_loss, var=var
        )
        return tot_loss, tasks_loss
