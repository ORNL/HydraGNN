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
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
import torch.nn.functional as F
from .EGCLStack import EGCLStack

from hydragnn.utils.model import unsorted_segment_mean


class HybridEGCLStack(EGCLStack):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Define new loss functions
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def loss_hpweighted(self, pred, value, head_index, var=None):
        """
        Overwrite this method to make split loss between
        MSE (atom pos) and Cross Entropy (atom types).
        """

        # weights for different tasks as hyper-parameters
        tot_loss = 0
        tasks_loss = []
        for ihead in range(self.num_heads):
            head_pred = pred[ihead]
            pred_shape = head_pred.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)

            # Calculate loss depending on head
            # Calculate cross entropy if atom types
            if ihead == 0:
                head_loss = self.cross_entropy(head_pred, head_val)
            # Calculate MSE if position noise
            elif ihead == 1:
                head_loss = self.mse(head_pred, head_val)

            # Add loss to total loss and list of tasks loss
            tot_loss += head_loss * self.loss_weights[ihead]
            tasks_loss.append(head_loss)

        return tot_loss, tasks_loss

    def __str__(self):
        return "HybridEGCLStack"
