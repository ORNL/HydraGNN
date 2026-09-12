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

import importlib

import torch


def test_named_task_loss_logging_preserves_output_names(monkeypatch):
    training = importlib.import_module("hydragnn.train.train_validate_test")
    messages = []
    monkeypatch.setattr(
        training,
        "print_distributed",
        lambda _verbosity, message: messages.append(message),
    )

    training._print_named_task_losses(
        1,
        ["energy", "band_gap"],
        {
            "Train": torch.tensor([1.0, 2.0]),
            "Val": torch.tensor([3.0, 4.0]),
            "Test": torch.tensor([5.0, 6.0]),
        },
    )

    assert messages == [
        "energy Train Loss: 1.00000000, Val Loss: 3.00000000, "
        "Test Loss: 5.00000000",
        "band_gap Train Loss: 2.00000000, Val Loss: 4.00000000, "
        "Test Loss: 6.00000000",
    ]
