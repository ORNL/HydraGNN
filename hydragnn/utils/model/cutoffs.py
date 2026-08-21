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
"""Reusable smooth cutoff functions for geometry-dependent interactions."""

import torch
from torch import nn


def septic_cutoff(distance: torch.Tensor, onset: float, cutoff: float) -> torch.Tensor:
    """Return a C3 septic switch from one at ``onset`` to zero at ``cutoff``.

    This function supplies an opt-in primitive for model implementations. It
    does not automatically multiply native model envelopes, which would alter
    their reference architectures through double tapering.
    """
    if not 0 <= onset < cutoff:
        raise ValueError("cutoff radii must satisfy 0 <= onset < cutoff")
    t = ((distance - onset) / (cutoff - onset)).clamp(0.0, 1.0)
    switch = 1 + t**4 * (-35 + t * (84 + t * (-70 + 20 * t)))
    return torch.where(
        distance <= onset,
        torch.ones_like(distance),
        torch.where(distance < cutoff, switch, torch.zeros_like(distance)),
    )


class SepticCutoff(nn.Module):
    """Module wrapper around :func:`septic_cutoff`."""

    def __init__(self, onset: float, cutoff: float):
        super().__init__()
        if not 0 <= onset < cutoff:
            raise ValueError("cutoff radii must satisfy 0 <= onset < cutoff")
        self.onset = float(onset)
        self.cutoff = float(cutoff)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return septic_cutoff(distance, self.onset, self.cutoff)
