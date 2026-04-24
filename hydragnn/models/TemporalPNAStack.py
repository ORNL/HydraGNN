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

from .TemporalBase import TemporalBase
from .PNAStack import PNAStack


class TemporalPNAStack(TemporalBase, PNAStack):
    """T-GCN with PNA spatial convolution.

    MRO: TemporalPNAStack → TemporalBase → PNAStack → Base → Module

    PNAStack requires positional args: input_args, conv_args, deg, edge_dim.
    These are forwarded transparently through TemporalBase via *args/**kwargs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"TemporalPNAStack({self._temporal_mode},{self._backbone_type.upper()})"
