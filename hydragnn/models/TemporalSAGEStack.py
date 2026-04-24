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
from .SAGEStack import SAGEStack


class TemporalSAGEStack(TemporalBase, SAGEStack):
    """T-GCN with GraphSAGE spatial convolution.

    MRO: TemporalSAGEStack → TemporalBase → SAGEStack → Base → Module
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"TemporalSAGEStack({self._temporal_mode},{self._backbone_type.upper()})"
