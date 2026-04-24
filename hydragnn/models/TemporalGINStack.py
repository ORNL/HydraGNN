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
from .GINStack import GINStack


class TemporalGINStack(TemporalBase, GINStack):
    """T-GCN with GIN spatial convolution.

    MRO: TemporalGINStack → TemporalBase → GINStack → Base → Module

    TemporalBase.forward()   handles the temporal loop.
    GINStack.get_conv()      supplies the GIN spatial convolution.
    Base._init_conv()        populates graph_convs / feature_layers via get_conv.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"TemporalGINStack({self._temporal_mode},{self._backbone_type.upper()})"
