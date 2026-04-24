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
from .GCNStack import GCNStack


class TemporalGCNStack(TemporalBase, GCNStack):
    """T-GCN with Kipf & Welling spectral GCN spatial convolution.

    MRO: TemporalGCNStack → TemporalBase → GCNStack → Base → Module

    TemporalBase.forward()   handles the temporal loop.
    GCNStack.get_conv()      supplies the spectral GCN spatial convolution.
    Base._init_conv()        populates graph_convs / feature_layers via get_conv.

    This is the closest HydraGNN equivalent to the original T-GCN architecture
    from Zhao et al. (2019), which also uses Kipf & Welling GCN for the spatial
    step and a GRU for the temporal step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"TemporalGCNStack({self._temporal_mode},{self._backbone_type.upper()})"
