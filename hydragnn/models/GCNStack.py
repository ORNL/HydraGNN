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

from torch_geometric.nn import GCNConv, BatchNorm, Sequential

from .Base import Base


class GCNStack(Base):
    """Kipf & Welling (2017) spectral GCN.

    Uses PyG's GCNConv which applies symmetric normalisation
    (A_hat = D^{-1/2} (A + I) D^{-1/2}) before message-passing.
    This is the same spectral convolution used in the original T-GCN paper.
    """

    def __init__(self, *args, **kwargs):
        self.is_edge_model = False  # GCNConv does not consume edge features
        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim, edge_dim=None):
        gcn = GCNConv(
            input_dim,
            output_dim,
            improved=False,
            add_self_loops=True,
            normalize=True,
        )

        return Sequential(
            self.input_args,
            [
                (gcn, self.conv_args + " -> inv_node_feat"),
                (
                    lambda x, equiv_node_feat: [x, equiv_node_feat],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
            ],
        )

    def __str__(self):
        return "GCNStack"
