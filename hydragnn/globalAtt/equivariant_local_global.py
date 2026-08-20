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

from typing import Any

import torch

from hydragnn.globalAtt.equivariant_features import (
    ScalarVectorIrrepsAdapter,
    create_local_feature_adapter,
)
from hydragnn.globalAtt.equivariant_transformer import EquivariantTransformerLayer


class EquivariantLocalGlobalConv(torch.nn.Module):
    """Apply a local MPNN update followed by equivariant global attention."""

    def __init__(
        self,
        channels: int,
        conv: torch.nn.Module,
        mpnn_type: str,
        heads: int,
        lmax: int,
        num_radial: int,
        feedforward_multiplier: int,
        allow_scalar_only: bool,
        require_tensor_coupling: bool,
        local_equivariance: bool,
        chunk_size: int | None,
    ):
        super().__init__()
        self.conv = conv
        self.adapter = create_local_feature_adapter(
            mpnn_type,
            channels,
            allow_scalar_only=allow_scalar_only,
            require_tensor_coupling=require_tensor_coupling,
            local_equivariance=local_equivariance,
        )
        self.global_layer = EquivariantTransformerLayer(
            self.adapter.irreps,
            heads=heads,
            lmax=lmax,
            num_radial=num_radial,
            feedforward_multiplier=feedforward_multiplier,
            require_tensor_coupling=require_tensor_coupling,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        inv_node_feat: torch.Tensor,
        equiv_node_feat: torch.Tensor,
        positions: torch.Tensor | None = None,
        graph_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run local and global updates without passing global data locally."""
        if positions is None:
            raise ValueError("EquivariantTransformer requires node positions")
        if graph_batch is None:
            raise ValueError("EquivariantTransformer requires a graph batch tensor")

        inv_node_feat, equiv_node_feat = self.conv(
            inv_node_feat=inv_node_feat,
            equiv_node_feat=equiv_node_feat,
            **kwargs,
        )
        if isinstance(self.adapter, ScalarVectorIrrepsAdapter):
            features = self.adapter(inv_node_feat, equiv_node_feat)
            features = self.global_layer(features, positions, graph_batch)
            return self.adapter.decode(features)

        features = self.adapter(inv_node_feat)
        features = self.global_layer(features, positions, graph_batch)
        return self.adapter.decode(features), equiv_node_feat
