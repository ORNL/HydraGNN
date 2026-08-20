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
    IrrepsFeatureAdapter,
    ScalarVectorIrrepsAdapter,
    create_local_feature_adapter,
)
from hydragnn.globalAtt.equivariant_transformer import EquivariantTransformerLayer


class EquivariantLocalGlobalConv(torch.nn.Module):
    """Combine local message passing with equivariant global attention.

    ``parallel`` (the default) applies the local and global branches to the
    same input and adds their outputs in the shared irrep representation,
    matching the local/global organization of GraphGPS. ``sequential`` keeps
    the original local-then-global composition for controlled experiments.
    """

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
        coupling_mode: str = "parallel",
        irreps: str | None = None,
    ):
        super().__init__()
        if coupling_mode not in {"parallel", "sequential"}:
            raise ValueError("coupling_mode must be 'parallel' or 'sequential'")
        self.coupling_mode = coupling_mode
        self.conv = conv
        if mpnn_type == "MACE":
            if irreps is None:
                raise ValueError("MACE integration requires explicit output irreps")
            self.adapter = IrrepsFeatureAdapter(irreps)
        else:
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

        input_inv_node_feat = inv_node_feat
        input_equiv_node_feat = equiv_node_feat
        local_inv_node_feat, local_equiv_node_feat = self.conv(
            inv_node_feat=inv_node_feat,
            equiv_node_feat=equiv_node_feat,
            **kwargs,
        )
        if isinstance(self.adapter, (ScalarVectorIrrepsAdapter, IrrepsFeatureAdapter)):
            local_features = self.adapter(local_inv_node_feat, local_equiv_node_feat)
            if self.coupling_mode == "parallel":
                if isinstance(self.adapter, IrrepsFeatureAdapter):
                    global_input = self.adapter.encode_parallel_input(
                        input_inv_node_feat, input_equiv_node_feat
                    )
                else:
                    global_input = self.adapter(
                        input_inv_node_feat, input_equiv_node_feat
                    )
                global_features = self.global_layer.forward_attention(
                    global_input, positions, graph_batch
                )
                features = self.global_layer.forward_feedforward(
                    local_features + global_features
                )
            else:
                features = self.global_layer(local_features, positions, graph_batch)
            return self.adapter.decode(features)

        local_features = self.adapter(local_inv_node_feat)
        if self.coupling_mode == "parallel":
            global_features = self.global_layer.forward_attention(
                self.adapter(input_inv_node_feat), positions, graph_batch
            )
            features = self.global_layer.forward_feedforward(
                local_features + global_features
            )
        else:
            features = self.global_layer(local_features, positions, graph_batch)
        return self.adapter.decode(features), local_equiv_node_feat
