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

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential
from typing import Optional, Dict, Any

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch


class EquiformerV2Conv(torch.nn.Module):
    """
    EquiformerV2-based global attention wrapper that maintains the same interface as GPSConv.

    This is a simplified wrapper that integrates EquiformerV2's attention mechanisms
    with HydraGNN's MPNN layers. For now, this is a placeholder structure that will
    be expanded with actual EquiformerV2 components.
    """

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 8,
        dropout: float = 0.0,
        lmax_list: list = [6],
        mmax_list: list = [2],
        sphere_channels: int = None,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.sphere_channels = sphere_channels or channels
        self.num_layers = num_layers

        # For now, create placeholder components
        # In the full implementation, these would be actual EquiformerV2 components
        self.placeholder_attention = torch.nn.MultiheadAttention(
            channels, heads, batch_first=True, dropout=dropout
        )

        # Feedforward network similar to GPS
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            Linear(channels * 2, channels),
            torch.nn.Dropout(dropout),
        )

        # Layer normalizations
        self.norm1 = torch.nn.LayerNorm(channels)
        self.norm2 = torch.nn.LayerNorm(channels)
        self.norm3 = torch.nn.LayerNorm(channels)

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.placeholder_attention._reset_parameters()
        for module in self.mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def forward(
        self,
        inv_node_feat: Tensor,
        equiv_node_feat: Tensor,
        graph_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass through EquiformerV2-based global attention.

        Args:
            inv_node_feat: Invariant node features
            equiv_node_feat: Equivariant node features (if any)
            graph_batch: Batch tensor for graph nodes
            **kwargs: Additional arguments passed to the MPNN layer

        Returns:
            Updated invariant and equivariant node features
        """
        hs = []

        # Local MPNN processing (if provided)
        if self.conv is not None:
            h, equiv_node_feat = self.conv(
                inv_node_feat=inv_node_feat, equiv_node_feat=equiv_node_feat, **kwargs
            )
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + inv_node_feat
            h = self.norm1(h)
            hs.append(h)

        # Global attention processing
        # For now, using placeholder attention - will be replaced with EquiformerV2
        h, mask = to_dense_batch(inv_node_feat, graph_batch)
        h, _ = self.placeholder_attention(
            h, h, h, key_padding_mask=~mask, need_weights=False
        )
        h = h[mask]

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + inv_node_feat  # Residual connection
        h = self.norm2(h)
        hs.append(h)

        # Combine local and global outputs
        out = sum(hs) if len(hs) > 1 else hs[0]

        # Final feedforward
        out = out + self.mlp(out)
        out = self.norm3(out)

        return out, equiv_node_feat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"lmax_list={self.lmax_list}, mmax_list={self.mmax_list})"
        )
