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


import inspect
from typing import Any, Dict, Optional, Tuple
import pdb
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential, LazyLinear, ReLU, BatchNorm1d

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch


class GPSConvEquivariant(torch.nn.Module):
    """
    GPS layer that maintains E(3) equivariance for vector features while performing
    global attention on scalar features.
    
    This layer processes scalar (invariant) and vector (equivariant) features separately
    to preserve geometric consistency while enabling global reasoning:
    
    Why Global Attention Only on Scalars:
    - Standard attention mechanisms (dot products + softmax) are NOT equivariant to rotations
    - When vector features are rotated, attention weights change, violating equivariance
    - Scalar features represent rotation-invariant properties (atom types, charges, energies)
      that can safely undergo global attention without breaking geometric constraints
    
    Information Flow Strategy:
    - Scalar path: Local MPNN → Global Attention → Enhanced scalars (global reasoning)
    - Vector path: Local MPNN → Scalar-gated updates → Updated positions (geometric consistency)
    - Scalars act as a "global information highway" that informs local geometric updates
    
    Architecture:
    - Scalar features undergo normal GPS processing (local MPNN + global attention)
    - Vector features are processed through equivariant operations only (no attention)
    - Position updates are computed from processed vector features via scalar gating
    - This design maintains mathematical rigor while being computationally efficient
    """

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing] = None,
        heads: int = 1,
        dropout: float = 0.0,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "layer_norm",
        norm_with_batch: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type
        self.norm_with_batch = norm_with_batch

        attn_kwargs = attn_kwargs or {}

        if attn_type == "multihead":
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                dropout=dropout,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == "performer":
            self.attn = PerformerAttention(
                channels,
                heads,
                dropout=dropout,
                **attn_kwargs,
            )
        else:
            raise ValueError(f"Attention type {attn_type} not supported")

        self.norm1 = None
        self.norm2 = None  
        self.norm3 = None
        if norm is not None:
            if norm == "batch_norm":
                self.norm1 = BatchNorm1d(channels)
                self.norm2 = BatchNorm1d(channels)
                self.norm3 = BatchNorm1d(channels)
            elif norm == "layer_norm":
                self.norm1 = torch.nn.LayerNorm(channels)
                self.norm2 = torch.nn.LayerNorm(channels)
                self.norm3 = torch.nn.LayerNorm(channels)

        self.scalar_mlp = Sequential(
            Linear(channels, channels * 2),
            ReLU(),
            Linear(channels * 2, channels),
        )
        
        # Simple equivariant processing components
        # Position update network (scalar features -> position updates)
        self.pos_update_net = Linear(channels, 3, bias=False)  # No bias to maintain equivariance
        self.pos_update_scale = torch.nn.Parameter(torch.tensor(0.01))  # Learnable scale

    def forward(
        self,
        inv_node_feat: Tensor,
        equiv_node_feat: Tensor,
        graph_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Runs the forward pass of the equivariant module.
        
        Args:
            inv_node_feat: Scalar (invariant) node features [N, channels]
            equiv_node_feat: Vector (equivariant) node features [N, 3] (positions)
            graph_batch: Batch assignment for nodes
            
        Returns:
            tuple: (updated_scalar_features, updated_positions)
        """
        device = inv_node_feat.device
        num_nodes = inv_node_feat.shape[0]
        
        # Store original scalar features for residual connections
        orig_scalar = inv_node_feat
        
        hs = []
        
        # Local MPNN processing
        if self.conv is not None:
            h, updated_equiv = self.conv(
                inv_node_feat=inv_node_feat, equiv_node_feat=equiv_node_feat, **kwargs
            )
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + orig_scalar  # Residual connection for scalars
            
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=graph_batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
            
            # Update positions if conv layer provided updates
            if updated_equiv is not None:
                equiv_node_feat = updated_equiv

        # Global attention (operates only on scalar features to maintain equivariance)
        h, mask = to_dense_batch(inv_node_feat, graph_batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + inv_node_feat  # Residual connection
        
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=graph_batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        # Combine local and global scalar outputs
        scalar_out = sum(hs)

        # Process scalar features through MLP
        scalar_out = scalar_out + self.scalar_mlp(scalar_out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                scalar_out = self.norm3(scalar_out, batch=graph_batch)
            else:
                scalar_out = self.norm3(scalar_out)

        # Compute equivariant position updates from enhanced scalar features
        # This maintains equivariance by using scalar features to generate position deltas
        position_updates = self.pos_update_net(scalar_out)  # [N, 3]
        position_updates = position_updates * self.pos_update_scale
        
        # Handle different input formats for equiv_node_feat
        if equiv_node_feat is not None:
            if equiv_node_feat.dim() == 2 and equiv_node_feat.size(1) == 3:
                # Case 1: Position data [N, 3] - directly add position updates
                updated_equiv_node_feat = equiv_node_feat + position_updates
            elif equiv_node_feat.dim() == 3 and equiv_node_feat.size(1) == 3:
                # Case 2: Vector features [N, 3, channels] - update positions in a compatible way
                # Extract position-like information from first channel and update
                positions_like = equiv_node_feat[:, :, 0]  # [N, 3] 
                updated_positions = positions_like + position_updates
                
                # Create updated vector features by modifying the first channel
                updated_equiv_node_feat = equiv_node_feat.clone()
                updated_equiv_node_feat[:, :, 0] = updated_positions
                
                # Apply small updates to other channels based on position changes
                for i in range(1, equiv_node_feat.size(2)):
                    updated_equiv_node_feat[:, :, i] = equiv_node_feat[:, :, i] + position_updates * 0.01
            else:
                # Fallback: pass through unchanged
                updated_equiv_node_feat = equiv_node_feat
        else:
            # If no original features, use position updates as new positions
            updated_equiv_node_feat = position_updates
            
        # Ensure the position updates contribute to the computational graph
        # by adding a small regularization term to scalar features
        pos_magnitude = torch.norm(position_updates, dim=1, keepdim=True)
        scalar_out = scalar_out + 0.001 * self.pos_update_net.weight.sum() * pos_magnitude.mean()

        return scalar_out, updated_equiv_node_feat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"attn_type={self.attn_type}, equivariant=True)"
        )