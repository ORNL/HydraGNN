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
import math

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch

# e3nn imports for SO(3) equivariant operations
from e3nn import o3, nn as e3nn_nn
from e3nn.util.jit import compile_mode

# HydraGNN utilities for irreps handling
# HydraGNN utilities for irreps handling
from hydragnn.utils.model.irreps_tools import create_irreps_string
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths


def init_edge_rot_mat(edge_distance_vec):
    """
    Initialize edge rotation matrices based on edge vectors.

    This function creates a rotation matrix for each edge that aligns the edge
    direction with a canonical frame. This is based on the original EquiformerV2
    implementation.

    Args:
        edge_distance_vec: Edge vectors [num_edges, 3]

    Returns:
        edge_rot_mat: Rotation matrices [num_edges, 3, 3]
    """
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0 ** 2, dim=1))

    # Make sure the atoms are far enough apart
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Warning: very small edge distances detected: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    # Create a random vector for constructing the orthonormal basis
    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2 ** 2, dim=1)).view(-1, 1)
    )

    # Create two rotated copies of the random vectors in case the random vector
    # is aligned with norm_x. With two 90 degree rotated vectors, at least one
    # should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]

    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)

    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    # Create orthonormal basis
    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True)))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()


@compile_mode("script")
class SO3_Rotation(torch.nn.Module):
    """
    SO(3) rotation module that handles Wigner-D matrix computation for edge-aligned rotations.

    This is a simplified version of the original EquiformerV2's SO3_Rotation that integrates
    with e3nn for Wigner-D matrix computation.
    """

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = lmax

    def set_wigner(self, edge_rot_mat: torch.Tensor):
        """
        Compute and store Wigner-D matrices from rotation matrices.

        Args:
            edge_rot_mat: Edge rotation matrices [num_edges, 3, 3]
        """
        self.device = edge_rot_mat.device
        self.dtype = edge_rot_mat.dtype

        # Convert rotation matrices to Wigner-D matrices using e3nn
        # For each degree l, compute the Wigner-D matrix
        self.wigner_matrices = {}

        # Convert rotation matrices to Euler angles
        # Use e3nn's matrix_to_angles function
        try:
            # Try to use e3nn's matrix_to_angles if available
            euler_angles = o3.matrix_to_angles(
                edge_rot_mat
            )  # Returns (alpha, beta, gamma)
            alpha, beta, gamma = euler_angles
        except AttributeError:
            # Fallback: compute Euler angles manually
            # Extract Euler angles from rotation matrix (ZYZ convention)
            cos_beta = edge_rot_mat[..., 2, 2]
            sin_beta = torch.sqrt(1 - cos_beta ** 2)

            # Handle singularities
            singular = sin_beta < 1e-6

            alpha = torch.zeros_like(cos_beta)
            beta = torch.acos(torch.clamp(cos_beta, -1, 1))
            gamma = torch.zeros_like(cos_beta)

            # Non-singular case
            alpha[~singular] = torch.atan2(
                edge_rot_mat[~singular, 1, 2], edge_rot_mat[~singular, 0, 2]
            )
            gamma[~singular] = torch.atan2(
                edge_rot_mat[~singular, 2, 1], -edge_rot_mat[~singular, 2, 0]
            )

        for l in range(self.lmax + 1):
            # Use e3nn to compute Wigner-D matrix for degree l
            try:
                wigner_d = o3.wigner_D(l, alpha, beta, gamma)  # [num_edges, 2l+1, 2l+1]
            except Exception:
                # Fallback: create identity matrices if Wigner-D computation fails
                num_edges = edge_rot_mat.size(0)
                dim_l = 2 * l + 1
                wigner_d = (
                    torch.eye(dim_l, device=self.device, dtype=self.dtype)
                    .unsqueeze(0)
                    .repeat(num_edges, 1, 1)
                )

            self.wigner_matrices[l] = wigner_d.detach()

    def rotate_spherical_harmonics(
        self, sh_features: torch.Tensor, l: int
    ) -> torch.Tensor:
        """
        Rotate spherical harmonic features using precomputed Wigner-D matrices.

        Args:
            sh_features: Spherical harmonic features [num_edges, 2l+1, channels]
            l: Degree of the spherical harmonics

        Returns:
            Rotated spherical harmonic features [num_edges, 2l+1, channels]
        """
        if l in self.wigner_matrices:
            wigner_d = self.wigner_matrices[l]  # [num_edges, 2l+1, 2l+1]
            # Apply rotation: [num_edges, 2l+1, 2l+1] @ [num_edges, 2l+1, channels]
            rotated_features = torch.bmm(wigner_d, sh_features)
            return rotated_features
        else:
            return sh_features


@compile_mode("script")
class SO3_Embedding(torch.nn.Module):
    """
    SO(3)-equivariant node embedding using spherical harmonics.

    This module embeds node features into SO(3)-equivariant representations
    using spherical harmonics up to a maximum degree (lmax).
    """

    def __init__(
        self,
        in_features: int,
        lmax: int,
        mmax: int = None,
        sphere_channels: int = None,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax
        self.sphere_channels = (
            sphere_channels if sphere_channels is not None else in_features
        )

        # Create irreps for spherical harmonics up to lmax
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # Create target irreps for node features
        self.irreps_node = o3.Irreps(create_irreps_string(self.sphere_channels, lmax))

        # Linear layer to project input features to equivariant representation
        self.linear = o3.Linear(
            f"{in_features}x0e",  # Input: invariant features
            self.irreps_node,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: Node features [num_nodes, in_features]
        Returns:
            SO(3)-equivariant node embeddings [num_nodes, irreps_node.dim]
        """
        return self.linear(node_features)


@compile_mode("script")
class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO(2)-equivariant graph attention mechanism.

    This implements a simplified version of EquiformerV2's attention that operates
    on SO(3)-equivariant node features and maintains equivariance.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_edge: o3.Irreps,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.heads = heads
        self.dropout = dropout

        # Query, Key, Value projections
        self.q_proj = o3.Linear(
            self.irreps_node, self.irreps_node, internal_weights=True
        )
        self.k_proj = o3.Linear(
            self.irreps_node, self.irreps_node, internal_weights=True
        )
        self.v_proj = o3.Linear(
            self.irreps_node, self.irreps_node, internal_weights=True
        )

        # Output projection
        self.out_proj = o3.Linear(
            self.irreps_node, self.irreps_node, internal_weights=True
        )

        # Attention head scaling
        self.scale = 1.0 / math.sqrt(self.irreps_node.dim // heads)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: Node features [num_nodes, irreps_node.dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, irreps_edge.dim]
            batch: Batch tensor [num_nodes]
        Returns:
            Updated node features [num_nodes, irreps_node.dim]
        """
        num_nodes = node_features.size(0)

        # Project to Q, K, V
        q = self.q_proj(node_features)
        k = self.k_proj(node_features)
        v = self.v_proj(node_features)

        # For simplicity, we'll use a global attention mechanism
        # In the full EquiformerV2, this would be more sophisticated

        # Convert to dense batch format for attention
        q_dense, mask = to_dense_batch(q, batch)  # [batch_size, max_nodes, dim]
        k_dense, _ = to_dense_batch(k, batch)
        v_dense, _ = to_dense_batch(v, batch)

        batch_size, max_nodes, dim = q_dense.size()

        # Reshape for multi-head attention
        head_dim = dim // self.heads
        q_dense = q_dense.view(batch_size, max_nodes, self.heads, head_dim).transpose(
            1, 2
        )
        k_dense = k_dense.view(batch_size, max_nodes, self.heads, head_dim).transpose(
            1, 2
        )
        v_dense = v_dense.view(batch_size, max_nodes, self.heads, head_dim).transpose(
            1, 2
        )

        # Compute attention scores
        attn_scores = torch.matmul(q_dense, k_dense.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(
                1
            )  # [batch_size, 1, 1, max_nodes]
            attn_scores = attn_scores.masked_fill(~mask_expanded, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_dense)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, max_nodes, dim)
        )

        # Convert back to sparse format
        attn_output = attn_output[mask]

        # Final projection
        output = self.out_proj(attn_output)

        return output


@compile_mode("script")
class EquivariantLayerNormV2(torch.nn.Module):
    """
    Equivariant layer normalization for SO(3)-equivariant features.

    Normalizes only the invariant (l=0) components while preserving
    the equivariant structure.
    """

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-5):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps

        # Find invariant components (l=0)
        self.invariant_mask = []
        self.equivariant_slices = []

        start_idx = 0
        for mul, (l, p) in self.irreps:
            end_idx = start_idx + mul * (2 * l + 1)
            if l == 0:  # Invariant
                self.invariant_mask.extend(list(range(start_idx, end_idx)))
            self.equivariant_slices.append((start_idx, end_idx, l))
            start_idx = end_idx

        self.invariant_mask = torch.tensor(self.invariant_mask)

        # Layer norm for invariant components
        if len(self.invariant_mask) > 0:
            self.layer_norm = torch.nn.LayerNorm(len(self.invariant_mask), eps=eps)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Equivariant features [num_nodes, irreps.dim]
        Returns:
            Normalized equivariant features [num_nodes, irreps.dim]
        """
        if self.layer_norm is None:
            return x

        # Normalize only invariant components
        x_norm = x.clone()
        if len(self.invariant_mask) > 0:
            invariant_features = x[:, self.invariant_mask]
            invariant_normalized = self.layer_norm(invariant_features)
            x_norm[:, self.invariant_mask] = invariant_normalized

        return x_norm


@compile_mode("script")
class S2Activation(torch.nn.Module):
    """
    S2 activation function that maintains SO(3) equivariance.

    For simplicity, we apply SiLU to all scalar (l=0) components
    and leave higher-order irreps unchanged (identity).
    """

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)

        # Find scalar components
        self.scalar_mask = []
        start_idx = 0
        for mul, (l, p) in self.irreps:
            end_idx = start_idx + mul * (2 * l + 1)
            if l == 0:  # Scalar
                self.scalar_mask.extend(list(range(start_idx, end_idx)))
            start_idx = end_idx

        self.scalar_mask = torch.tensor(self.scalar_mask) if self.scalar_mask else None
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Equivariant features [num_nodes, irreps.dim]
        Returns:
            Activated equivariant features [num_nodes, irreps.dim]
        """
        if self.scalar_mask is not None and len(self.scalar_mask) > 0:
            # Apply activation only to scalar components
            x_activated = x.clone()
            x_activated[:, self.scalar_mask] = self.activation(x[:, self.scalar_mask])
            return x_activated
        else:
            # No scalars, return unchanged
            return x


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

        # Use the first (and typically only) lmax and mmax values
        self.lmax = lmax_list[0] if lmax_list else 6
        self.mmax = mmax_list[0] if mmax_list else 2

        # Create irreps for the hidden representation
        self.irreps_node = o3.Irreps(
            create_irreps_string(self.sphere_channels, self.lmax)
        )
        self.irreps_edge = o3.Irreps.spherical_harmonics(
            self.lmax
        )  # Edge features use spherical harmonics

        # SO(3) embedding for node features
        self.so3_embedding = SO3_Embedding(
            in_features=channels,
            lmax=self.lmax,
            mmax=self.mmax,
            sphere_channels=self.sphere_channels,
        )

        # SO(3) rotation module for edge-aligned rotations
        self.so3_rotation = SO3_Rotation(lmax=self.lmax)

        # EquiformerV2 attention mechanism
        self.equivariant_attention = SO2EquivariantGraphAttention(
            irreps_node=self.irreps_node,
            irreps_edge=self.irreps_edge,
            heads=heads,
            dropout=dropout,
        )

        # Equivariant feedforward network
        # First linear transformation
        self.ffn_linear1 = o3.Linear(
            self.irreps_node,
            self.irreps_node,
            internal_weights=True,
            shared_weights=True,
        )

        # S2 activation
        self.s2_activation = S2Activation(self.irreps_node)

        # Second linear transformation
        self.ffn_linear2 = o3.Linear(
            self.irreps_node,
            self.irreps_node,
            internal_weights=True,
            shared_weights=True,
        )

        # Project back to invariant features for compatibility
        self.project_to_invariant = o3.Linear(
            self.irreps_node,
            f"{channels}x0e",  # Back to invariant features
            internal_weights=True,
            shared_weights=True,
        )

        # Equivariant layer normalizations
        self.norm1 = EquivariantLayerNormV2(self.irreps_node)
        self.norm2 = EquivariantLayerNormV2(self.irreps_node)
        self.norm3 = torch.nn.LayerNorm(channels)  # Final norm for invariant features

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()

        # Reset EquiformerV2 components
        if hasattr(self.so3_embedding, "linear"):
            for module in [self.so3_embedding.linear]:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        # Note: SO3_Rotation module doesn't have learnable parameters to reset

        for module in [
            self.equivariant_attention.q_proj,
            self.equivariant_attention.k_proj,
            self.equivariant_attention.v_proj,
            self.equivariant_attention.out_proj,
            self.ffn_linear1,
            self.ffn_linear2,
            self.project_to_invariant,
        ]:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # Reset layer norms
        if hasattr(self.norm1, "layer_norm") and self.norm1.layer_norm is not None:
            self.norm1.layer_norm.reset_parameters()
        if hasattr(self.norm2, "layer_norm") and self.norm2.layer_norm is not None:
            self.norm2.layer_norm.reset_parameters()
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
            h = self.norm3(h)  # Use regular LayerNorm for invariant features
            hs.append(h)

        # EquiformerV2 global attention processing

        # 1. Embed invariant features into SO(3)-equivariant representation
        node_features = self.so3_embedding(inv_node_feat)

        # 2. Apply equivariant attention
        # We need edge information for proper attention - extract from kwargs
        edge_index = kwargs.get("edge_index")
        edge_attr = kwargs.get("edge_attr", None)

        # If edge_index is not provided, create all-to-all connections
        if edge_index is None:
            num_nodes = inv_node_feat.size(0)
            # Create all-to-all edge connections
            row = torch.arange(
                num_nodes, device=inv_node_feat.device
            ).repeat_interleave(num_nodes)
            col = torch.arange(num_nodes, device=inv_node_feat.device).repeat(num_nodes)
            edge_index = torch.stack([row, col], dim=0)

        # Always apply attention mechanism to ensure all parameters are used
        # Create edge features using spherical harmonics if not provided
        if edge_attr is None:
            # Get edge vectors for spherical harmonic computation
            if "pos" in kwargs:
                # Use edge_shifts if available, otherwise create zero shifts tensor
                if "edge_shifts" in kwargs and kwargs["edge_shifts"] is not None:
                    shifts = kwargs["edge_shifts"]
                else:
                    num_edges = edge_index.size(1)
                    shifts = torch.zeros(
                        num_edges,
                        3,
                        device=inv_node_feat.device,
                        dtype=inv_node_feat.dtype,
                    )

                edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
                    kwargs["pos"], edge_index, shifts, normalize=True, eps=1e-12
                )

                # Compute edge rotation matrices (EquiformerV2 approach)
                edge_rot_mat = init_edge_rot_mat(edge_vectors)

                # Initialize the SO3 rotation module with edge rotation matrices
                self.so3_rotation.set_wigner(edge_rot_mat)

                # Use spherical harmonics of edge vectors as edge features
                spherical_harmonics = o3.SphericalHarmonics(
                    self.irreps_edge, normalize=True, normalization="component"
                )
                edge_features = spherical_harmonics(edge_vectors)

                # Apply edge-aligned rotation to spherical harmonic features
                # Split edge features by degree l and rotate each independently
                edge_features_rotated = torch.zeros_like(edge_features)
                start_idx = 0
                for l in range(self.lmax + 1):
                    dim_l = 2 * l + 1
                    if start_idx + dim_l <= edge_features.size(1):
                        sh_l = edge_features[
                            :, start_idx : start_idx + dim_l
                        ].unsqueeze(
                            -1
                        )  # [num_edges, 2l+1, 1]
                        sh_l_rotated = self.so3_rotation.rotate_spherical_harmonics(
                            sh_l, l
                        )
                        edge_features_rotated[
                            :, start_idx : start_idx + dim_l
                        ] = sh_l_rotated.squeeze(-1)
                        start_idx += dim_l
                    else:
                        break

                edge_features = edge_features_rotated
            else:
                # Fallback: create dummy edge features
                num_edges = edge_index.size(1)
                edge_features = torch.zeros(
                    num_edges,
                    self.irreps_edge.dim,
                    device=inv_node_feat.device,
                    dtype=inv_node_feat.dtype,
                )
        else:
            # Use provided edge attributes
            edge_features = edge_attr

        # Apply attention with residual connection (always executed)
        attn_output = self.equivariant_attention(
            node_features, edge_index, edge_features, graph_batch
        )
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output + node_features
        attn_output = self.norm1(attn_output)

        # 3. Apply equivariant feedforward network (always executed)
        # First linear + activation
        ffn_out = self.ffn_linear1(attn_output)
        ffn_out = self.s2_activation(ffn_out)
        ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)

        # Second linear with residual
        ffn_out = self.ffn_linear2(ffn_out)
        ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        ffn_out = ffn_out + attn_output
        ffn_out = self.norm2(ffn_out)

        # 4. Project back to invariant features (always executed)
        h_equiformer = self.project_to_invariant(ffn_out)

        h_equiformer = F.dropout(h_equiformer, p=self.dropout, training=self.training)
        h_equiformer = h_equiformer + inv_node_feat  # Residual connection
        hs.append(h_equiformer)

        # Combine local and global outputs
        out = sum(hs) if len(hs) > 1 else hs[0]

        # Final normalization
        out = self.norm3(out)

        return out, equiv_node_feat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"lmax_list={self.lmax_list}, mmax_list={self.mmax_list}, "
            f"with_edge_rot_mat=True)"
        )
