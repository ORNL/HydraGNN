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
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ModuleDict, Sequential
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import to_dense_batch

from .structural import StructuralAttentionContext


class StructuralCoordinatePerformerAttention(torch.nn.Module):
    """Performer attention with a low-rank squared-distance logit bias.

    Content Q/K/V are projected normally. Before the positive random-feature
    map, structural coordinates are appended so their dot product contributes
    ``a * ||x_i - x_j||^2`` up to a query-only term. The normalizer is based on
    the original head width, not the augmented width.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        coordinate_dim: int,
        head_channels: Optional[int] = 64,
        qkv_bias: bool = False,
        attn_out_bias: bool = True,
        dropout: float = 0.0,
        num_random_features: Optional[int] = None,
        feature_epsilon: float = 1.0e-6,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("Performer channels must be divisible by heads.")
        if head_channels is None:
            head_channels = channels // heads
        if int(coordinate_dim) <= 0:
            raise ValueError("coordinate_dim must be positive.")

        self.heads = int(heads)
        self.head_channels = int(head_channels)
        self.coordinate_dim = int(coordinate_dim)
        self.augmented_channels = (
            self.head_channels + self.coordinate_dim + 1
        )
        if num_random_features is None:
            num_random_features = int(
                self.augmented_channels
                * math.log(max(2, self.augmented_channels))
            )
        self.num_random_features = max(1, int(num_random_features))
        self.feature_epsilon = float(feature_epsilon)

        inner_channels = self.head_channels * self.heads
        self.q = Linear(channels, inner_channels, bias=qkv_bias)
        self.k = Linear(channels, inner_channels, bias=qkv_bias)
        self.v = Linear(channels, inner_channels, bias=qkv_bias)
        self.attn_out = Linear(inner_channels, channels, bias=attn_out_bias)
        self.dropout = Dropout(dropout)
        self.register_buffer(
            "projection_matrix",
            torch.empty(self.num_random_features, self.augmented_channels),
        )
        self.redraw_projection_matrix()

    def _augment_qk(self, q, k, structural_coordinates, coefficient):
        """Append structural factors and apply the original-width scaling."""

        if q.shape != k.shape or q.dim() != 4:
            raise ValueError("Projected Q and K must have matching [B,H,N,D] shapes.")
        batch_size, heads, num_nodes, head_channels = q.shape
        if heads != self.heads or head_channels != self.head_channels:
            raise ValueError(
                "Projected Q/K shape does not match the configured Performer heads."
            )
        expected = (batch_size, num_nodes, self.coordinate_dim)
        if tuple(structural_coordinates.shape) != expected:
            raise ValueError(
                f"Expected structural coordinates with shape {expected}, got "
                f"{tuple(structural_coordinates.shape)}."
            )

        coordinates = structural_coordinates.to(device=q.device, dtype=q.dtype)
        coordinates = coordinates.unsqueeze(1).expand(-1, heads, -1, -1)
        squared_norm = coordinates.square().sum(dim=-1, keepdim=True)
        ones = torch.ones_like(squared_norm)
        coefficient = torch.as_tensor(
            coefficient, device=q.device, dtype=q.dtype
        )
        bias_scale = math.sqrt(self.head_channels) * coefficient

        q_augmented = torch.cat((q, coordinates, ones), dim=-1)
        k_augmented = torch.cat(
            (
                k,
                -2.0 * bias_scale * coordinates,
                bias_scale * squared_norm,
            ),
            dim=-1,
        )

        # Scaling each side by d^(-1/4) makes their dot product use 1/sqrt(d),
        # where d is the original head width rather than d + rank + 1.
        normalizer = self.head_channels ** -0.25
        return q_augmented * normalizer, k_augmented * normalizer

    def _positive_features(self, value, is_query):
        """Positive FAVOR-style features for the softmax dot-product kernel."""

        value = value.float()
        projection = self.projection_matrix.float()
        projected = torch.einsum("bhnd,md->bhnm", value, projection)
        diagonal = 0.5 * value.square().sum(dim=-1, keepdim=True)
        if is_query:
            # A per-query factor cancels between numerator and denominator.
            stabilizer = projected.amax(dim=-1, keepdim=True).detach()
        else:
            # Key scaling must be common to all keys in one batch/head.
            stabilizer = projected.amax(dim=(-2, -1), keepdim=True).detach()
        ratio = self.num_random_features ** -0.5
        return ratio * (
            torch.exp(projected - diagonal - stabilizer)
            + self.feature_epsilon
        )

    @staticmethod
    def _linear_attention(q, k, v):
        key_sum = k.sum(dim=-2)
        denominator = torch.einsum("bhnm,bhm->bhn", q, key_sum)
        denominator = denominator.clamp_min(torch.finfo(q.dtype).tiny)
        key_value = torch.einsum("bhnm,bhnv->bhmv", k, v)
        output = torch.einsum("bhnm,bhmv->bhnv", q, key_value)
        return output / denominator.unsqueeze(-1)

    def forward(
        self,
        x,
        structural_coordinates,
        coefficient,
        mask=None,
    ):
        batch_size, num_nodes, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = map(
            lambda value: value.reshape(
                batch_size, num_nodes, self.heads, self.head_channels
            ).permute(0, 2, 1, 3),
            (q, k, v),
        )
        q, k = self._augment_qk(q, k, structural_coordinates, coefficient)
        q = self._positive_features(q, is_query=True)
        k = self._positive_features(k, is_query=False)
        v_dtype = v.dtype
        v = v.float()
        if mask is not None:
            valid = mask[:, None, :, None]
            k = k.masked_fill(~valid, 0.0)
            v = v.masked_fill(~valid, 0.0)
        out = self._linear_attention(q, k, v).to(v_dtype)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        return self.dropout(self.attn_out(out))

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.projection_matrix.normal_()

    def _reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.attn_out.reset_parameters()
        self.redraw_projection_matrix()


class EffectiveResistancePerformerAttention(
    StructuralCoordinatePerformerAttention
):
    """Compatibility wrapper for the original OPF-specific class name."""

    def __init__(self, *args, resistance_dim: int, **kwargs):
        super().__init__(*args, coordinate_dim=resistance_dim, **kwargs)


class HeteroGPSConv(torch.nn.Module):
    """GPS-style block for heterogeneous node dictionaries.

    This module mirrors the homogeneous GPS wrapper behavior:
    - local branch can update both invariant and equivariant channels;
    - global attention updates only invariant channels;
    - invariant local/global outputs are fused with residual MLP.

    For backward compatibility with invariant-only hetero models, the wrapper
    also accepts/returns invariant-only dictionaries.
    """

    def __init__(
        self,
        channels: int,
        metadata: tuple,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        attn_node_types: Optional[list[str]] = None,
        pe_dim: int = 0,
        direct_rpe_dim: int = 0,
        rpe_hidden_dim: int = 0,
        rpe_zero_diagonal: bool = True,
        resistance_qk_dim: int = 0,
        pairwise_feature_dim: Optional[int] = None,
        qk_coordinate_dim: Optional[int] = None,
    ):
        super().__init__()

        self.channels = channels
        self.node_types = list(metadata[0])
        if attn_node_types is None:
            self.attn_node_types = list(self.node_types)
        else:
            requested = set(attn_node_types)
            unknown = requested - set(self.node_types)

            if unknown:
                raise ValueError(
                    f"Unknown attention node types: {sorted(unknown)}"
                )

            self.attn_node_types = [
                node_type
                for node_type in self.node_types
                if node_type in requested
            ]

            if not self.attn_node_types:
                raise ValueError(
                    "HeteroGPSConv requires at least one attention node type."
                )
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type
        self.pe_dim = int(pe_dim)
        if pairwise_feature_dim is not None and direct_rpe_dim not in (
            0,
            pairwise_feature_dim,
        ):
            raise ValueError(
                "direct_rpe_dim and pairwise_feature_dim specify different widths."
            )
        if qk_coordinate_dim is not None and resistance_qk_dim not in (
            0,
            qk_coordinate_dim,
        ):
            raise ValueError(
                "resistance_qk_dim and qk_coordinate_dim specify different widths."
            )
        self.pairwise_feature_dim = int(
            direct_rpe_dim if pairwise_feature_dim is None else pairwise_feature_dim
        )
        self.qk_coordinate_dim = int(
            resistance_qk_dim
            if qk_coordinate_dim is None
            else qk_coordinate_dim
        )
        # Compatibility attributes for the original experimental API.
        self.direct_rpe_dim = self.pairwise_feature_dim
        self.resistance_qk_dim = self.qk_coordinate_dim
        self.rpe_zero_diagonal = bool(rpe_zero_diagonal)
        active_rpe_count = sum(
            width > 0
            for width in (
                self.pe_dim,
                self.pairwise_feature_dim,
                self.qk_coordinate_dim,
            )
        )
        if active_rpe_count > 1:
            raise ValueError("Only one attention RPE can be active at a time.")
        if self.pairwise_feature_dim > 0 and len(self.attn_node_types) != 1:
            raise ValueError(
                "Direct pairwise features currently require exactly one "
                "attention node type."
            )
        if self.qk_coordinate_dim > 0:
            if attn_type != "performer":
                raise ValueError(
                    "Effective-resistance Q/K augmentation requires "
                    "attn_type='performer'."
                )
            if len(self.attn_node_types) != 1:
                raise ValueError(
                    "Structural Q/K augmentation requires exactly one attention "
                    "node type."
                )

        attn_kwargs = attn_kwargs or {}
        if attn_type == "multihead":
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == "performer":
            if self.qk_coordinate_dim > 0:
                self.attn = StructuralCoordinatePerformerAttention(
                    channels=channels,
                    heads=heads,
                    coordinate_dim=self.qk_coordinate_dim,
                    **attn_kwargs,
                )
            else:
                self.attn = PerformerAttention(
                    channels=channels,
                    heads=heads,
                    **attn_kwargs,
                )
        else:
            raise ValueError(f"{attn_type} is not supported")

        self.rpe_mlp = None
        rpe_input_dim = (
            3 * self.pe_dim if self.pe_dim > 0 else self.pairwise_feature_dim
        )
        if rpe_input_dim > 0:
            if attn_type != "multihead":
                raise ValueError(
                    "Attention RPE bias currently requires attn_type='multihead'."
                )
            hidden_dim = int(rpe_hidden_dim) if int(rpe_hidden_dim) > 0 else channels
            use_bias = not (
                self.pairwise_feature_dim > 0 and self.rpe_zero_diagonal
            )
            self.rpe_mlp = Sequential(
                Linear(rpe_input_dim, hidden_dim, bias=use_bias),
                activation_resolver(act, **(act_kwargs or {})),
                Linear(hidden_dim, heads, bias=use_bias),
            )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1_dict = ModuleDict(
            {
                node_type: normalization_resolver(norm, channels, **norm_kwargs)
                for node_type in self.node_types
            }
        )
        self.norm2_dict = ModuleDict(
            {
                node_type: normalization_resolver(norm, channels, **norm_kwargs)
                for node_type in self.node_types
            }
        )
        self.norm3_dict = ModuleDict(
            {
                node_type: normalization_resolver(norm, channels, **norm_kwargs)
                for node_type in self.node_types
            }
        )
        self._norm_with_batch = {}
        for node_type in self.node_types:
            norm_module = self.norm1_dict[node_type]
            if norm_module is None:
                self._norm_with_batch[node_type] = False
            else:
                signature = inspect.signature(norm_module.forward)
                self._norm_with_batch[node_type] = "batch" in signature.parameters

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        if self.rpe_mlp is not None:
            reset(self.rpe_mlp)
        reset(self.mlp)

        for norm_dict in (self.norm1_dict, self.norm2_dict, self.norm3_dict):
            for norm in norm_dict.values():
                if norm is not None:
                    norm.reset_parameters()

    def _pack_x_dict(self, x_dict, batch_dict):
        """Pack hetero node dictionaries into one token tensor for attention."""
        xs = []
        batches = []
        split_sizes = []
        pack_node_types = []

        for node_type in self.attn_node_types:
            if node_type not in x_dict:
                continue

            x = x_dict[node_type]
            if x.size(-1) != self.channels:
                raise ValueError(
                    f"Expected {self.channels} channels for node type "
                    f"'{node_type}', got {x.size(-1)}."
                )

            batch = batch_dict.get(node_type)
            if batch is None:
                batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
            else:
                batch = batch.to(device=x.device, dtype=torch.long)

            if batch.size(0) != x.size(0):
                raise ValueError(
                    f"Batch size for node type '{node_type}' does not match "
                    f"features: {batch.size(0)} vs {x.size(0)}."
                )

            xs.append(x)
            batches.append(batch)
            split_sizes.append(x.size(0))
            pack_node_types.append(node_type)

        if not xs:
            raise ValueError("HeteroGPSConv requires at least one node feature tensor.")

        x_all = torch.cat(xs, dim=0)
        batch_all = torch.cat(batches, dim=0)
        return x_all, batch_all, split_sizes, pack_node_types

    def _unpack_x_dict(self, x_all, split_sizes, pack_node_types):
        """Split packed attention output back into a hetero x_dict."""
        out_dict = {}
        start = 0
        for node_type, split_size in zip(pack_node_types, split_sizes):
            end = start + split_size
            out_dict[node_type] = x_all[start:end]
            start = end

        if start != x_all.size(0):
            raise ValueError(
                "Packed attention output size does not match hetero split sizes."
            )

        return out_dict

    def _apply_norm(self, norm_dict, node_type, x, batch):
        norm = norm_dict[node_type]
        if norm is None:
            return x
        if self._norm_with_batch.get(node_type, False):
            return norm(x, batch=batch)
        return norm(x)

    def _pack_svd_rpe_dict(self, svd_rpe_dict, pack_node_types):
        if svd_rpe_dict is None:
            return None
        packed = {}
        required = ("u_real", "u_imag", "v_real", "v_imag", "s")
        for name in required:
            values = []
            for node_type in pack_node_types:
                node_values = svd_rpe_dict.get(node_type)
                if node_values is None or name not in node_values:
                    raise ValueError(
                        f"SVD-RPE is missing '{name}' for attention node type "
                        f"'{node_type}'."
                    )
                value = node_values[name]
                if value.size(-1) != self.pe_dim:
                    raise ValueError(
                        f"Expected SVD-RPE width {self.pe_dim} for '{node_type}', "
                        f"got {value.size(-1)}."
                    )
                values.append(value)
            packed[name] = torch.cat(values, dim=0)
        return packed

    def _dense_direct_pairwise_rpe(
        self,
        direct_pairwise_rpe,
        mask,
        device,
        dtype,
    ):
        if direct_pairwise_rpe is None:
            raise ValueError(
                "Direct pairwise RPE is active, but no pair matrices were provided."
            )
        matrices = list(direct_pairwise_rpe)
        batch_size, max_nodes = mask.shape
        if len(matrices) != batch_size:
            raise ValueError(
                f"Expected {batch_size} pairwise RPE matrices, got {len(matrices)}."
            )
        counts = mask.sum(dim=1).detach().cpu().tolist()
        if batch_size == 1 and counts[0] == max_nodes:
            matrix = matrices[0].to(device=device, dtype=dtype)
            expected = (max_nodes, max_nodes, self.pairwise_feature_dim)
            if tuple(matrix.shape) != expected:
                raise ValueError(
                    f"Expected direct pairwise RPE shape {expected}, got "
                    f"{tuple(matrix.shape)}."
                )
            return matrix.unsqueeze(0)

        dense_rpe = torch.zeros(
            (batch_size, max_nodes, max_nodes, self.pairwise_feature_dim),
            device=device,
            dtype=dtype,
        )
        for graph_index, (matrix, count) in enumerate(zip(matrices, counts)):
            matrix = matrix.to(device=device, dtype=dtype)
            expected = (count, count, self.pairwise_feature_dim)
            if tuple(matrix.shape) != expected:
                raise ValueError(
                    f"Expected direct pairwise RPE graph {graph_index} shape "
                    f"{expected}, got {tuple(matrix.shape)}."
                )
            dense_rpe[graph_index, :count, :count] = matrix
        return dense_rpe

    def _apply_global_attention(
        self,
        x_dict,
        batch_dict,
        svd_rpe_dict=None,
        direct_pairwise_rpe=None,
        resistance_qk=None,
        resistance_coefficient=None,
        structural_context: Optional[StructuralAttentionContext] = None,
    ):
        """Apply all-node attention across every node type in each graph. Needs to be ordered by graph, original order restored after for unpacking."""
        (
            x_all,
            batch_all,
            split_sizes,
            pack_node_types,
        ) = self._pack_x_dict(x_dict, batch_dict)
        perm = torch.argsort(batch_all, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        x_sorted = x_all[perm]
        batch_sorted = batch_all[perm]
        dense, mask = to_dense_batch(x_sorted, batch_sorted)

        dense_resistance_qk = None
        if self.qk_coordinate_dim > 0:
            if resistance_qk is None or resistance_coefficient is None:
                raise ValueError(
                    "Effective-resistance Q/K attention requires coordinates "
                    "and a coefficient."
                )
            expected = (x_all.size(0), self.qk_coordinate_dim)
            if tuple(resistance_qk.shape) != expected:
                raise ValueError(
                    f"Expected packed resistance coordinates {expected}, got "
                    f"{tuple(resistance_qk.shape)}."
                )
            resistance_sorted = resistance_qk.to(
                device=x_sorted.device, dtype=x_sorted.dtype
            )[perm]
            dense_resistance_qk, resistance_mask = to_dense_batch(
                resistance_sorted, batch_sorted
            )
            if not torch.equal(mask, resistance_mask):
                raise RuntimeError(
                    "Resistance-coordinate batching mask does not match node mask."
                )

        attn_bias = None
        if self.rpe_mlp is not None:
            if self.pe_dim > 0:
                packed_svd = self._pack_svd_rpe_dict(
                    svd_rpe_dict, pack_node_types
                )
                if packed_svd is None:
                    raise ValueError(
                        "pe_dim is nonzero but the batch has no Ybus SVD-RPE "
                        "factors. Re-run OPF preprocessing with "
                        "pe_encoder='svd_ybus'."
                    )
                dense_svd = {}
                for name, value in packed_svd.items():
                    value_sorted = value.to(x_sorted.device)[perm]
                    dense_value, value_mask = to_dense_batch(
                        value_sorted, batch_sorted
                    )
                    if not torch.equal(mask, value_mask):
                        raise RuntimeError(
                            "SVD-RPE batching mask does not match node mask."
                        )
                    dense_svd[name] = dense_value

                # U_i * conjugate(V_j), expressed without complex tensors so it
                # remains compatible with mixed precision and serialization.
                ur = dense_svd["u_real"].unsqueeze(2)
                ui = dense_svd["u_imag"].unsqueeze(2)
                vr = dense_svd["v_real"].unsqueeze(1)
                vi = dense_svd["v_imag"].unsqueeze(1)
                cross_real = ur * vr + ui * vi
                cross_imag = ui * vr - ur * vi
                singular = dense_svd["s"].unsqueeze(2).expand_as(cross_real)
                pair_rpe = torch.cat((cross_real, cross_imag, singular), dim=-1)
            else:
                pair_rpe = self._dense_direct_pairwise_rpe(
                    direct_pairwise_rpe,
                    mask,
                    device=x_sorted.device,
                    dtype=x_sorted.dtype,
                )
            # [batch, heads, query, key] -> MultiheadAttention's 3-D mask.
            attn_bias = self.rpe_mlp(pair_rpe).permute(0, 3, 1, 2)
            attn_bias = attn_bias.reshape(
                -1, dense.size(1), dense.size(1)
            ).to(dtype=dense.dtype)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            key_padding_mask = ~mask
            if attn_bias is not None:
                # Match mask dtypes to avoid deprecated mixed bool/float masks.
                key_padding_mask = torch.zeros(
                    mask.shape, device=dense.device, dtype=dense.dtype
                ).masked_fill(~mask, float("-inf"))
            dense, _ = self.attn(
                dense,
                dense,
                dense,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_bias,
                need_weights=False,
            )
        elif isinstance(self.attn, StructuralCoordinatePerformerAttention):
            dense = self.attn(
                dense,
                dense_resistance_qk,
                resistance_coefficient,
                mask=mask,
            )
        elif isinstance(self.attn, PerformerAttention):
            dense = self.attn(dense, mask=mask)

        x_attn = dense[mask]
        x_attn = F.dropout(x_attn, p=self.dropout, training=self.training)
        x_attn = x_attn + x_sorted
        x_attn = x_attn[inv_perm]
        global_out = self._unpack_x_dict(x_attn, split_sizes, pack_node_types)

        for node_type, x in x_dict.items():
            if node_type in global_out:
                h = global_out[node_type]
            else:
                h = x

            global_out[node_type] = self._apply_norm(
                self.norm2_dict,
                node_type,
                h,
                batch_dict.get(node_type),
            )

        return global_out

    def _call_local_conv(
        self,
        inv_node_feat_dict,
        equiv_node_feat_dict,
        edge_index_dict,
        edge_attr_dict,
    ):
        """Call local conv with best-effort signature adaptation.

        Supports both:
        - equivariant hetero convs returning (inv_dict, equiv_dict), and
        - invariant hetero convs returning inv_dict only.
        """
        if self.conv is None:
            return None, equiv_node_feat_dict

        # Equivariant-style API (preferred).
        try:
            local_out = self.conv(
                inv_node_feat_dict=inv_node_feat_dict,
                equiv_node_feat_dict=equiv_node_feat_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
        except TypeError:
            # Invariant hetero API fallback.
            if edge_attr_dict is None:
                local_out = self.conv(inv_node_feat_dict, edge_index_dict)
            else:
                local_out = self.conv(
                    inv_node_feat_dict,
                    edge_index_dict,
                    edge_attr_dict=edge_attr_dict,
                )

        if isinstance(local_out, tuple) and len(local_out) == 2:
            local_inv, local_equiv = local_out
        else:
            local_inv = local_out
            local_equiv = equiv_node_feat_dict

        return local_inv, local_equiv

    def forward(
        self,
        x_dict=None,
        edge_index_dict=None,
        batch_dict=None,
        edge_attr_dict=None,
        inv_node_feat_dict=None,
        equiv_node_feat_dict=None,
        svd_rpe_dict=None,
        direct_pairwise_rpe=None,
        resistance_qk=None,
        resistance_coefficient=None,
    ):
        """Run one hetero local-message-passing + global-attention block.

        Accepts both legacy invariant-only arguments (x_dict, ...) and
        generalized equivariant arguments (inv_node_feat_dict,
        equiv_node_feat_dict, ...).
        """
        if structural_context is not None:
            if svd_rpe_dict is None:
                svd_rpe_dict = structural_context.factorized_pairwise_features
            if direct_pairwise_rpe is None:
                direct_pairwise_rpe = structural_context.pairwise_features
            if resistance_qk is None:
                resistance_qk = structural_context.qk_coordinates
            if resistance_coefficient is None:
                resistance_coefficient = structural_context.qk_coefficient
        if inv_node_feat_dict is None:
            inv_node_feat_dict = x_dict
        if inv_node_feat_dict is None:
            raise ValueError("HeteroGPSConv.forward requires invariant node features.")
        if edge_index_dict is None:
            raise ValueError("HeteroGPSConv.forward requires edge_index_dict.")

        if batch_dict is None:
            batch_dict = {
                node_type: torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                for node_type, x in inv_node_feat_dict.items()
            }

        local_out, local_equiv = self._call_local_conv(
            inv_node_feat_dict,
            equiv_node_feat_dict,
            edge_index_dict,
            edge_attr_dict,
        )

        local_out_norm = None
        if local_out is not None:
            local_out_norm = {}
            for node_type, x in inv_node_feat_dict.items():
                h = local_out.get(node_type)
                if h is None:
                    h = torch.zeros_like(x)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = h + x
                local_out_norm[node_type] = self._apply_norm(
                    self.norm1_dict,
                    node_type,
                    h,
                    batch_dict.get(node_type),
                )
            local_out = local_out_norm

        global_out = self._apply_global_attention(
            inv_node_feat_dict,
            batch_dict,
            svd_rpe_dict,
            direct_pairwise_rpe,
            resistance_qk,
            resistance_coefficient,
        )

        out = {}
        for node_type, x in inv_node_feat_dict.items():
            h = global_out[node_type]
            if local_out is not None:
                h = h + local_out[node_type]
            h = h + self.mlp(h)
            out[node_type] = self._apply_norm(
                self.norm3_dict,
                node_type,
                h,
                batch_dict.get(node_type),
            )

        return out, local_equiv

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"attn_type={self.attn_type})"
        )
