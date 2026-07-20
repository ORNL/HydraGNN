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
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ModuleDict, Sequential
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import to_dense_batch


class HeteroGPSConv(torch.nn.Module):
    """Sketch of a GPS-style block for heterogeneous node dictionaries.

    This file intentionally does not keep the homogeneous GPS forward signature.
    A hetero GPS block should operate on ``x_dict``/``batch_dict`` and return an
    updated ``x_dict``. The local branch should be a hetero message-passing
    module, and the global branch should pack all node types into a shared token
    sequence before attention.
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
    ):
        super().__init__()

        self.channels = channels
        self.node_types = list(metadata[0])
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == "multihead":
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == "performer":
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            raise ValueError(f"{attn_type} is not supported")

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

        for node_type in self.node_types:
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

    def _apply_global_attention(self, x_dict, batch_dict):
        """Apply all-node attention across every node type in each graph. Needs to be ordered by graph, original order restored after for unpacking."""
        (
            x_all,
            batch_all,
            split_sizes,
            pack_node_types,
        ) = self._pack_x_dict(x_dict, batch_dict)
        perm = torch.argsort(batch_all)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        x_sorted = x_all[perm]
        batch_sorted = batch_all[perm]
        dense, mask = to_dense_batch(x_sorted, batch_sorted)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            dense, _ = self.attn(
                dense,
                dense,
                dense,
                key_padding_mask=~mask,
                need_weights=False,
            )
        elif isinstance(self.attn, PerformerAttention):
            dense = self.attn(dense, mask=mask)

        x_attn = dense[mask]
        x_attn = F.dropout(x_attn, p=self.dropout, training=self.training)
        x_attn = x_attn + x_sorted
        x_attn = x_attn[inv_perm]
        global_out = self._unpack_x_dict(x_attn, split_sizes, pack_node_types)

        for node_type, x in global_out.items():
            global_out[node_type] = self._apply_norm(
                self.norm2_dict,
                node_type,
                x,
                batch_dict.get(node_type),
            )

        return global_out

    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict=None):
        """Run one hetero local-message-passing + global-attention block."""
        local_out = None
        if self.conv is not None:
            if edge_attr_dict is None:
                local_out = self.conv(x_dict, edge_index_dict)
            else:
                local_out = self.conv(
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict=edge_attr_dict,
                )

            local_out_norm = {}
            for node_type, x in x_dict.items():
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

        global_out = self._apply_global_attention(x_dict, batch_dict)

        out = {}
        for node_type, x in x_dict.items():
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

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"attn_type={self.attn_type})"
        )
