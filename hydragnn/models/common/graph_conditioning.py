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
"""Conditioning of node and pooled graph embeddings on ``data.graph_attr``.

Three modes are supported (``graph_attr_conditioning_mode``):

* ``"film"``: per-graph FiLM modulation of the invariant node channels.
* ``"concat_node"``: concatenate graph attributes to every node and project
  back to the channel width.
* ``"fuse_pool"``: concatenate graph attributes to the pooled graph embedding
  and project back to the channel width.
"""

import torch
from torch.nn import Linear, Sequential

GRAPH_ATTR_CONDITIONING_MODES = ("film", "concat_node", "fuse_pool")


def validate_graph_attr_conditioning_mode(mode: str) -> str:
    mode = mode.lower()
    if mode not in GRAPH_ATTR_CONDITIONING_MODES:
        raise ValueError(
            "graph_attr_conditioning_mode must be one of: 'film', 'concat_node', 'fuse_pool'."
        )
    return mode


def _to(module, first_linear, device, dtype):
    """Move ``module`` if ``first_linear`` is not on ``device``/``dtype``."""
    weight = first_linear.weight
    if weight.device != device or (dtype is not None and weight.dtype != dtype):
        module = module.to(device=device, dtype=dtype)
    return module


def _per_graph(graph_attr, num_graphs, what):
    """Reshape ``graph_attr`` to ``[num_graphs, feat_dim]``."""
    if graph_attr.dim() == 1:
        if graph_attr.numel() % num_graphs != 0:
            raise ValueError(
                f"One-dimensional graph_attr with numel={graph_attr.numel()} is not divisible by num_graphs={num_graphs}."
            )
        return graph_attr.view(num_graphs, graph_attr.numel() // num_graphs)
    if graph_attr.dim() == 2:
        if graph_attr.size(0) != num_graphs:
            raise ValueError(
                f"graph_attr first dim {graph_attr.size(0)} does not match {what}={num_graphs}."
            )
        return graph_attr
    raise ValueError(f"Unsupported graph_attr ndim={graph_attr.dim()}; expected 1/2.")


class GraphConditioningMixin:
    """Graph-attribute conditioning for models with ``hidden_dim`` channels.

    Requires ``hidden_dim``, ``activation_function``,
    ``use_graph_attr_conditioning`` and ``graph_attr_conditioning_mode``.
    Conditioning modules are created lazily on first use; subclasses that
    build them eagerly override the ``_ensure_*`` methods.
    """

    def _init_graph_conditioning_modules(self):
        self.graph_conditioner = None
        self.graph_concat_projector = None
        self.graph_concat_projector_in_dim = None
        self.graph_pool_projector = None
        self.graph_pool_projector_in_dim = None

    def _build_graph_conditioner(self, graph_attr_dim: int):
        hidden = max(self.hidden_dim, graph_attr_dim)
        self.graph_conditioner = Sequential(
            Linear(graph_attr_dim, hidden),
            self.activation_function,
            Linear(hidden, 2 * self.hidden_dim),
        )

    def _build_graph_concat_projector(self, graph_attr_dim: int, channel_dim: int):
        in_dim = channel_dim + graph_attr_dim
        self.graph_concat_projector = Linear(in_dim, channel_dim)
        self.graph_concat_projector_in_dim = in_dim

    def _build_graph_pool_projector(self, graph_attr_dim: int, channel_dim: int):
        in_dim = channel_dim + graph_attr_dim
        self.graph_pool_projector = Sequential(
            Linear(in_dim, channel_dim),
            self.activation_function,
            Linear(channel_dim, channel_dim),
        )
        self.graph_pool_projector_in_dim = in_dim

    def _ensure_graph_conditioner(self, graph_attr_dim: int, device):
        """Instantiate (or move) the FiLM conditioner."""
        if self.graph_conditioner is None:
            self._build_graph_conditioner(graph_attr_dim)
        self.graph_conditioner = _to(
            self.graph_conditioner, self.graph_conditioner[0], device, None
        )

    def _ensure_graph_concat_projector(
        self, graph_attr_dim: int, channel_dim: int, device, dtype=None
    ):
        """Instantiate (or move) the projector used by ``concat_node``."""
        if self.graph_concat_projector_in_dim != channel_dim + graph_attr_dim:
            self._build_graph_concat_projector(graph_attr_dim, channel_dim)
        self.graph_concat_projector = _to(
            self.graph_concat_projector, self.graph_concat_projector, device, dtype
        )

    def _ensure_graph_pool_projector(
        self, graph_attr_dim: int, channel_dim: int, device, dtype=None
    ):
        """Instantiate (or move) the projector used by ``fuse_pool``."""
        if self.graph_pool_projector_in_dim != channel_dim + graph_attr_dim:
            self._build_graph_pool_projector(graph_attr_dim, channel_dim)
        self.graph_pool_projector = _to(
            self.graph_pool_projector, self.graph_pool_projector[0], device, dtype
        )

    def _graph_attr(self, data):
        graph_attr = getattr(data, "graph_attr", None)
        if graph_attr is None:
            raise ValueError(
                "use_graph_attr_conditioning=True but data.graph_attr is missing."
            )
        return graph_attr

    def _graph_attr_per_graph(self, graph_attr, batch):
        """Reshape ``graph_attr`` to one row per graph referenced by ``batch``."""
        return _per_graph(graph_attr, int(batch.max().item() + 1), "num_graphs")

    def _apply_graph_conditioning(self, inv_node_feat, batch, data):
        """Apply graph_attr conditioning (FiLM, concat_node, or defer to fuse_pool) to invariant node channels."""
        if not self.use_graph_attr_conditioning:
            return inv_node_feat

        graph_attr = self._graph_attr(data).to(
            device=inv_node_feat.device, dtype=inv_node_feat.dtype
        )
        # Batch can be None for single-graph inputs.
        if batch is None:
            batch = torch.zeros(
                inv_node_feat.size(0), device=inv_node_feat.device, dtype=torch.long
            )
        graph_attr = self._graph_attr_per_graph(graph_attr, batch)
        mode = self.graph_attr_conditioning_mode

        if mode == "film":
            self._ensure_graph_conditioner(graph_attr.size(-1), inv_node_feat.device)

            # FiLM: inv = (1 + scale) * inv + shift, scale/shift are per-graph then broadcast by batch.
            scale, shift = self.graph_conditioner(graph_attr).split(
                self.hidden_dim, dim=-1
            )
            scale = torch.tanh(scale)  # keep modulation bounded

            channel_dim = inv_node_feat.size(-1)
            scale_b = scale[batch]
            shift_b = shift[batch]
            if channel_dim != self.hidden_dim:
                if channel_dim % self.hidden_dim != 0:
                    raise ValueError(
                        f"Graph conditioning expects channels divisible by hidden_dim (got {channel_dim} vs {self.hidden_dim})."
                    )
                factor = channel_dim // self.hidden_dim
                scale_b = scale_b.repeat_interleave(factor, dim=-1)
                shift_b = shift_b.repeat_interleave(factor, dim=-1)
            return inv_node_feat * (1 + scale_b) + shift_b

        if mode == "concat_node":
            self._ensure_graph_concat_projector(
                graph_attr_dim=graph_attr.size(-1),
                channel_dim=inv_node_feat.size(-1),
                device=inv_node_feat.device,
                dtype=inv_node_feat.dtype,
            )
            graph_attr_b = graph_attr[batch]
            if inv_node_feat.size(0) != graph_attr_b.size(0):
                raise RuntimeError(
                    f"Graph conditioning dimension mismatch: inv_node_feat has "
                    f"{inv_node_feat.size(0)} rows but batch has {graph_attr_b.size(0)} "
                    f"nodes. This typically means the MPNN dropped isolated nodes "
                    f"(nodes with no edges within the radius cutoff). "
                    f"inv_node_feat.shape={list(inv_node_feat.shape)}, "
                    f"graph_attr_b.shape={list(graph_attr_b.shape)}"
                )
            fused = torch.cat([inv_node_feat, graph_attr_b], dim=-1)
            return self.graph_concat_projector(fused)

        if mode == "fuse_pool":
            # Pool-level fusion is applied later; leave node features unchanged here.
            return inv_node_feat

        raise ValueError(f"Unsupported graph_attr_conditioning_mode: {mode}")

    def _apply_graph_pool_conditioning(self, x_graph, data):
        """Fuse pooled graph embedding with graph_attr when conditioning_mode='fuse_pool'."""
        if not self.use_graph_attr_conditioning:
            return x_graph
        if self.graph_attr_conditioning_mode != "fuse_pool":
            return x_graph

        graph_attr = _per_graph(
            self._graph_attr(data), x_graph.size(0), "pooled num_graphs"
        )
        # Preserve the model/input dtype to avoid mixed precision mismatches under fp64 runs.
        graph_attr = graph_attr.to(device=x_graph.device, dtype=x_graph.dtype)
        self._ensure_graph_pool_projector(
            graph_attr_dim=graph_attr.size(-1),
            channel_dim=x_graph.size(-1),
            device=x_graph.device,
            dtype=x_graph.dtype,
        )
        fused = torch.cat([x_graph, graph_attr], dim=-1)
        return self.graph_pool_projector(fused)
