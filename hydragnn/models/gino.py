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
# Portions ported from neuraloperator/neuraloperator@a6648eeeb60fe0837f2d7d021517a945a159a0c8
# (neuralop/models/gino.py, neuralop/layers/{gno_block,integral_transform,
# embeddings,neighbor_search}.py), Copyright (c) 2023 NeuralOperator
# developers, MIT License; see LICENSES/NEURALOP-MIT.txt.
"""GINO: Geometry-Informed Neural Operator (Li et al., NeurIPS 2023).

Port of ``neuralop.models.GINO`` with the default ``linear`` transforms,
``transformer`` positional embeddings and no weighting function:

1. ``gno_in``: integral transform from the input point cloud (``pos``,
   features ``x``) onto a regular latent grid, mean over the points within
   ``in_radius`` of each grid point,
   ``h(g) = mean_y k(e(y), e(g)) * x(y)`` with ``k`` a ``LinearChannelMLP``
   and ``e`` the sinusoidal embedding;
2. lifting ``ChannelMLP`` and ``num_conv_layers`` FNO blocks on the grid
   (no grid embedding, no padding, as upstream);
3. ``gno_out``: integral transform from the grid onto query points, sum over
   grid points within ``out_radius``, ``k(e(g), e(q)) * h(g)``;
4. projection ``ChannelMLP(hidden -> ratio*hidden -> out)``, one per head.

Neighbour search is a dense ``cdist`` with ``dist <= radius`` (upstream's
native search). Upstream shares one geometry across a batch; here every graph
has its own point cloud and query points, and all graphs share the latent grid
``linspace(box_min, box_max, latent_resolution)`` per dim. Node heads query at
``pos``; query heads query at ``query_pos`` (graph ids from ``query_cell``,
see :func:`hydragnn.data.meshes.query_batch`, or ``query_graph``).
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.fno import ChannelMLP, FNOBlocks, LinearChannelMLP
from hydragnn.models.fno import resolve_activation
from hydragnn.models.operator_base import OperatorBase


class SinusoidalEmbedding(nn.Module):
    """``transformer`` embedding: interleaved ``sin/cos(x_i * w_k)``,
    ``w_k = (1 / max_positions) ** (2 k / L)``."""

    def __init__(self, in_channels, num_frequencies, max_positions=10000):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.max_positions = max_positions

    @property
    def out_channels(self):
        return 2 * self.num_frequencies * self.in_channels

    def forward(self, x):
        k = torch.arange(self.num_frequencies, device=x.device, dtype=x.dtype)
        freqs = (1.0 / self.max_positions) ** (k / self.num_frequencies * 2)
        angles = x.unsqueeze(-1) * freqs
        return torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(x.shape[0], -1)


def radius_neighbors(queries, data, radius, query_graph=None, data_graph=None):
    """Pairs ``(query, data)`` with ``|q - y| <= radius`` (and equal graphs).

    Returns ``(query_index, data_index)`` sorted by query, as upstream's CSR.
    """
    within = torch.cdist(queries, data) <= radius
    if query_graph is not None:
        within &= query_graph.unsqueeze(1) == data_graph.unsqueeze(0)
    q, y = within.nonzero(as_tuple=True)
    return q, y


class GNOBlock(nn.Module):
    """Linear kernel integral transform ``sum/mean_y k(e(y), e(x)) f(y)``."""

    def __init__(
        self,
        out_channels,
        coord_dim,
        radius,
        reduction,
        embed_channels,
        embed_max_positions,
        hidden_layers,
        act,
    ):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.radius = radius
        self.reduction = reduction
        self.pos_embedding = SinusoidalEmbedding(
            coord_dim, embed_channels, embed_max_positions
        )
        layers = [2 * self.pos_embedding.out_channels, *hidden_layers, out_channels]
        self.channel_mlp = LinearChannelMLP(layers, act)

    def forward(self, y, x, f_y, x_index, y_index):
        """``x_index``/``y_index`` list neighbour pairs; output has ``len(x)`` rows."""
        kernel = self.channel_mlp(
            torch.cat(
                [self.pos_embedding(y)[y_index], self.pos_embedding(x)[x_index]],
                dim=-1,
            )
        )
        values = kernel * f_y[y_index]
        out = values.new_zeros(x.shape[0], values.shape[1])
        out = out.index_add(0, x_index, values)
        if self.reduction == "mean":
            count = torch.bincount(x_index, minlength=x.shape[0]).clamp(min=1)
            out = out / count.unsqueeze(1).to(out.dtype)
        return out


def _graph_ids(data):
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(data.num_nodes, dtype=torch.long, device=data.pos.device)
    return batch


def _query_graph_ids(data, owner="GINO"):
    graph = getattr(data, "query_graph", None)
    if graph is not None:
        return graph
    if getattr(data, "query_cell", None) is not None:
        from hydragnn.data.meshes import query_batch

        return query_batch(data)
    if getattr(data, "batch", None) is None:
        return torch.zeros(
            data.query_pos.shape[0], dtype=torch.long, device=data.pos.device
        )
    raise ValueError(f"{owner} query heads need query_cell or query_graph on batches")


class GINO(OperatorBase):
    supported_output_levels = frozenset({"node", "query"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_layers,
        coord_dim=2,
        n_modes=(16, 16),
        latent_resolution=(32, 32),
        box_min=0.0,
        box_max=1.0,
        in_radius=0.033,
        out_radius=0.033,
        fno_in_channels=None,
        embed_channels=32,
        embed_max_positions=10000,
        in_channel_mlp_hidden_layers=(80, 80, 80),
        out_channel_mlp_hidden_layers=(512, 256),
        lifting_channel_ratio=2,
        projection_channel_ratio=4,
        channel_mlp_expansion=0.5,
        activation_function="gelu",
        loss_function_type="mse",
        loss_weights=None,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        n_modes = [int(m) for m in n_modes]
        latent_resolution = [int(r) for r in latent_resolution]
        if len(n_modes) != coord_dim or len(latent_resolution) != coord_dim:
            raise ValueError(
                "gino_n_modes and gino_latent_resolution need gino_coord_dim entries"
            )
        if coord_dim not in (2, 3):
            raise ValueError("GINO supports gino_coord_dim 2 or 3")
        if num_layers < 1:
            raise ValueError("GINO needs num_conv_layers >= 1")
        if in_radius <= 0 or out_radius <= 0:
            raise ValueError("GINO radii must be positive")
        # The linear kernel k(y, x) * f(y) is elementwise, so its width must
        # broadcast against the input features (upstream has the same rule).
        if fno_in_channels is None:
            fno_in_channels = input_dim
        if input_dim not in (1, fno_in_channels):
            raise ValueError(
                "gino_fno_in_channels must equal the input feature width "
                f"({input_dim}) unless that width is 1"
            )
        self.coord_dim = coord_dim
        self.latent_resolution = latent_resolution
        self.hidden_dim = hidden_dim
        box_min = [float(box_min)] * coord_dim if _scalar(box_min) else list(box_min)
        box_max = [float(box_max)] * coord_dim if _scalar(box_max) else list(box_max)
        axes = [
            torch.linspace(lo, hi, n)
            for lo, hi, n in zip(box_min, box_max, latent_resolution)
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        self.register_buffer("latent_queries", grid.reshape(-1, coord_dim))
        act = resolve_activation(activation_function)
        common = dict(
            coord_dim=coord_dim,
            embed_channels=embed_channels,
            embed_max_positions=embed_max_positions,
            act=act,
        )
        self.gno_in = GNOBlock(
            fno_in_channels,
            radius=in_radius,
            reduction="mean",
            hidden_layers=list(in_channel_mlp_hidden_layers),
            **common,
        )
        self.lifting = ChannelMLP(
            fno_in_channels,
            hidden_dim,
            int(lifting_channel_ratio * hidden_dim),
            2,
            act,
        )
        self.fno_blocks = FNOBlocks(
            hidden_dim, n_modes, num_layers, channel_mlp_expansion, act
        )
        self.gno_out = GNOBlock(
            hidden_dim,
            radius=out_radius,
            reduction="sum",
            hidden_layers=list(out_channel_mlp_hidden_layers),
            **common,
        )
        projection = int(projection_channel_ratio * hidden_dim)
        out_mult = 1 + self.var_output
        self.heads = nn.ModuleList(
            ChannelMLP(hidden_dim, dim * out_mult, projection, 2, act)
            for dim in self.head_dims
        )
        if freeze_conv:
            for name, module in self.named_children():
                if name != "heads":
                    module.requires_grad_(False)

    @property
    def num_latent(self):
        return self.latent_queries.shape[0]

    def encode(self, data):
        pos, graph = data.pos, _graph_ids(data)
        num_graphs = int(graph.max()) + 1 if graph.numel() else 1
        grid = self.latent_queries.to(pos.dtype)
        m = self.num_latent
        # Every graph sees the shared grid; latent row g * M + j is grid point j of g.
        j, y = radius_neighbors(grid, pos, self.gno_in.radius)
        latent_pos = grid.repeat(num_graphs, 1)
        h = self.gno_in(pos, latent_pos, data.x, graph[y] * m + j, y)
        return {"latent": h, "num_graphs": num_graphs}

    def process(self, state, data):
        b, c = state["num_graphs"], state["latent"].shape[1]
        x = state["latent"].reshape(b, *self.latent_resolution, c).movedim(-1, 1)
        x = self.lifting(x)
        for i in range(self.fno_blocks.n_layers):
            if self.conv_checkpointing and self.training:
                x = checkpoint(self.fno_blocks, x, i, use_reentrant=False)
            else:
                x = self.fno_blocks(x, i)
        latent = x.movedim(1, -1).reshape(b * self.num_latent, self.hidden_dim)
        return {**state, "latent": latent}

    def _query(self, state, queries, query_graph):
        grid = self.latent_queries.to(queries.dtype)
        m = self.num_latent
        q, j = radius_neighbors(queries, grid, self.gno_out.radius)
        latent_pos = grid.repeat(state["num_graphs"], 1)
        return self.gno_out(
            latent_pos, queries, state["latent"], q, query_graph[q] * m + j
        )

    def decode(self, state, data):
        latents = {}
        if "node" in self.head_type:
            latents["node"] = self._query(state, data.pos, _graph_ids(data))
        if "query" in self.head_type:
            latents["query"] = self._query(
                state, data.query_pos, _query_graph_ids(data)
            )
        return [
            head(latents[level].t().unsqueeze(0))[0].t()
            for level, head in zip(self.head_type, self.heads)
        ]


def _scalar(value):
    return isinstance(value, (int, float))
