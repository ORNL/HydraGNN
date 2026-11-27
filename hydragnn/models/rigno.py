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
"""RIGNO: Region Interaction Graph Neural Operator (arXiv:2501.19205).

PyTorch port of camlab-ethz/rigno ``rigno/models/rigno.py`` on the graphs
from :func:`hydragnn.data.meshes.build_rigno_graph`.

* Encoder: one typed message-passing step physical -> regional (p2r).
  Physical nodes embed ``[x, (time, dt), x_norm]``. Regional nodes embed
  zeros of the same width plus ``[x_r, f_p2r r]``.
* Processor: ``num_steps`` unshared interaction-network steps on the r2r
  multimesh. Edge update ``[e, v_s, v_r]``, node update ``[v, mean(e)]``,
  residuals on both.
* Decoder: one step regional -> physical (r2p) onto the encoder's physical
  latents, then one MLP per head. Query heads use r2q edges onto embedded
  query coordinates, so outputs can live at arbitrary points.

All aggregations are means. Block MLPs end in LayerNorm, and with
``time_conditioning`` also in a ``ConditionedNorm`` on the graph-level lead
time ``dt``: ``x (1 + dt s(dt)) + dt b(dt)``. Embedders are not conditioned.
During training each stage keeps a random ``1 - edge_masking`` fraction of
its edges. The mask is drawn from torch's global RNG, so ``torch.manual_seed``
makes it reproducible. Evaluation uses all edges.

Deviations from upstream: output heads are HydraGNN MLPs with ``hidden_dim``
width; non-periodic domains only; query points get their own embedder
because upstream reuses input-node latents and so needs ``x_out == x_inp``.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.meshgraphnet import _mlp
from hydragnn.models.operator_base import OperatorBase


class ConditionedNorm(nn.Module):
    def __init__(self, dim, hidden=16):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(1, hidden), nn.Sigmoid(), nn.Linear(hidden, dim)
        )
        self.bias = nn.Sequential(
            nn.Linear(1, hidden), nn.Sigmoid(), nn.Linear(hidden, dim)
        )
        for layer in (*self.scale, *self.bias):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.normal_(layer.bias, std=0.01)

    def forward(self, x, tau):
        return x * (1 + tau * self.scale(tau)) + tau * self.bias(tau)


class _Block(nn.Module):
    """MLP + LayerNorm, optionally followed by a ConditionedNorm."""

    def __init__(self, in_dim, latent, mlp_layers, act, conditioned):
        super().__init__()
        self.mlp = _mlp(in_dim, latent, latent, mlp_layers, act, True)
        self.cond = ConditionedNorm(latent) if conditioned else None

    def forward(self, x, tau):
        x = self.mlp(x)
        return x if self.cond is None else self.cond(x, tau)


def _mean(values, index, size):
    total = values.new_zeros(size, values.shape[1]).index_add_(0, index, values)
    count = torch.bincount(index, minlength=size).clamp(min=1)
    return total / count.unsqueeze(1).to(values.dtype)


class _Interaction(nn.Module):
    """One typed step: edges then receivers, residuals on both."""

    def __init__(self, latent, mlp_layers, act, conditioned, update_senders):
        super().__init__()
        self.edge = _Block(3 * latent, latent, mlp_layers, act, conditioned)
        self.recv = _Block(2 * latent, latent, mlp_layers, act, conditioned)
        # Upstream also updates nodes without incoming edges, from themselves.
        self.send = (
            _Block(latent, latent, mlp_layers, act, conditioned)
            if update_senders
            else None
        )

    def forward(self, send, recv, edge, index, taus):
        tau_s, tau_r, tau_e = taus
        src, dst = index
        new_edge = self.edge(torch.cat([edge, send[src], recv[dst]], dim=-1), tau_e)
        agg = _mean(new_edge, dst, recv.shape[0])
        new_recv = recv + self.recv(torch.cat([recv, agg], dim=-1), tau_r)
        new_send = send if self.send is None else send + self.send(send, tau_s)
        return new_send, new_recv, edge + new_edge


def _edge_features(x_send, x_recv, index):
    rel = x_send[index[0]] - x_recv[index[1]]
    scale = 2.0 * x_send.shape[1] ** 0.5
    return torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1) / scale


class RIGNO(OperatorBase):
    supported_output_levels = frozenset({"node", "query"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_steps,
        activation_function="silu",
        loss_function_type="mse",
        loss_weights=None,
        mlp_layers=2,
        edge_masking=0.5,
        overlap_p2r=1.0,
        overlap_r2p=2.0,
        concat_time=False,
        time_conditioning=False,
        spatial_dim=2,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if mlp_layers < 1:
            raise ValueError("rigno_mlp_layers must be >= 1")
        if not 0.0 <= edge_masking < 1.0:
            raise ValueError("rigno_edge_masking must lie in [0, 1)")
        if spatial_dim != 2:
            raise ValueError("RIGNO supports spatial_dim=2 only")
        self.edge_masking = edge_masking
        self.overlap_p2r = overlap_p2r
        self.overlap_r2p = overlap_r2p
        self.concat_time = concat_time
        self.time_conditioning = time_conditioning
        d, act, cond = spatial_dim, activation_function, time_conditioning
        p_in = input_dim + 2 * concat_time

        def embed(in_dim):
            return _mlp(in_dim, hidden_dim, hidden_dim, mlp_layers, act, True)

        self.pnode_embed = embed(p_in + d)
        self.rnode_embed = embed(p_in + d + 1)
        self.p2r_embed = embed(d + 1)
        self.r2r_embed = embed(d + 1)
        self.r2p_embed = embed(d + 1)
        self.encoder = _Interaction(hidden_dim, mlp_layers, act, cond, True)
        self.processor = nn.ModuleList(
            _Interaction(hidden_dim, mlp_layers, act, cond, False)
            for _ in range(num_steps)
        )
        self.decoder = _Interaction(hidden_dim, mlp_layers, act, cond, False)
        out_mult = 1 + self.var_output
        self.heads = nn.ModuleList(
            _mlp(hidden_dim, hidden_dim, dim * out_mult, mlp_layers, act, False)
            for dim in self.head_dims
        )
        if "query" in self.head_type:
            self.query_embed = embed(d)
            self.r2q_embed = embed(d + 1)
            self.query_decoder = _Interaction(hidden_dim, mlp_layers, act, cond, False)
        if freeze_conv:
            for name, module in self.named_children():
                if name != "heads":
                    module.requires_grad_(False)

    # -- helpers ----------------------------------------------------------

    def _mask(self, index):
        if not self.training or self.edge_masking == 0.0:
            return index
        keep = int((1.0 - self.edge_masking) * index.shape[1])
        perm = torch.randperm(index.shape[1], device=index.device)[:keep]
        return index[:, perm]

    def _tau(self, data, graph_ids):
        if not self.time_conditioning:
            return None
        dt = getattr(data, "dt", None)
        if dt is None:
            raise ValueError("RIGNO time_conditioning needs graph-level data.dt")
        return dt.reshape(-1, 1)[graph_ids].to(data.rigno_rpos.dtype)

    @staticmethod
    def _node_graph(data):
        batch = getattr(data, "batch", None)
        if batch is None:
            return torch.zeros(data.num_nodes, dtype=torch.long, device=data.pos.device)
        return batch

    def _taus(self, data, send_graph, recv_graph, index):
        if not self.time_conditioning:
            return None, None, None
        return (
            self._tau(data, send_graph),
            self._tau(data, recv_graph),
            self._tau(data, recv_graph[index[1]]),
        )

    def _pnode_inputs(self, data):
        feats = [data.x]
        if self.concat_time:
            for key in ("time", "dt"):
                value = getattr(data, key, None)
                if value is None:
                    raise ValueError(f"RIGNO concat_time needs graph-level data.{key}")
                graph = self._node_graph(data)
                feats.append(value.reshape(-1, 1)[graph].to(data.x.dtype))
        return torch.cat(feats, dim=-1)

    # -- encode / process / decode ---------------------------------------

    def encode(self, data):
        u = self._pnode_inputs(data)
        pgraph, rgraph = self._node_graph(data), data.rigno_rgraph
        radius = (self.overlap_p2r * data.rigno_rradius).unsqueeze(1)
        dummy = u.new_zeros(data.rigno_rpos.shape[0], u.shape[1])
        pnode = self.pnode_embed(torch.cat([u, data.rigno_x], dim=-1))
        rnode = self.rnode_embed(torch.cat([dummy, data.rigno_rpos, radius], dim=-1))
        index = self._mask(data.rigno_p2r_index)
        edge = self.p2r_embed(_edge_features(data.rigno_x, data.rigno_rpos, index))
        taus = self._taus(data, pgraph, rgraph, index)
        pnode, rnode, _ = self.encoder(pnode, rnode, edge, index, taus)
        return {"pnode": pnode, "rnode": rnode}

    def process(self, state, data):
        rnode, rpos, rgraph = state["rnode"], data.rigno_rpos, data.rigno_rgraph
        index = self._mask(data.rigno_r2r_index)
        edge = self.r2r_embed(_edge_features(rpos, rpos, index))
        taus = self._taus(data, rgraph, rgraph, index)
        for step in self.processor:
            if self.conv_checkpointing and self.training:
                _, rnode, edge = checkpoint(
                    step, rnode, rnode, edge, index, taus, use_reentrant=False
                )
            else:
                _, rnode, edge = step(rnode, rnode, edge, index, taus)
        return {**state, "rnode": rnode}

    def decode(self, state, data):
        rnode, rgraph = state["rnode"], data.rigno_rgraph
        latents = {}
        if "node" in self.head_type:
            index = self._mask(data.rigno_r2p_index)
            edge = self.r2p_embed(_edge_features(data.rigno_rpos, data.rigno_x, index))
            taus = self._taus(data, rgraph, self._node_graph(data), index)
            latents["node"] = self.decoder(rnode, state["pnode"], edge, index, taus)[1]
        if "query" in self.head_type:
            index = self._mask(data.rigno_r2q_index)
            qx = data.rigno_qx
            edge = self.r2q_embed(_edge_features(data.rigno_rpos, qx, index))
            taus = self._taus(data, rgraph, data.rigno_qgraph, index)
            query = self.query_embed(qx)
            latents["query"] = self.query_decoder(rnode, query, edge, index, taus)[1]
        return [head(latents[level]) for level, head in zip(self.head_type, self.heads)]
