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
"""MeshGraphNet (Pfaff et al., ICLR 2021, arXiv:2010.03409).

Encoder: node MLP on ``x``; edge MLP on ``(pos_s - pos_r, |pos_s - pos_r|)``.
Processor: ``num_steps`` residual blocks
``e' = e + f_e([e, v_s, v_r])``, ``v' = v + f_v([v, sum_{e' -> v} e'])``.
Decoder: one MLP per node head. Encoder and processor MLPs end in LayerNorm.
This is the single-mesh variant (no world edges).
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.operator_base import OperatorBase
from hydragnn.models.utils import activation_function_selection


def _mlp(in_dim, hidden_dim, out_dim, num_layers, activation, layer_norm):
    layers = []
    dim = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), activation]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class _ProcessorStep(nn.Module):
    def __init__(self, latent, mlp_layers, activation):
        super().__init__()
        self.edge_mlp = _mlp(3 * latent, latent, latent, mlp_layers, activation, True)
        self.node_mlp = _mlp(2 * latent, latent, latent, mlp_layers, activation, True)

    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        edge = edge + self.edge_mlp(torch.cat([edge, node[src], node[dst]], dim=-1))
        agg = torch.zeros_like(node).index_add_(0, dst, edge)
        node = node + self.node_mlp(torch.cat([node, agg], dim=-1))
        return node, edge


class MeshGraphNet(OperatorBase):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_steps,
        activation_function="relu",
        loss_function_type="mse",
        loss_weights=None,
        mlp_layers=2,
        spatial_dim=3,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if mlp_layers < 1:
            raise ValueError("mgn_mlp_layers must be >= 1")
        self.spatial_dim = spatial_dim
        act = activation_function_selection(activation_function)
        self.node_encoder = _mlp(input_dim, hidden_dim, hidden_dim, mlp_layers, act, True)
        self.edge_encoder = _mlp(
            spatial_dim + 1, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.processor = nn.ModuleList(
            _ProcessorStep(hidden_dim, mlp_layers, act) for _ in range(num_steps)
        )
        out_mult = 1 + self.var_output
        self.decoders = nn.ModuleList(
            _mlp(hidden_dim, hidden_dim, dim * out_mult, mlp_layers, act, False)
            for dim in self.head_dims
        )

    def encode(self, data):
        pos = data.pos[:, : self.spatial_dim]
        src, dst = data.edge_index
        rel = pos[src] - pos[dst]
        edge_in = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)
        return {"node": self.node_encoder(data.x), "edge": self.edge_encoder(edge_in)}

    def process(self, state, data):
        node, edge = state["node"], state["edge"]
        for step in self.processor:
            if self.conv_checkpointing and self.training:
                node, edge = checkpoint(
                    step, node, edge, data.edge_index, use_reentrant=False
                )
            else:
                node, edge = step(node, edge, data.edge_index)
        return {"node": node, "edge": edge}

    def decode(self, state, data):
        return [decoder(state["node"]) for decoder in self.decoders]
