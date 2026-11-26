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
Decoder: one MLP per head. Encoder and processor MLPs end in LayerNorm.
This is the single-mesh variant (no world edges).

Mesh-level heads (HydraGNN extension, not in the paper) need ``MeshData``:

* ``face``: MLP ``g(c_o, c_n, n_f, A_f, d_on)`` on the owner/neighbour cell
  latents (``face_index``), the face normal and area, and the centroid
  distance. With ``conservative_faces`` the output is
  ``g(c_o, c_n, n_f) - g(c_n, c_o, -n_f)``, so it changes sign with the face
  orientation and a flux shared by two cells cancels exactly.
* ``cell``: MLP on the mean node latent of each cell.
* ``query``: MLP on the node latent linearly interpolated (barycentric
  weights) at ``query_pos`` inside the triangle ``query_cell``; see
  :func:`hydragnn.data.meshes.locate_queries`. Triangle meshes only.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.data.meshes.mesh_data import cell_of_conn
from hydragnn.models.operator_base import OperatorBase
from hydragnn.models.utils import activation_function_selection


def _mlp(in_dim, hidden_dim, out_dim, num_layers, activation, layer_norm):
    layers = []
    dim = in_dim
    for _ in range(num_layers - 1):
        # Fresh instance per layer so learnable activations (PReLU) are not shared.
        layers += [
            nn.Linear(dim, hidden_dim),
            activation_function_selection(activation),
        ]
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
    supported_output_levels = frozenset({"node", "face", "cell", "query"})

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
        conservative_faces=True,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if mlp_layers < 1:
            raise ValueError("mgn_mlp_layers must be >= 1")
        if "query" in self.head_type and spatial_dim != 2:
            raise ValueError("MeshGraphNet query heads need mgn_spatial_dim=2")
        self.spatial_dim = spatial_dim
        self.conservative_faces = conservative_faces
        act = activation_function
        self.node_encoder = _mlp(
            input_dim, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.edge_encoder = _mlp(
            spatial_dim + 1, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.processor = nn.ModuleList(
            _ProcessorStep(hidden_dim, mlp_layers, act) for _ in range(num_steps)
        )
        out_mult = 1 + self.var_output
        in_dims = {
            "node": hidden_dim,
            "cell": hidden_dim,
            "query": hidden_dim,
            "face": 2 * hidden_dim + spatial_dim + 2,
        }
        self.decoders = nn.ModuleList(
            _mlp(in_dims[level], hidden_dim, dim * out_mult, mlp_layers, act, False)
            for level, dim in zip(self.head_type, self.head_dims)
        )
        if freeze_conv:
            # Like Base: freeze the message-passing trunk, keep decoders trainable.
            for module in (self.node_encoder, self.edge_encoder, self.processor):
                module.requires_grad_(False)

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
        node = state["node"]
        outputs = []
        for level, dim, decoder in zip(self.head_type, self.head_dims, self.decoders):
            if level == "node":
                outputs.append(decoder(node))
            elif level == "cell":
                outputs.append(decoder(self._cell_latent(node, data)))
            elif level == "query":
                outputs.append(decoder(self._query_latent(node, data)))
            else:
                outputs.append(self._decode_face(decoder, dim, node, data))
        return outputs

    def _cell_latent(self, node, data):
        cells = cell_of_conn(data)
        total = node.new_zeros(data.cell_size.numel(), node.shape[1])
        total.index_add_(0, cells, node[data.cell_conn])
        return total / data.cell_size.unsqueeze(1).to(node.dtype)

    def _query_latent(self, node, data):
        if not bool((data.cell_size == 3).all()):
            raise ValueError("MeshGraphNet query heads support triangle meshes only")
        # cell_conn offsets stay aligned with triangles after batching.
        tri = data.cell_conn.view(-1, 3)[data.query_cell]
        # Geometry stays in coordinate precision; only the weights are cast,
        # so low-precision latents do not distort the interpolation.
        corners = data.pos[tri][..., :2]
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        v0, v1 = b - a, c - a
        rel = data.query_pos[:, :2].to(a.dtype) - a
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        l1 = (rel[:, 0] * v1[:, 1] - rel[:, 1] * v1[:, 0]) / det
        l2 = (v0[:, 0] * rel[:, 1] - v0[:, 1] * rel[:, 0]) / det
        weights = torch.stack([1 - l1 - l2, l1, l2], dim=1).to(node.dtype)
        return (weights.unsqueeze(-1) * node[tri]).sum(dim=1)

    def _decode_face(self, decoder, dim, node, data):
        cell = self._cell_latent(node, data)
        centroid = self._cell_latent(
            data.pos[:, : self.spatial_dim].to(node.dtype), data
        )
        o, n = data.face_index
        normal = data.face_normal[:, : self.spatial_dim].to(node.dtype)
        area = data.face_area.to(node.dtype)
        dist = (centroid[n] - centroid[o]).norm(dim=-1, keepdim=True)
        out = decoder(torch.cat([cell[o], cell[n], normal, area, dist], dim=-1))
        if not self.conservative_faces:
            return out
        flipped = decoder(torch.cat([cell[n], cell[o], -normal, area, dist], dim=-1))
        mean = out[:, :dim] - flipped[:, :dim]
        # Variance columns (GaussianNLL) must not flip sign with orientation.
        return torch.cat([mean, out[:, dim:] + flipped[:, dim:]], dim=-1)
