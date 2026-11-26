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

"""MultiScale MeshGraphNets (Fortunato et al. 2022, arXiv:2210.00612).

Two-level MGN on a :class:`~hydragnn.data.meshes.MeshHierarchy` built with
``node_transfer=True``. Following Sec. 2 and App. A.1 of the paper:

* Encoder: the fine graph is encoded as in MGN. The coarse graph encodes
  geometry only: relative edge coordinates (and their norm) and a one-hot
  boundary/interior node type. Down (``down_index``, fine -> coarse) and up
  (``up_index``, coarse -> fine) edges encode the relative sender/receiver
  coordinates with MLPs of the same architecture.
* Processor: a schedule string such as ``"1H 11L 1H"`` or
  ``"3H 6L 3H 6L 3H"``. ``H``/``L`` are MGN blocks on the fine/coarse graph.
  A down (``D``) or up (``U``) block is inserted at every level change. Every
  block has its own weights and updates edges then receivers, both with
  residuals: ``e' = e + f_E(e, v_s, v_r)``, ``v_r' = v_r + f_V(v_r, sum e')``.
  As in the paper, the message-passing count includes D and U, so
  ``"1H 11L 1H"`` is 15 steps. Down/up edge latents persist across V-cycles.
* Decoder: on fine nodes only, identical to :class:`MeshGraphNet` (so the
  HydraGNN face/cell/query heads are available too).
"""

import re

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.meshgraphnet import MeshGraphNet, _ProcessorStep, _mlp

_TOKEN = re.compile(r"(\d+)([HL])")


def parse_schedule(schedule):
    """Expand ``"1H 2L 1H"`` into ``["H", "D", "L", "L", "U", "H"]``."""
    tokens = schedule.split()
    if not tokens or not all(_TOKEN.fullmatch(t) for t in tokens):
        raise ValueError(f"msmgn_schedule must look like '1H 11L 1H', got {schedule!r}")
    ops = []
    for token in tokens:
        count, level = _TOKEN.fullmatch(token).groups()
        if int(count) < 1:
            raise ValueError("msmgn_schedule counts must be >= 1")
        if ops and ops[-1] != level:
            ops.append("D" if level == "L" else "U")
        ops += [level] * int(count)
    if ops[0] != "H" or ops[-1] != "H":
        raise ValueError("msmgn_schedule must start and end on the fine level (H)")
    return ops


def default_schedule(num_steps):
    """Single V-cycle ``1H (n-4)L 1H`` with ``n`` steps, counting D and U."""
    if num_steps < 5:
        raise ValueError("MultiscaleMeshGraphNet needs num_conv_layers >= 5")
    return f"1H {num_steps - 4}L 1H"


class _TransferStep(nn.Module):
    """Bipartite MGN block: update edges, then receiver nodes."""

    def __init__(self, latent, mlp_layers, activation):
        super().__init__()
        self.edge_mlp = _mlp(3 * latent, latent, latent, mlp_layers, activation, True)
        self.node_mlp = _mlp(2 * latent, latent, latent, mlp_layers, activation, True)

    def forward(self, src_node, dst_node, edge, edge_index):
        src, dst = edge_index
        edge = edge + self.edge_mlp(
            torch.cat([edge, src_node[src], dst_node[dst]], dim=-1)
        )
        agg = torch.zeros_like(dst_node).index_add_(0, dst, edge)
        dst_node = dst_node + self.node_mlp(torch.cat([dst_node, agg], dim=-1))
        return dst_node, edge


def _edge_features(pos_src, pos_dst):
    rel = pos_src - pos_dst
    return torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)


class MultiscaleMeshGraphNet(MeshGraphNet):
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
        spatial_dim=2,
        conservative_faces=True,
        schedule=None,
        freeze_conv=False,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            output_type,
            0,
            activation_function=activation_function,
            loss_function_type=loss_function_type,
            loss_weights=loss_weights,
            mlp_layers=mlp_layers,
            spatial_dim=spatial_dim,
            conservative_faces=conservative_faces,
        )
        if spatial_dim != 2:
            raise ValueError("MultiscaleMeshGraphNet needs mgn_spatial_dim=2")
        schedule = schedule or default_schedule(num_steps)
        self.schedule = schedule
        self.ops = parse_schedule(schedule)
        if len(self.ops) != num_steps:
            raise ValueError(
                f"msmgn_schedule {schedule!r} has {len(self.ops)} message-passing "
                f"steps (counting D and U) but num_conv_layers is {num_steps}"
            )
        act = activation_function
        edge_in = spatial_dim + 1
        self.coarse_node_encoder = _mlp(
            2, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.coarse_edge_encoder = _mlp(
            edge_in, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.down_edge_encoder = _mlp(
            edge_in, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.up_edge_encoder = _mlp(
            edge_in, hidden_dim, hidden_dim, mlp_layers, act, True
        )
        self.processor = nn.ModuleList(
            (
                _ProcessorStep(hidden_dim, mlp_layers, act)
                if op in "HL"
                else _TransferStep(hidden_dim, mlp_layers, act)
            )
            for op in self.ops
        )
        if freeze_conv:
            for name, module in self.named_children():
                if name != "decoders":
                    module.requires_grad_(False)

    def encode(self, data):
        state = super().encode(data)
        pos = data.pos[:, : self.spatial_dim]
        cpos = data.coarse_pos[:, : self.spatial_dim]
        boundary = data.coarse_boundary_node_mask.long()
        node_type = nn.functional.one_hot(boundary, 2).to(state["node"].dtype)
        src, dst = data.coarse_edge_index
        down_s, down_d = data.down_index
        up_s, up_d = data.up_index
        state.update(
            coarse_node=self.coarse_node_encoder(node_type),
            coarse_edge=self.coarse_edge_encoder(_edge_features(cpos[src], cpos[dst])),
            down_edge=self.down_edge_encoder(_edge_features(pos[down_s], cpos[down_d])),
            up_edge=self.up_edge_encoder(_edge_features(cpos[up_s], pos[up_d])),
        )
        return state

    def _run(self, step, *args):
        if self.conv_checkpointing and self.training:
            return checkpoint(step, *args, use_reentrant=False)
        return step(*args)

    def process(self, state, data):
        s = dict(state)
        for op, step in zip(self.ops, self.processor):
            if op == "H":
                s["node"], s["edge"] = self._run(
                    step, s["node"], s["edge"], data.edge_index
                )
            elif op == "L":
                s["coarse_node"], s["coarse_edge"] = self._run(
                    step, s["coarse_node"], s["coarse_edge"], data.coarse_edge_index
                )
            elif op == "D":
                s["coarse_node"], s["down_edge"] = self._run(
                    step, s["node"], s["coarse_node"], s["down_edge"], data.down_index
                )
            else:
                s["node"], s["up_edge"] = self._run(
                    step, s["coarse_node"], s["node"], s["up_edge"], data.up_index
                )
        return s
