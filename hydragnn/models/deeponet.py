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
"""DeepONet: deep operator network (Lu et al., Nat. Mach. Intell. 2021).

``G(u)(y) = sum_k b_k(u(x_1), ..., u(x_m)) t_k(y) + b_0``: the branch net
encodes the input function at ``m`` fixed sensors, the trunk net encodes the
evaluation point ``y``. Follows the DeepXDE reference
(``deepxde/nn/pytorch/deeponet.py``, ``deeponet_strategy.py``): the branch
FNN has no output activation, the trunk output goes through the activation,
and every output has its own scalar bias.

* ``stacked=True`` is the stacked DeepONet of the paper: one scalar-output
  branch net per basis coefficient (``p`` nets); ``False`` is the unstacked
  form with one branch net of width ``p``.
* Multiple outputs ``K`` (all head columns, including GaussianNLL variance
  columns) use DeepXDE's ``multi_output_strategy``: ``independent`` (K
  separate DeepONets), ``split_both`` (branch and trunk of width ``p``
  split into K groups, ``p % K == 0``), ``split_branch`` (branch of width
  ``K p``, shared trunk), ``split_trunk`` (trunk of width ``K p``, shared
  branch). ``None`` means a single output; with K > 1 it falls back to
  ``independent``, as DeepXDE does.

Sensors come from ``data.deeponet_sensors [G, m * c]`` (graph-level, so meshes
of any size share one sensor layout) or, without it, from ``data.x`` of graphs
with exactly ``m`` nodes each, in node order. The trunk evaluates node heads
at ``pos`` and query heads at ``query_pos`` (graph ids from ``query_graph``
or ``query_cell``), so outputs can be queried at arbitrary points.
"""

import math

import torch
from torch import nn

from hydragnn.models.fno import resolve_activation
from hydragnn.models.gino import _graph_ids, _query_graph_ids
from hydragnn.models.operator_base import OperatorBase

STRATEGIES = (None, "independent", "split_both", "split_branch", "split_trunk")


class GroupedFNN(nn.Module):
    """``groups`` independent fully connected nets evaluated in one einsum.

    ``widths = [in, h_1, ..., out]``; output is ``[N, groups * out]`` with the
    columns of group ``g`` at ``g * out : (g + 1) * out``. Hidden layers use
    ``act``; the last layer is linear.
    """

    def __init__(self, groups, widths, act):
        super().__init__()
        self.groups = groups
        self.act = act
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in zip(widths, widths[1:]):
            bound = 1.0 / math.sqrt(fan_in)
            self.weights.append(
                nn.Parameter(
                    torch.empty(groups, fan_in, fan_out).uniform_(-bound, bound)
                )
            )
            self.biases.append(
                nn.Parameter(torch.empty(groups, fan_out).uniform_(-bound, bound))
            )

    def forward(self, x):
        # Group-major [G, N, i] so bmm yields weight grads with the
        # parameters' own (contiguous) strides, as DDP buckets expect.
        h = x.unsqueeze(0).expand(self.groups, -1, -1)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b.unsqueeze(1), h, w)
            if i < last:
                h = self.act(h)
        return h.transpose(0, 1).reshape(x.shape[0], -1)


class DeepONet(OperatorBase):
    supported_output_levels = frozenset({"node", "query"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_layers,
        num_sensors,
        sensor_channels=None,
        branch_layers=None,
        trunk_layers=None,
        basis_dim=64,
        multi_output_strategy=None,
        stacked=False,
        coord_dim=2,
        activation_function="tanh",
        loss_function_type="mse",
        loss_weights=None,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if num_sensors is None or num_sensors < 1:
            raise ValueError("DeepONet needs deeponet_num_sensors >= 1")
        if basis_dim < 1:
            raise ValueError("deeponet_basis_dim must be >= 1")
        if coord_dim < 1:
            raise ValueError("deeponet_coord_dim must be >= 1")
        if multi_output_strategy not in STRATEGIES:
            raise ValueError(
                f"deeponet_multi_output_strategy must be one of {STRATEGIES}"
            )
        self.num_sensors = int(num_sensors)
        self.sensor_channels = int(sensor_channels or input_dim)
        self.coord_dim = int(coord_dim)
        self.stacked = bool(stacked)
        out_mult = 1 + self.var_output
        self.num_outputs = k = sum(d * out_mult for d in self.head_dims)
        strategy = multi_output_strategy
        if k == 1 and strategy is not None:
            raise ValueError(
                "deeponet_multi_output_strategy needs more than one output column"
            )
        if k > 1 and strategy is None:
            strategy = "independent"
        if strategy == "split_both" and basis_dim % k:
            raise ValueError(
                f"split_both needs deeponet_basis_dim divisible by the {k} outputs"
            )
        self.strategy = strategy
        p = basis_dim
        # (branch groups, branch width per group, trunk groups, trunk width)
        layout = {
            None: (1, p, 1, p),
            "independent": (k, p, k, p),
            "split_both": (1, p, 1, p),
            "split_branch": (1, k * p, 1, p),
            "split_trunk": (1, p, 1, k * p),
        }[strategy]
        branch_groups, branch_width, trunk_groups, trunk_width = layout
        # Coefficient groups per contraction: [G, Kb, q] x [N, Kt, q].
        self._kb = k if strategy in ("independent", "split_both", "split_branch") else 1
        self._kt = k if strategy in ("independent", "split_both", "split_trunk") else 1
        self._q = p // k if strategy == "split_both" else p

        branch_hidden = list(branch_layers or [hidden_dim] * num_layers)
        trunk_hidden = list(trunk_layers or [hidden_dim] * num_layers)
        self.act = resolve_activation(activation_function)
        sensor_width = self.num_sensors * self.sensor_channels
        if self.stacked:
            # One scalar-output branch net per basis coefficient.
            self.branch = GroupedFNN(
                branch_groups * branch_width,
                [sensor_width, *branch_hidden, 1],
                self.act,
            )
        else:
            self.branch = GroupedFNN(
                branch_groups, [sensor_width, *branch_hidden, branch_width], self.act
            )
        self.trunk = GroupedFNN(
            trunk_groups, [self.coord_dim, *trunk_hidden, trunk_width], self.act
        )
        self.bias = nn.Parameter(torch.zeros(k))
        if freeze_conv:
            # Keep the output biases trainable, like the heads of other models.
            self.branch.requires_grad_(False)
            self.trunk.requires_grad_(False)

    # -- sensors ------------------------------------------------------------

    def _num_graphs(self, data):
        batch = getattr(data, "batch", None)
        if batch is None:
            return 1
        return int(data.num_graphs)

    def sensors(self, data):
        """Branch input ``[G, m * c]``."""
        g = self._num_graphs(data)
        width = self.num_sensors * self.sensor_channels
        sensors = getattr(data, "deeponet_sensors", None)
        if sensors is not None:
            if sensors.dim() != 2 or tuple(sensors.shape) != (g, width):
                raise ValueError(
                    f"data.deeponet_sensors must have shape [num_graphs={g}, "
                    f"num_sensors*sensor_channels={width}], got {tuple(sensors.shape)}"
                )
            return sensors
        x = data.x
        if x.shape[1] != self.sensor_channels:
            raise ValueError(
                f"DeepONet sensors from data.x need {self.sensor_channels} "
                f"channels, got {x.shape[1]}"
            )
        counts = torch.bincount(_graph_ids(data), minlength=g)
        if bool((counts != self.num_sensors).any()):
            raise ValueError(
                f"DeepONet reads sensors from data.x only for graphs with exactly "
                f"deeponet_num_sensors={self.num_sensors} nodes, got "
                f"{sorted(set(counts.tolist()))}; provide data.deeponet_sensors "
                "for meshes of other sizes"
            )
        return x.reshape(g, width)

    # -- encode / process / decode -----------------------------------------

    def encode(self, data):
        coeff = self.branch(self.sensors(data))
        return {"branch": coeff.reshape(coeff.shape[0], self._kb, self._q)}

    def process(self, state, data):
        return state

    def evaluate(self, state, points, graph):
        """All ``K`` output columns at ``points [Q, >= coord_dim]``."""
        if points.shape[1] < self.coord_dim:
            raise ValueError(
                f"DeepONet trunk needs {self.coord_dim} coordinates, "
                f"got {points.shape[1]}"
            )
        basis = self.act(self.trunk(points[:, : self.coord_dim]))
        basis = basis.reshape(points.shape[0], self._kt, self._q)
        out = (state["branch"][graph] * basis).sum(dim=-1)
        return out.expand(-1, self.num_outputs) + self.bias

    def decode(self, state, data):
        columns = {}
        if "node" in self.head_type:
            columns["node"] = self.evaluate(state, data.pos, _graph_ids(data))
        if "query" in self.head_type:
            columns["query"] = self.evaluate(
                state, data.query_pos, _query_graph_ids(data, "DeepONet")
            )
        outputs, start = [], 0
        for level, dim in zip(self.head_type, self.head_dims):
            width = dim * (1 + self.var_output)
            outputs.append(columns[level][:, start : start + width])
            start += width
        return outputs
