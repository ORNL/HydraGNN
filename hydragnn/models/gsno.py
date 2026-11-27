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
"""GSNO: Graph Spectral Neural Operator on irregular meshes.

Implements the layer of the GSNO paper (graph Laplacian eigenbasis ``Phi``,
``k_s`` spatial modes, optional temporal FFT with ``k_t`` modes):

``v_{l+1} = act(W v_l + Phi F_t^{-1}(R . F_t(Phi^T v_l)))``

with ``W`` a pointwise linear map, ``act`` GELU, ``R`` complex
``[k_s, k_t, C, C]`` (real ``[k_s, C, C]`` for steady problems, where ``F_t``
is dropped), lifting ``P`` and projection ``Q`` (``width -> 128 -> out``).
The basis ``data.gsno_basis [N, >= k_s]`` comes from
:func:`hydragnn.data.meshes.attach_graph_spectral_basis`; graphs in a batch may
have different meshes since every graph is projected onto its own basis.

Temporal I/O (an assumption, the paper only states ``T_in = 5 -> T_out = 46``):
node inputs are ``T_in`` snapshots flattened time-major
(``[N, T_in * c_in]``); every output time ``t`` sees all input snapshots plus
the time coordinate ``t / T_out`` (the FNO-3D convention), the operator acts on
the ``T_out`` axis and each head predicts ``[N, T_out * c_out]`` time-major.
``R`` uses the FNO-1D init ``U(0, 1) / C^2``.

The layer is invariant to eigenvector sign flips (each mode enters through
``phi_k phi_k^T``). Rotations inside a degenerate eigenspace leave it invariant
only if all modes of that eigenspace share ``R``; ``tie_degenerate=True``
(an extension, off by default as in the paper) shares them using
``data.gsno_groups``.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.fno import resolve_activation
from hydragnn.models.operator_base import OperatorBase


def _graph_ids(data):
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    return batch


class GraphSpectralConv(nn.Module):
    """``Phi F_t^{-1}(R . F_t(Phi^T v))`` for ``v`` of shape ``[N, T, C]``."""

    def __init__(self, channels, spatial_modes, temporal_modes=0):
        super().__init__()
        self.spatial_modes = spatial_modes
        self.temporal_modes = temporal_modes
        scale = 1.0 / (channels * channels)
        shape = [spatial_modes]
        if temporal_modes:
            shape += [temporal_modes]
        shape += [channels, channels]
        if temporal_modes:
            shape += [2]
        self.weight = nn.Parameter(scale * torch.rand(*shape))

    def forward(self, v, basis, graph, num_graphs, groups=None):
        """``groups [G, >= k_s]`` (optional) ties the weights of every mode
        to the first mode of its eigenspace, per graph."""
        n, t, c = v.shape
        phi = basis[:, : self.spatial_modes]
        coeff = v.new_zeros(num_graphs, self.spatial_modes, t, c)
        coeff = coeff.index_add(0, graph, phi[:, :, None, None] * v[:, None])
        weight = self.weight
        if groups is None:
            weight = weight.unsqueeze(0).expand(num_graphs, *weight.shape)
        else:
            weight = weight[groups[:, : self.spatial_modes]]
        if self.temporal_modes:
            spectrum = torch.fft.rfft(coeff, dim=2)
            kept = min(self.temporal_modes, spectrum.shape[2])
            weight = torch.view_as_complex(weight)[:, :, :kept]
            out = spectrum.new_zeros(spectrum.shape)
            out[:, :, :kept] = torch.einsum(
                "gkti,gktio->gkto", spectrum[:, :, :kept], weight
            )
            coeff = torch.fft.irfft(out, n=t, dim=2)
        else:
            coeff = torch.einsum("gkti,gkio->gkto", coeff, weight)
        return torch.einsum("nk,nkto->nto", phi, coeff[graph])


class GSNO(OperatorBase):
    supported_output_levels = frozenset({"node"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_layers,
        spatial_modes=8,
        temporal_modes=0,
        input_time_steps=1,
        output_time_steps=1,
        projection_channels=128,
        tie_degenerate=False,
        activation_function="gelu",
        loss_function_type="mse",
        loss_weights=None,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if num_layers < 1:
            raise ValueError("GSNO needs num_conv_layers >= 1")
        if spatial_modes < 1 or temporal_modes < 0:
            raise ValueError("gsno_spatial_modes >= 1 and gsno_temporal_modes >= 0")
        temporal = temporal_modes > 0
        if not temporal and (input_time_steps != 1 or output_time_steps != 1):
            raise ValueError("steady GSNO (gsno_temporal_modes=0) needs 1 time step")
        if temporal and temporal_modes > output_time_steps // 2 + 1:
            raise ValueError("gsno_temporal_modes must be <= T_out // 2 + 1")
        if input_dim % input_time_steps:
            raise ValueError("input_dim must be divisible by gsno_input_time_steps")
        for dim in output_dim:
            if dim % output_time_steps:
                raise ValueError(
                    "output_dim must be divisible by gsno_output_time_steps"
                )
        self.temporal = temporal
        self.tie_degenerate = tie_degenerate
        self.spatial_modes = spatial_modes
        self.output_time_steps = output_time_steps
        self.act = resolve_activation(activation_function)
        self.lifting = nn.Linear(input_dim + int(temporal), hidden_dim)
        self.convs = nn.ModuleList(
            GraphSpectralConv(hidden_dim, spatial_modes, temporal_modes)
            for _ in range(num_layers)
        )
        self.skips = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        out_mult = 1 + self.var_output
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, projection_channels),
                _Activation(self.act),
                nn.Linear(projection_channels, dim // output_time_steps * out_mult),
            )
            for dim in self.head_dims
        )
        if freeze_conv:
            for name, module in self.named_children():
                if name != "heads":
                    module.requires_grad_(False)

    def encode(self, data):
        basis = getattr(data, "gsno_basis", None)
        if basis is None or basis.shape[1] < self.spatial_modes:
            raise ValueError(
                f"GSNO needs data.gsno_basis with >= {self.spatial_modes} columns "
                "(see hydragnn.data.meshes.attach_graph_spectral_basis)"
            )
        x = data.x
        t = self.output_time_steps
        x = x.unsqueeze(1).expand(-1, t, -1)
        if self.temporal:
            time = torch.arange(t, dtype=x.dtype, device=x.device) / t
            x = torch.cat([x, time.view(1, t, 1).expand(x.shape[0], t, 1)], dim=-1)
        return {"node": self.lifting(x)}

    def _layer(self, v, basis, graph, num_graphs, groups, index):
        spectral = self.convs[index](v, basis, graph, num_graphs, groups)
        return self.act(spectral + self.skips[index](v))

    def process(self, state, data):
        graph = _graph_ids(data)
        num_graphs = int(graph.max()) + 1
        basis = data.gsno_basis.to(state["node"].dtype)
        groups = None
        if self.tie_degenerate:
            groups = getattr(data, "gsno_groups", None)
            if groups is None:
                raise ValueError("gsno_tie_degenerate needs data.gsno_groups")
            groups = groups.reshape(num_graphs, -1)
        v = state["node"]
        for i in range(len(self.convs)):
            args = (v, basis, graph, num_graphs, groups, i)
            if self.conv_checkpointing and self.training:
                v = checkpoint(self._layer, *args, use_reentrant=False)
            else:
                v = self._layer(*args)
        return {"node": v}

    def decode(self, state, data):
        outputs = []
        for head in self.heads:
            out = head(state["node"])  # [N, T, c (x2 with variance)]
            if self.var_output:
                mean, var = out.chunk(2, dim=-1)
                out = torch.cat([mean.flatten(1), var.flatten(1)], dim=1)
            else:
                out = out.flatten(1)
            outputs.append(out)
        return outputs


class _Activation(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)
