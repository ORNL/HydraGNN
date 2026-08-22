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

# Adapted from the MIT-licensed reference AMP implementation (see LICENSE in
# this directory):
#   Github: https://github.com/rinikerlab/amp_bms
#     source/amp/AMPHelpers.py      (build_poles, aniso_features)
#     source/utilities/Utilities.py (ff_module, scalar_product, detrace, build_Rx2)
#     source/modules/Modules.py     (BesselKernel)
#   Paper:  Thuerlemann et al., J. Am. Chem. Soc. 2026, DOI 10.1021/jacs.6c00217
#           (eqs. 3-8: local tensor-product bases R^k, anisotropic features
#            g_0..g_8, message + hidden-state update, per-atom readout V_AMP).
#
# The reference stores per-edge bond bases Rx1 / Rx2 already expanded across the
# multipole-channel axis. Here we keep them channel-free ([E, 3] and [E, 3, 3])
# and broadcast the channel dimension inside the contractions; this is
# mathematically identical but avoids materializing the redundant channel copies.

import torch
import torch.nn as nn
from torch import Tensor

import torch_scatter


def ff_module(
    node_size,
    num_layers,
    input_size,
    with_bias=True,
    output_size=None,
    activation=None,
    final_activation=None,
):
    """Feed-forward stack matching the reference ``ff_module``.

    ``num_layers`` hidden (Linear -> activation) blocks of width ``node_size``,
    followed by an optional unbiased projection to ``output_size``. When
    ``output_size`` is ``None`` the module emits ``node_size`` features.
    """
    if activation is None:
        activation = nn.SiLU()
    layers = []
    for idl in range(num_layers):
        in_dim = input_size if idl == 0 else node_size
        layers.append(nn.Linear(in_dim, node_size, bias=with_bias))
        layers.append(activation)
    if output_size is not None:
        layers.append(nn.Linear(node_size, output_size, bias=False))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


def scalar_product(x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
    """Contract the last (Cartesian) axis: sum_a x_a y_a."""
    return (x * y).sum(dim=-1, keepdim=keepdim)


def detrace(rxr: Tensor) -> Tensor:
    """Remove the isotropic (trace) part of a rank-2 Cartesian tensor.

    ``rxr``: [..., 3, 3] -> traceless [..., 3, 3]. Matches the reference
    ``detrace``: subtract the mean of the diagonal from each diagonal element.
    """
    trace_mean = rxr.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # [...]
    eye = torch.eye(3, dtype=rxr.dtype, device=rxr.device)
    return rxr - trace_mean[..., None, None] * eye


def build_Rx2(rx1: Tensor) -> Tensor:
    """Rank-2 traceless bond basis R^2 = detrace(u (x) u) from unit vectors.

    ``rx1``: [E, 3] (unit bond vectors) -> ``rx2``: [E, 3, 3].
    """
    return detrace(rx1.unsqueeze(-1) * rx1.unsqueeze(-2))


def build_poles(
    coefficients: Tensor,
    envelope: Tensor,
    rx1: Tensor,
    rx2: Tensor,
    receivers: Tensor,
    n_nodes: int,
):
    """Build per-atom dipole/quadrupole multipoles (paper eqs. 3-5, 7).

    Each edge contributes a learned scalar coefficient times the (equivariant)
    bond bases; contributions are scatter-summed onto the receiving node. The
    coefficients are damped by the radial ``envelope`` so multipoles vanish
    smoothly at the cutoff.

    Args:
        coefficients: [E, 2 * n_channels] eq-message-layer output (dipole half,
            quadrupole half).
        envelope:     [E, 1] radial envelope.
        rx1:          [E, 3] unit bond vectors (R^1).
        rx2:          [E, 3, 3] traceless bond tensors (R^2).
        receivers:    [E] destination node index per edge.
        n_nodes:      number of atoms N.

    Returns:
        dipos: [N, n_channels, 3], quads: [N, n_channels, 3, 3].
    """
    coeffs = coefficients * envelope  # [E, 2C]
    dip_c, quad_c = coeffs.tensor_split(2, dim=-1)  # [E, C], [E, C]
    edge_dip = dip_c.unsqueeze(-1) * rx1.unsqueeze(1)  # [E, C, 3]
    edge_quad = quad_c.unsqueeze(-1).unsqueeze(-1) * rx2.unsqueeze(1)  # [E, C, 3, 3]
    dipos = torch_scatter.scatter_add(edge_dip, receivers, dim=0, dim_size=n_nodes)
    quads = torch_scatter.scatter_add(edge_quad, receivers, dim=0, dim_size=n_nodes)
    return dipos, quads


def aniso_features(
    dipos: Tensor,
    quads: Tensor,
    rx1: Tensor,
    rx2: Tensor,
    senders: Tensor,
    receivers: Tensor,
) -> Tensor:
    """Anisotropic edge features g_0..g_8 (paper eq. 6).

    Nine rotation-invariant scalars per channel, formed by fully contracting the
    sender/receiver multipoles with each other and with the bond bases. Because
    every free Cartesian index is summed, each feature is invariant under SO(3),
    which is what makes the downstream energy invariant and forces equivariant.

    Returns:
        [E, n_channels * 9] in the reference channel-major order
        (D1.R, D2.R, D.D, Q1:R2, Q2:R2, (Q1.R).D2, (Q2.R).D1, Q1:Q2, (Q1.R).(Q2.R)).
    """
    d1 = dipos[senders]  # [E, C, 3]
    d2 = dipos[receivers]  # [E, C, 3]
    q1 = quads[senders]  # [E, C, 3, 3]
    q2 = quads[receivers]  # [E, C, 3, 3]
    r1 = rx1.unsqueeze(1)  # [E, 1, 3]

    D1_R = scalar_product(d1, r1)  # [E, C, 1]
    D2_R = scalar_product(d2, r1)  # [E, C, 1]
    dipo_dipo = scalar_product(d1, d2)  # [E, C, 1]
    Q1_R1 = torch.einsum("ecjk,ek->ecj", q1, rx1)  # [E, C, 3]
    Q2_R1 = torch.einsum("ecjk,ek->ecj", q2, rx1)  # [E, C, 3]
    Q1_R2 = torch.einsum("ecjk,ejk->ec", q1, rx2).unsqueeze(-1)  # [E, C, 1]
    Q2_R2 = torch.einsum("ecjk,ejk->ec", q2, rx2).unsqueeze(-1)  # [E, C, 1]
    quad_dipo = scalar_product(Q1_R1, d2)  # [E, C, 1]
    dipo_quad = scalar_product(Q2_R1, d1)  # [E, C, 1]
    quad_quad = torch.einsum("ecjk,ecjk->ec", q1, q2).unsqueeze(-1)  # [E, C, 1]
    quad_R = scalar_product(Q1_R1, Q2_R1)  # [E, C, 1]

    feats = torch.cat(
        (
            D1_R,
            D2_R,
            dipo_dipo,
            Q1_R2,
            Q2_R2,
            quad_dipo,
            dipo_quad,
            quad_quad,
            quad_R,
        ),
        dim=-1,
    )  # [E, C, 9]
    return feats.reshape(feats.shape[0], feats.shape[1] * 9)


class BesselKernel(nn.Module):
    """Bessel radial embedding with a polynomial envelope (paper eq. of the
    ML/ML edge features), ported from the reference ``BesselKernel``.

    ``forward`` returns both the Bessel features (envelope * sin(freq * x)) and
    the envelope itself. The envelope is
        1/x + a x^(p-1) + b x^p + c x^(p+1),   x = r / cutoff,
    with a, b, c the standard polynomial-cutoff coefficients so the polynomial
    part vanishes at the cutoff.

    The reference calls this quantity an ``envelope``, but it is more precisely
    the bounded polynomial cutoff C(x) divided by x. It therefore behaves as
    1/x near the origin rather than as a conventional [0, 1] cutoff. The Bessel
    features remain finite because sin(n*pi*x) cancels that singular factor,
    while the raw envelope separately scales messages and multipole
    coefficients exactly as in AMP-BMS.
    """

    def __init__(
        self,
        cutoff: float = 4.0,
        n_bessel: int = 8,
        trainable: bool = False,
        p: float = 6.0,
    ):
        super().__init__()
        if cutoff <= 0:
            raise ValueError("BesselKernel requires cutoff > 0.")
        if n_bessel <= 0:
            raise ValueError("BesselKernel requires n_bessel > 0.")
        if p <= 1:
            raise ValueError("BesselKernel requires envelope exponent p > 1.")

        frequencies = (
            torch.pi
            * torch.arange(1, n_bessel + 1, dtype=torch.get_default_dtype())
        ).unsqueeze(0)
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.cutoff = float(cutoff)
        self.p = float(p)
        self.a = -(p + 1) * (p + 2) / 2
        self.b = p * (p + 2)
        self.c = -p * (p + 1) / 2

    def envelope_bessel(self, x: Tensor) -> Tensor:
        xp_ = torch.pow(x, self.p - 1)
        xp = xp_ * x
        return torch.reciprocal(x) + self.a * xp_ + self.b * xp + self.c * xp * x

    def forward(self, r1: Tensor):
        x = r1 / self.cutoff
        envelope = self.envelope_bessel(x)
        return envelope * torch.sin(self.frequencies * x), envelope
