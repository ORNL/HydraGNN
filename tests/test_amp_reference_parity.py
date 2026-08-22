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

"""Numerical parity checks against the authors' AMP-BMS implementation.

The compact reference functions below follow rinikerlab/amp_bms commit
935f1789d70892eab4968bc9c4b409cab1d1ae1f, specifically
``source/amp/AMPHelpers.py`` and ``source/modules/Modules.py``. They retain the
reference implementation's explicitly channel-expanded Rx1/Rx2 representation,
whereas HydraGNN broadcasts channel-free bases. This makes the comparison useful
for detecting channel ordering, contraction, and sender/receiver regressions.

The adapted reference code is MIT licensed; see
``hydragnn/utils/model/amp/LICENSE``.
"""

import numpy as np
import torch

from hydragnn.utils.model.amp import (
    BesselKernel,
    aniso_features,
    build_Rx2,
    build_poles,
)


def _reference_scatter_sum(src, index, dim_size):
    expanded_index = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(src)
    output = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    return output.scatter_add_(0, expanded_index, src)


def _reference_build_poles(
    coefficients, envelope, rx1_expanded, rx2_expanded, receivers, num_nodes
):
    dipole_coefficients, quadrupole_coefficients = (
        coefficients * envelope
    ).tensor_split(2, dim=-1)
    dipoles = _reference_scatter_sum(
        dipole_coefficients.unsqueeze(-1) * rx1_expanded,
        receivers,
        num_nodes,
    )
    quadrupoles = _reference_scatter_sum(
        quadrupole_coefficients[..., None, None] * rx2_expanded,
        receivers,
        num_nodes,
    )
    return dipoles, quadrupoles


def _reference_scalar_product(x, y):
    return (x * y).sum(dim=-1, keepdim=True)


def _reference_aniso_features(
    dipoles, quadrupoles, rx1_expanded, rx2_expanded, senders, receivers
):
    dipoles_1 = dipoles[senders]
    dipoles_2 = dipoles[receivers]
    quadrupoles_1 = quadrupoles[senders]
    quadrupoles_2 = quadrupoles[receivers]

    dipole_dipole = _reference_scalar_product(dipoles_1, dipoles_2)
    dipole_1_rx1 = _reference_scalar_product(dipoles_1, rx1_expanded)
    dipole_2_rx1 = _reference_scalar_product(dipoles_2, rx1_expanded)
    quadrupole_1_rx1 = torch.einsum("ecjk,eck->ecj", quadrupoles_1, rx1_expanded)
    quadrupole_2_rx1 = torch.einsum("ecjk,eck->ecj", quadrupoles_2, rx1_expanded)
    quadrupole_1_rx2 = torch.einsum(
        "ecjk,ecjk->ec", quadrupoles_1, rx2_expanded
    ).unsqueeze(-1)
    quadrupole_2_rx2 = torch.einsum(
        "ecjk,ecjk->ec", quadrupoles_2, rx2_expanded
    ).unsqueeze(-1)
    quadrupole_dipole = _reference_scalar_product(quadrupole_1_rx1, dipoles_2)
    dipole_quadrupole = _reference_scalar_product(quadrupole_2_rx1, dipoles_1)
    quadrupole_quadrupole = torch.einsum(
        "ecjk,ecjk->ec", quadrupoles_1, quadrupoles_2
    ).unsqueeze(-1)
    quadrupole_rx1 = _reference_scalar_product(quadrupole_1_rx1, quadrupole_2_rx1)

    features = torch.cat(
        (
            dipole_1_rx1,
            dipole_2_rx1,
            dipole_dipole,
            quadrupole_1_rx2,
            quadrupole_2_rx2,
            quadrupole_dipole,
            dipole_quadrupole,
            quadrupole_quadrupole,
            quadrupole_rx1,
        ),
        dim=-1,
    )
    return features.view(features.shape[0], features.shape[1] * 9)


def _reference_bessel(distances, cutoff, num_radial, envelope_p):
    frequencies = np.pi * torch.linspace(
        1,
        num_radial,
        num_radial,
        dtype=torch.get_default_dtype(),
    ).unsqueeze(0)
    # Like an nn.Module buffer, frequencies are created in the default dtype and
    # subsequently converted when the model is moved to its execution dtype.
    frequencies = frequencies.to(dtype=distances.dtype, device=distances.device)
    scaled_distances = distances / cutoff
    power_minus_one = torch.pow(scaled_distances, envelope_p - 1)
    power = power_minus_one * scaled_distances
    coefficient_a = -(envelope_p + 1) * (envelope_p + 2) / 2
    coefficient_b = envelope_p * (envelope_p + 2)
    coefficient_c = -envelope_p * (envelope_p + 1) / 2
    envelope = (
        torch.reciprocal(scaled_distances)
        + coefficient_a * power_minus_one
        + coefficient_b * power
        + coefficient_c * power * scaled_distances
    )
    radial = envelope * torch.sin(frequencies * scaled_distances)
    return radial, envelope


def pytest_amp_randomized_multipole_reference_parity():
    generator = torch.Generator().manual_seed(20260721)
    dtype = torch.float64
    num_edges = 29
    num_nodes = 11
    num_channels = 8

    senders = torch.randint(num_nodes, (num_edges,), generator=generator)
    receivers = torch.randint(num_nodes, (num_edges,), generator=generator)
    rx1 = torch.randn(num_edges, 3, dtype=dtype, generator=generator)
    rx1 = rx1 / torch.linalg.vector_norm(rx1, dim=-1, keepdim=True)
    rx2 = build_Rx2(rx1)
    coefficients = torch.randn(
        num_edges, 2 * num_channels, dtype=dtype, generator=generator
    )
    envelope = torch.rand(num_edges, 1, dtype=dtype, generator=generator)

    # AMP-BMS materializes identical bond bases for every multipole channel.
    rx1_expanded = rx1.unsqueeze(1).expand(-1, num_channels, -1)
    rx2_expanded = rx2.unsqueeze(1).expand(-1, num_channels, -1, -1)
    expected_dipoles, expected_quadrupoles = _reference_build_poles(
        coefficients,
        envelope,
        rx1_expanded,
        rx2_expanded,
        receivers,
        num_nodes,
    )
    actual_dipoles, actual_quadrupoles = build_poles(
        coefficients, envelope, rx1, rx2, receivers, num_nodes
    )

    torch.testing.assert_close(actual_dipoles, expected_dipoles, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_quadrupoles, expected_quadrupoles, rtol=0.0, atol=0.0
    )

    expected_features = _reference_aniso_features(
        expected_dipoles,
        expected_quadrupoles,
        rx1_expanded,
        rx2_expanded,
        senders,
        receivers,
    )
    actual_features = aniso_features(
        actual_dipoles, actual_quadrupoles, rx1, rx2, senders, receivers
    )
    torch.testing.assert_close(actual_features, expected_features, rtol=0.0, atol=0.0)


def pytest_amp_randomized_bessel_reference_parity():
    generator = torch.Generator().manual_seed(20260722)
    dtype = torch.float64
    cutoff = 4.0
    num_radial = 8
    envelope_p = 6.0
    distances = 0.05 + (cutoff - 0.10) * torch.rand(
        31, 1, dtype=dtype, generator=generator
    )

    expected_radial, expected_envelope = _reference_bessel(
        distances, cutoff, num_radial, envelope_p
    )
    kernel = BesselKernel(cutoff=cutoff, n_bessel=num_radial, p=envelope_p).to(dtype)
    actual_radial, actual_envelope = kernel(distances)

    torch.testing.assert_close(actual_radial, expected_radial, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_envelope, expected_envelope, rtol=0.0, atol=0.0)
