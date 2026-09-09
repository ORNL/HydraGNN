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

"""Dataset-independent preprocessing helpers for atomistic materials data."""

from dataclasses import dataclass
from typing import Callable, Literal

import torch

GPA_PER_EV_PER_ANGSTROM_CUBED = 160.21766208

StressUnit = Literal["ev_per_angstrom_cubed", "gpa", "kbar"]
StressSign = Literal["tension_positive", "compression_positive"]


@dataclass(frozen=True)
class StressSignDiagnostic:
    """Result of comparing reported stress with finite-difference stress."""

    inferred_source_sign: Literal[
        "tension_positive", "compression_positive", "ambiguous"
    ]
    finite_difference_stress: torch.Tensor
    reported_stress: torch.Tensor
    tension_positive_rmse: float
    compression_positive_rmse: float


def _voigt_to_full(stress: torch.Tensor) -> torch.Tensor:
    """Convert ASE-order Voigt stress ``[xx, yy, zz, yz, xz, xy]`` to 3x3."""
    xx, yy, zz, yz, xz, xy = stress.unbind()
    return torch.stack(
        (
            torch.stack((xx, xy, xz)),
            torch.stack((xy, yy, yz)),
            torch.stack((xz, yz, zz)),
        )
    )


def normalize_stress(
    stress,
    *,
    source_unit: StressUnit,
    source_sign: StressSign,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a symmetric 3x3 stress tensor in eV/Å³, positive in tension.

    One-dimensional input uses ASE's Voigt ordering
    ``[xx, yy, zz, yz, xz, xy]``. Full tensors must already be symmetric;
    silently symmetrizing malformed source data would hide conversion errors.
    """
    value = torch.as_tensor(stress, dtype=dtype)
    if value.shape == (6,):
        value = _voigt_to_full(value)
    elif value.shape != (3, 3):
        raise ValueError("stress must have shape (6,) or (3, 3)")
    if not torch.isfinite(value).all():
        raise ValueError("stress contains non-finite values")
    if not torch.allclose(value, value.T, rtol=1.0e-5, atol=1.0e-7):
        raise ValueError("stress tensor must be symmetric")

    unit_scale = {
        "ev_per_angstrom_cubed": 1.0,
        "gpa": 1.0 / GPA_PER_EV_PER_ANGSTROM_CUBED,
        "kbar": 0.1 / GPA_PER_EV_PER_ANGSTROM_CUBED,
    }
    if source_unit not in unit_scale:
        raise ValueError(f"unsupported stress unit: {source_unit}")
    if source_sign not in {"tension_positive", "compression_positive"}:
        raise ValueError(f"unsupported stress sign convention: {source_sign}")

    sign = -1.0 if source_sign == "compression_positive" else 1.0
    return value * (sign * unit_scale[source_unit])


def diagnose_stress_sign(
    positions: torch.Tensor,
    cell: torch.Tensor,
    reported_stress: torch.Tensor,
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
    *,
    strain_step: float = 1.0e-4,
    ambiguity_rtol: float = 0.05,
    ambiguity_atol: float = 1.0e-8,
) -> StressSignDiagnostic:
    """Infer a source stress sign using central energy finite differences.

    ``energy_fn`` receives strained Cartesian positions and cell vectors and
    must return one scalar total energy in eV. Positions and cells are strained
    together, preserving fractional coordinates. The reported stress must be a
    symmetric ``3 x 3`` tensor in eV/Å³; its sign convention may be unknown.

    The result is ``ambiguous`` when the RMSE values for the reported tensor
    and its negation differ by no more than the configured tolerance. This is
    expected for a nearly stress-free configuration.
    """
    positions = torch.as_tensor(positions)
    cell = torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
    reported_stress = torch.as_tensor(
        reported_stress, dtype=positions.dtype, device=positions.device
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape [3, 3]")
    if reported_stress.shape != (3, 3):
        raise ValueError("reported_stress must have shape [3, 3]")
    if not torch.isfinite(positions).all() or not torch.isfinite(cell).all():
        raise ValueError("positions and cell must contain only finite values")
    if not torch.isfinite(reported_stress).all():
        raise ValueError("reported_stress contains non-finite values")
    if not torch.allclose(reported_stress, reported_stress.T, rtol=1.0e-5, atol=1.0e-7):
        raise ValueError("reported_stress must be symmetric")
    if strain_step <= 0:
        raise ValueError("strain_step must be positive")
    if ambiguity_rtol < 0 or ambiguity_atol < 0:
        raise ValueError("ambiguity tolerances must be non-negative")

    volume = torch.det(cell).abs()
    if not torch.isfinite(volume) or volume <= 0:
        raise ValueError("stress diagnosis requires a non-singular cell")

    identity = torch.eye(3, dtype=cell.dtype, device=cell.device)
    numerical_stress = torch.zeros_like(cell)

    def evaluate(strain):
        transform = identity + strain
        value = energy_fn(positions @ transform, cell @ transform)
        value = torch.as_tensor(value, dtype=cell.dtype, device=cell.device)
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("energy_fn must return one finite scalar energy")
        return value.reshape(())

    for row in range(3):
        for col in range(row, 3):
            basis = torch.zeros_like(cell)
            basis[row, col] = 1.0
            basis[col, row] = 1.0
            energy_plus = evaluate(strain_step * basis)
            energy_minus = evaluate(-strain_step * basis)
            derivative = (energy_plus - energy_minus) / (2.0 * strain_step)
            # An off-diagonal probe changes both symmetric tensor entries.
            component = derivative / (volume * (2.0 if row != col else 1.0))
            numerical_stress[row, col] = component
            numerical_stress[col, row] = component

    tension_rmse = torch.sqrt(torch.mean((reported_stress - numerical_stress) ** 2))
    compression_rmse = torch.sqrt(
        torch.mean((-reported_stress - numerical_stress) ** 2)
    )
    error_scale = max(
        float(tension_rmse), float(compression_rmse), float(ambiguity_atol)
    )
    if abs(float(tension_rmse - compression_rmse)) <= ambiguity_rtol * error_scale:
        inferred_sign = "ambiguous"
    elif tension_rmse < compression_rmse:
        inferred_sign = "tension_positive"
    else:
        inferred_sign = "compression_positive"

    return StressSignDiagnostic(
        inferred_source_sign=inferred_sign,
        finite_difference_stress=numerical_stress.detach(),
        reported_stress=reported_stress.detach(),
        tension_positive_rmse=float(tension_rmse),
        compression_positive_rmse=float(compression_rmse),
    )


def validate_materials_sample(data, *, require_stress: bool = False):
    """Validate fields required for scalable atomistic serialization.

    The validated object is returned unchanged so this helper composes easily
    with dataset pipelines. Invalid samples raise ``ValueError`` with a field-
    specific reason, allowing callers to count and report rejected records.
    """
    required = ("pos", "atomic_numbers", "forces")
    for name in required:
        value = getattr(data, name, None)
        if value is None:
            raise ValueError(f"materials sample is missing {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    num_nodes = data.pos.shape[0]
    if data.pos.shape != (num_nodes, 3):
        raise ValueError("pos must have shape [N, 3]")
    if data.forces.shape != (num_nodes, 3):
        raise ValueError("forces must have shape [N, 3]")
    if data.atomic_numbers.shape not in {(num_nodes,), (num_nodes, 1)}:
        raise ValueError("atomic_numbers must have shape [N] or [N, 1]")

    cell = getattr(data, "cell", None)
    if cell is not None and cell.shape not in {(3, 3), (1, 3, 3)}:
        raise ValueError("cell must have shape [3, 3] or [1, 3, 3]")

    stress = getattr(data, "stress", None)
    if require_stress and stress is None:
        raise ValueError("materials sample is missing stress")
    if stress is not None:
        if stress.shape != (3, 3):
            raise ValueError("stress must have shape [3, 3]")
        if not torch.isfinite(stress).all():
            raise ValueError("stress contains non-finite values")
        if not torch.allclose(stress, stress.T, rtol=1.0e-5, atol=1.0e-7):
            raise ValueError("stress tensor must be symmetric")

    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if torch.any(edge_index[0] == edge_index[1]):
            raise ValueError("edge_index contains self-loops")

    return data
