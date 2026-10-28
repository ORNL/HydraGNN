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
"""Construct explicit periodic attention sources under documented conventions."""

from dataclasses import dataclass
from itertools import product

import torch


@dataclass(frozen=True)
class PeriodicAttentionSources:
    """Image-expanded keys/values and their relation to central atoms."""

    features: torch.Tensor
    positions: torch.Tensor
    batch: torch.Tensor
    base_index: torch.Tensor
    is_central_image: torch.Tensor


def normalize_replication(replication: int | tuple[int, int, int] | list[int]):
    """Return nonnegative replication counts for the three lattice axes."""
    if isinstance(replication, int) and not isinstance(replication, bool):
        values = (replication,) * 3
    elif isinstance(replication, (tuple, list)) and len(replication) == 3:
        values = tuple(replication)
    else:
        raise TypeError("periodic replication must be an integer or length-3 list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("periodic replication counts must be nonnegative integers")
    return values


def normalize_periodic_metadata(cell, pbc, num_graphs, *, device, dtype):
    """Normalize common PyG cell/PBC batch layouts without changing values."""
    if cell is None or pbc is None:
        raise ValueError(
            "periodic equivariant attention requires data.cell and data.pbc"
        )
    cell = torch.as_tensor(cell, device=device, dtype=dtype)
    if cell.shape == (3, 3) and num_graphs == 1:
        cell = cell.unsqueeze(0)
    elif cell.shape == (3 * num_graphs, 3):
        cell = cell.reshape(num_graphs, 3, 3)
    if cell.shape != (num_graphs, 3, 3):
        raise ValueError(f"cell must have shape [{num_graphs}, 3, 3]")
    if not torch.isfinite(cell).all():
        raise ValueError("cell must contain only finite values")
    if torch.any(torch.linalg.det(cell).abs() <= 1.0e-12):
        raise ValueError("each periodic cell must be nonsingular")

    pbc = torch.as_tensor(pbc, device=device)
    if pbc.shape == (3,) and num_graphs == 1:
        pbc = pbc.unsqueeze(0)
    elif pbc.numel() == 3 * num_graphs:
        pbc = pbc.reshape(num_graphs, 3)
    if pbc.shape != (num_graphs, 3):
        raise ValueError(f"pbc must have shape [{num_graphs}, 3]")
    return cell, pbc.to(torch.bool)


def build_periodic_attention_sources(
    node_features,
    positions,
    batch,
    cell,
    pbc,
    replication=1,
):
    """Expand central atoms into explicit periodic key/value images.

    These conventions are intentional modeling choices:

    * Original atoms remain the only queries and outputs, preventing replicated
      cells from multiplying graph-level predictions.
    * Every selected image is a distinct source. Its Cartesian displacement is
      used exactly as constructed; minimum-image wrapping is forbidden because
      it would collapse different explicit images into duplicate pairs.
    * There is no distance cutoff inside the selected supercell. The finite
      replication extent itself is the controlled approximation to the infinite
      crystal and must be checked for convergence.
    * Base coordinates are not silently wrapped. This keeps the map smooth in
      ``positions`` and ``cell`` for force/stress autograd and makes the chosen
      central-cell representation visible to the caller.
    """
    replication = normalize_replication(replication)
    graph_ids = torch.unique(batch, sorted=True)
    if graph_ids.numel() and not torch.equal(
        graph_ids, torch.arange(graph_ids.numel(), device=batch.device)
    ):
        raise ValueError("periodic attention requires contiguous graph identifiers")
    cells, pbcs = normalize_periodic_metadata(
        cell,
        pbc,
        graph_ids.numel(),
        device=positions.device,
        dtype=positions.dtype,
    )

    all_features, all_positions, all_batch = [], [], []
    all_base, all_central = [], []
    for graph_id in graph_ids.tolist():
        node_indices = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        ranges = [
            (
                range(-replication[axis], replication[axis] + 1)
                if pbcs[graph_id, axis]
                else (0,)
            )
            for axis in range(3)
        ]
        integer_shifts = torch.tensor(
            list(product(*ranges)), device=positions.device, dtype=positions.dtype
        )
        # Cell rows are lattice vectors, matching PyG's ``shift @ cell``
        # convention. This operation remains differentiable with respect to the
        # cell; the integer image labels are deliberately nondifferentiable.
        cartesian_shifts = integer_shifts @ cells[graph_id]
        num_images = integer_shifts.shape[0]
        num_nodes = node_indices.shape[0]
        all_features.append(node_features[node_indices].repeat(num_images, 1))
        all_positions.append(
            positions[node_indices].repeat(num_images, 1)
            + cartesian_shifts.repeat_interleave(num_nodes, dim=0)
        )
        all_batch.append(batch.new_full((num_images * num_nodes,), graph_id))
        all_base.append(node_indices.repeat(num_images))
        all_central.append(
            (integer_shifts == 0).all(dim=1).repeat_interleave(num_nodes)
        )

    if not all_features:
        return PeriodicAttentionSources(
            node_features.new_empty((0, node_features.shape[1])),
            positions.new_empty((0, 3)),
            batch.new_empty((0,)),
            batch.new_empty((0,)),
            torch.empty((0,), dtype=torch.bool, device=batch.device),
        )
    return PeriodicAttentionSources(
        torch.cat(all_features),
        torch.cat(all_positions),
        torch.cat(all_batch),
        torch.cat(all_base),
        torch.cat(all_central),
    )
