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
"""Two-level mesh hierarchy (fine + coarse) that batches as one sample.

The fine level uses the plain :class:`MeshData` keys. Coarse-level tensors
carry a ``coarse_`` prefix and ``fine_to_coarse [C_fine]`` maps every fine
cell to the coarse cell that contains it. All coarse indices are offset by
coarse counts during batching, so restriction/prolongation work on batches.
"""

import torch

from .mesh_data import (
    MeshData,
    _CELL_INDEXED,
    _FACE_COLUMNS,
    _NODE_INDEXED,
    locate_queries,
)

PREFIX = "coarse_"


class MeshHierarchy(MeshData):
    @property
    def num_coarse_cells(self):
        if "coarse_cell_size" in self:
            return int(self.coarse_cell_size.numel())
        return int(self.coarse_cell_volume.shape[0])

    @property
    def num_coarse_nodes(self):
        return int(self.coarse_pos.shape[0])

    def __inc__(self, key, value, *args, **kwargs):
        if key == "fine_to_coarse":
            return self.num_coarse_cells
        if key.startswith(PREFIX):
            base = key[len(PREFIX) :]
            if base in _CELL_INDEXED:
                return self.num_coarse_cells
            if base in _NODE_INDEXED or base == "edge_index":
                return self.num_coarse_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.startswith(PREFIX):
            base = key[len(PREFIX) :]
            if base in _FACE_COLUMNS or base == "edge_index":
                return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def build_hierarchy(fine, coarse, fine_to_coarse=None, volume_rtol=1e-8):
    """Combine two nested :class:`MeshData` levels into one :class:`MeshHierarchy`.

    ``fine_to_coarse`` defaults to the coarse cell containing each fine
    centroid (triangle meshes only, via :func:`locate_queries`). The fine
    volumes mapped to each coarse cell must sum to its volume (within
    ``volume_rtol``), so :func:`restrict` preserves integrals.
    """
    if fine_to_coarse is None:
        probe = MeshData(
            pos=coarse.pos, cell_conn=coarse.cell_conn, cell_size=coarse.cell_size
        )
        fine_to_coarse = locate_queries(probe, fine.cell_centroid).query_cell
    fine_to_coarse = torch.as_tensor(fine_to_coarse, dtype=torch.long)
    if fine_to_coarse.shape != (fine.num_cells,):
        raise ValueError("fine_to_coarse must have one entry per fine cell")
    if fine_to_coarse.numel() and not (
        0 <= int(fine_to_coarse.min()) and int(fine_to_coarse.max()) < coarse.num_cells
    ):
        raise ValueError("fine_to_coarse coarse cell id out of range")
    if torch.unique(fine_to_coarse).numel() != coarse.num_cells:
        raise ValueError("every coarse cell must contain at least one fine cell")
    # restrict() preserves integrals only if the fine cells tile each coarse cell.
    child = torch.zeros_like(coarse.cell_volume).index_add_(
        0, fine_to_coarse, fine.cell_volume.to(coarse.cell_volume.dtype)
    )
    if not torch.allclose(child, coarse.cell_volume, rtol=volume_rtol, atol=0):
        raise ValueError(
            "fine cell volumes do not sum to their coarse cell volume; "
            "non-nested meshes are not supported"
        )

    data = MeshHierarchy()
    for key, value in fine.items():
        data[key] = value
    for key, value in coarse.items():
        data[PREFIX + key] = value
    data.fine_to_coarse = fine_to_coarse
    return data


def restrict(data, fine_values):
    """Volume-weighted mean of fine-cell values over each coarse cell."""
    weight = data.cell_volume.to(fine_values.dtype)
    num = fine_values.new_zeros(data.num_coarse_cells, fine_values.shape[1])
    den = fine_values.new_zeros(data.num_coarse_cells, 1)
    num.index_add_(0, data.fine_to_coarse, fine_values * weight)
    den.index_add_(0, data.fine_to_coarse, weight)
    return num / den


def prolong(data, coarse_values):
    """Piecewise-constant injection of coarse-cell values to fine cells."""
    return coarse_values[data.fine_to_coarse]
