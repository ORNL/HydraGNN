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

With ``node_transfer=True`` (triangle meshes) :func:`build_hierarchy` also
adds the node transfer graphs of MultiScale MeshGraphNets (Fortunato et al.
2022, arXiv:2210.00612, Sec. 2): ``down_index [2, 3 N_fine]`` links every fine
node (row 0) to the three corners (row 1) of the coarse triangle containing
it, and ``up_index [2, 3 N_coarse]`` links every coarse node (row 0) to the
corners (row 1) of the fine triangle containing it. ``down_weight`` and
``up_weight`` hold the barycentric weights, so :func:`prolong_nodes` and
:func:`restrict_nodes` are exact for linear fields.
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
        # Rows index different node sets, so each row has its own offset.
        if key == "down_index":
            return torch.tensor([[self.num_nodes], [self.num_coarse_nodes]])
        if key == "up_index":
            return torch.tensor([[self.num_coarse_nodes], [self.num_nodes]])
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


def _barycentric_links(mesh, points):
    """Corners and barycentric weights of the triangle of ``mesh`` holding each point."""
    probe = MeshData(pos=mesh.pos, cell_conn=mesh.cell_conn, cell_size=mesh.cell_size)
    cell = locate_queries(probe, points).query_cell
    tri = mesh.cell_conn.view(-1, 3)[cell]
    a, b, c = (mesh.pos[tri[:, k], :2] for k in range(3))
    v0, v1, rel = b - a, c - a, points[:, :2].to(a.dtype) - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    l1 = (rel[:, 0] * v1[:, 1] - rel[:, 1] * v1[:, 0]) / det
    l2 = (v0[:, 0] * rel[:, 1] - v0[:, 1] * rel[:, 0]) / det
    weight = torch.stack([1 - l1 - l2, l1, l2], dim=1)
    src = torch.arange(points.shape[0]).repeat_interleave(3)
    return torch.stack([src, tri.reshape(-1)]), weight.reshape(-1)


def build_hierarchy(
    fine, coarse, fine_to_coarse=None, volume_rtol=1e-8, node_transfer=False
):
    """Combine two nested :class:`MeshData` levels into one :class:`MeshHierarchy`.

    ``fine_to_coarse`` defaults to the coarse cell containing each fine
    centroid (triangle meshes only, via :func:`locate_queries`). The fine
    volumes mapped to each coarse cell must sum to its volume (within
    ``volume_rtol``), so :func:`restrict` preserves integrals.
    ``node_transfer`` adds the MS-MGN down/up node graphs (triangles only).
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
    if node_transfer:
        data.down_index, data.down_weight = _barycentric_links(coarse, fine.pos)
        data.up_index, data.up_weight = _barycentric_links(fine, coarse.pos)
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


def _interpolate(index, weight, values, num_out):
    out = values.new_zeros(num_out, values.shape[1])
    w = weight.to(values.dtype).unsqueeze(1)
    return out.index_add_(0, index[0], w * values[index[1]])


def prolong_nodes(data, coarse_values):
    """Linear interpolation of coarse-node values to fine nodes (``down_index``)."""
    return _interpolate(
        data.down_index, data.down_weight, coarse_values, data.num_nodes
    )


def restrict_nodes(data, fine_values):
    """Linear interpolation of fine-node values at coarse nodes (``up_index``)."""
    return _interpolate(
        data.up_index, data.up_weight, fine_values, data.num_coarse_nodes
    )
