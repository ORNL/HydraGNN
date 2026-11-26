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
"""Mesh-aware PyG ``Data`` whose cell/face indices survive batching.

Entity tensors (all optional except ``pos``):

* nodes: ``pos [N, d]``, ``edge_index [2, E]`` (node graph)
* cells: ``cell_conn [nnz]`` node ids, ``cell_size [C]`` nodes per cell,
  ``cell_volume [C, 1]``, ``cell_centroid [C, d]``
* interior faces, stored once and oriented ``face_index[0] -> face_index[1]``
  (cell ids): ``face_index [2, F]``, ``face_normal [F, d]`` (unit, pointing
  from owner to neighbour), ``face_area [F, 1]``
* boundary faces: ``bface_owner [Fb]`` (cell id), ``bface_normal [Fb, d]``
  (outward unit), ``bface_area [Fb, 1]``, ``bface_tag [Fb]``

Cell connectivity is stored as sizes rather than CSR pointers so PyG can
concatenate it without offset bookkeeping; use :func:`cell_offsets`.
"""

import torch
from torch_geometric.data import Data

_CELL_INDEXED = frozenset({"face_index", "bface_owner"})


class MeshData(Data):
    @property
    def num_cells(self):
        if "cell_size" in self:
            return int(self.cell_size.numel())
        if "cell_volume" in self:
            return int(self.cell_volume.shape[0])
        return 0

    @property
    def num_faces(self):
        return int(self.face_index.shape[1]) if "face_index" in self else 0

    def __inc__(self, key, value, *args, **kwargs):
        if key in _CELL_INDEXED:
            return self.num_cells
        if key == "cell_conn":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "face_index":
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def cell_offsets(data):
    """Start position of each cell inside ``cell_conn``."""
    return torch.cumsum(data.cell_size, 0) - data.cell_size


def cell_batch(data):
    """Graph id of every cell (zeros for an unbatched mesh)."""
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(data.num_cells, dtype=torch.long, device=data.pos.device)
    return batch[data.cell_conn[cell_offsets(data)]]


def from_triangles(pos, triangles, **fields):
    """Build a :class:`MeshData` from a 2D triangulation.

    ``pos`` is ``[N, 2]`` float, ``triangles`` is ``[C, 3]`` long. Extra
    keyword tensors (e.g. ``x``, ``y``) are attached unchanged. Boundary
    faces get tag 0; callers may overwrite ``bface_tag``.
    """
    pos = torch.as_tensor(pos)
    tri = torch.as_tensor(triangles, dtype=torch.long)
    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError("from_triangles expects pos of shape [N, 2]")
    if tri.dim() != 2 or tri.shape[1] != 3:
        raise ValueError("from_triangles expects triangles of shape [C, 3]")

    # Counter-clockwise orientation gives positive signed area.
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    signed = 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    if (signed == 0).any():
        raise ValueError("degenerate triangle with zero area")
    flip = signed < 0
    tri = tri.clone()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    area = signed.abs()
    centroid = pos[tri].mean(dim=1)

    # Each triangle contributes edges (v0,v1), (v1,v2), (v2,v0).
    num_cells = tri.shape[0]
    local = torch.tensor([[0, 1], [1, 2], [2, 0]])
    half = tri[:, local].reshape(-1, 2)
    owner = torch.arange(num_cells).repeat_interleave(3)
    key = torch.sort(half, dim=1).values
    uniq, inverse, counts = torch.unique(
        key, dim=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        raise ValueError("non-manifold mesh: an edge is shared by >2 triangles")

    # Outward normal for a CCW half-edge (p -> q) is (dy, -dx) / length.
    p, q = pos[half[:, 0]], pos[half[:, 1]]
    d = q - p
    length = d.norm(dim=1, keepdim=True)
    normal = torch.stack([d[:, 1], -d[:, 0]], dim=1) / length

    order = torch.argsort(inverse, stable=True)
    sorted_inv = inverse[order]
    first = torch.ones_like(sorted_inv, dtype=torch.bool)
    first[1:] = sorted_inv[1:] != sorted_inv[:-1]
    first_idx = order[first]
    shared = counts[sorted_inv[first]] == 2
    owner_half = first_idx[shared]
    pos_in_order = torch.nonzero(first).flatten()[shared]
    neigh_half = order[pos_in_order + 1]
    boundary_half = first_idx[~shared]

    face_index = torch.stack([owner[owner_half], owner[neigh_half]])
    node_edges = uniq.t()

    data = MeshData(
        pos=pos,
        edge_index=torch.cat([node_edges, node_edges.flip(0)], dim=1),
        cell_conn=tri.reshape(-1),
        cell_size=torch.full((num_cells,), 3, dtype=torch.long),
        cell_volume=area.unsqueeze(1),
        cell_centroid=centroid,
        face_index=face_index,
        face_normal=normal[owner_half],
        face_area=length[owner_half],
        bface_owner=owner[boundary_half],
        bface_normal=normal[boundary_half],
        bface_area=length[boundary_half],
        bface_tag=torch.zeros(boundary_half.numel(), dtype=torch.long),
        **fields,
    )
    boundary_nodes = torch.unique(half[boundary_half].reshape(-1))
    mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    mask[boundary_nodes] = True
    data.boundary_node_mask = mask
    return data


def unit_square_mesh(nx, ny, dtype=torch.float64, **fields):
    """Structured triangulation of ``[0, 1]^2`` with ``2 * nx * ny`` triangles."""
    xs = torch.linspace(0.0, 1.0, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    i, j = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    v00 = (i * (ny + 1) + j).reshape(-1)
    v10 = v00 + (ny + 1)
    v01 = v00 + 1
    v11 = v10 + 1
    tri = torch.cat(
        [torch.stack([v00, v10, v11], 1), torch.stack([v00, v11, v01], 1)], dim=0
    )
    return from_triangles(pos, tri, **fields)
