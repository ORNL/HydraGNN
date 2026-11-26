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
  from owner to neighbour), ``face_area [F, 1]``, ``face_nodes [k, F]``
  (node ids of each face; k = 2 for triangles, 3 for tetrahedra)
* boundary faces: ``bface_owner [Fb]`` (cell id), ``bface_normal [Fb, d]``
  (outward unit), ``bface_area [Fb, 1]``, ``bface_tag [Fb]``,
  ``bface_nodes [k, Fb]``
* query points (off-mesh outputs): ``query_pos [Q, d]`` and
  ``query_cell [Q]``, the id of the cell containing each point; see
  :func:`locate_queries`
* optional metadata: ``material_id [C]`` (long), graph-level ``time``,
  ``dt`` and ``fidelity`` of shape ``[1, 1]``; see :func:`validate_mesh`

Cell connectivity is stored as sizes rather than CSR pointers so PyG can
concatenate it without offset bookkeeping; use :func:`cell_offsets`.
"""

import torch
from torch_geometric.data import Data

_CELL_INDEXED = frozenset({"face_index", "bface_owner", "query_cell"})
_NODE_INDEXED = frozenset({"cell_conn", "face_nodes", "bface_nodes"})
_FACE_COLUMNS = frozenset({"face_index", "face_nodes", "bface_nodes"})


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
        if key in _NODE_INDEXED:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in _FACE_COLUMNS:
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


def query_batch(data):
    """Graph id of every query point (zeros for an unbatched mesh)."""
    return cell_batch(data)[data.query_cell]


def locate_queries(data, query_pos, eps=1e-9):
    """Attach ``query_pos`` and the containing triangle ``query_cell`` to ``data``.

    Brute force over all cells (O(Q * C) memory), meant for preprocessing.
    Points on a shared edge go to the lowest cell id. Raises if a point lies
    outside the mesh.
    """
    if not bool((data.cell_size == 3).all()):
        raise ValueError("locate_queries supports triangle meshes only")
    query_pos = torch.as_tensor(query_pos, dtype=data.pos.dtype)
    tri = data.pos[data.cell_conn.view(-1, 3)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1 = b - a, c - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    rel = query_pos[:, None, :] - a[None]
    l1 = (rel[..., 0] * v1[:, 1] - rel[..., 1] * v1[:, 0]) / det
    l2 = (v0[:, 0] * rel[..., 1] - v0[:, 1] * rel[..., 0]) / det
    inside = (l1 >= -eps) & (l2 >= -eps) & (l1 + l2 <= 1 + eps)
    found = inside.any(dim=1)
    if not bool(found.all()):
        raise ValueError(f"{int((~found).sum())} query points lie outside the mesh")
    data.query_pos = query_pos
    data.query_cell = inside.to(torch.int8).argmax(dim=1)
    return data


def cell_of_conn(data):
    """Cell id of every entry in ``cell_conn``."""
    cells = torch.arange(data.cell_size.numel(), device=data.cell_size.device)
    return cells.repeat_interleave(data.cell_size)


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
    if tri.numel() and (tri.min() < 0 or tri.max() >= pos.shape[0]):
        # Negative indices would be offset by batching into another mesh.
        raise ValueError(f"triangle indices must lie in [0, {pos.shape[0]})")

    # Counter-clockwise orientation gives positive signed area.
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    signed = 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    if (signed == 0).any():
        raise ValueError("degenerate triangle with zero area")
    flip = signed < 0
    tri = tri.clone()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    area = signed.abs()

    # Each triangle contributes edges (v0,v1), (v1,v2), (v2,v0).
    half = tri[:, torch.tensor([[0, 1], [1, 2], [2, 0]])].reshape(-1, 2)
    # Outward normal for a CCW half-edge (p -> q) is (dy, -dx) / length.
    d = pos[half[:, 1]] - pos[half[:, 0]]
    length = d.norm(dim=1, keepdim=True)
    normal = torch.stack([d[:, 1], -d[:, 0]], dim=1) / length
    return _assemble(pos, tri, area, half, normal, length, half, fields)


# Outward-oriented faces of a positively oriented tetrahedron (v0, v1, v2, v3).
_TET_FACES = torch.tensor([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
_TET_EDGES = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def from_tetrahedra(pos, tetrahedra, **fields):
    """Build a :class:`MeshData` from a 3D tetrahedral mesh.

    ``pos`` is ``[N, 3]`` float, ``tetrahedra`` is ``[C, 4]`` long; cells are
    reordered to positive volume. Faces are triangles (``face_nodes [3, F]``).
    """
    pos = torch.as_tensor(pos)
    tet = torch.as_tensor(tetrahedra, dtype=torch.long)
    if pos.dim() != 2 or pos.shape[1] != 3:
        raise ValueError("from_tetrahedra expects pos of shape [N, 3]")
    if tet.dim() != 2 or tet.shape[1] != 4:
        raise ValueError("from_tetrahedra expects tetrahedra of shape [C, 4]")
    if tet.numel() and (tet.min() < 0 or tet.max() >= pos.shape[0]):
        raise ValueError(f"tetrahedron indices must lie in [0, {pos.shape[0]})")

    a, b, c, e = (pos[tet[:, k]] for k in range(4))
    signed = torch.linalg.det(torch.stack([b - a, c - a, e - a], dim=1)) / 6.0
    if (signed == 0).any():
        raise ValueError("degenerate tetrahedron with zero volume")
    flip = signed < 0
    tet = tet.clone()
    tet[flip] = tet[flip][:, [0, 2, 1, 3]]

    tri = tet[:, _TET_FACES].reshape(-1, 3)
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    cross = torch.linalg.cross(p1 - p0, p2 - p0)
    norm = cross.norm(dim=1, keepdim=True)
    edges = tet[:, _TET_EDGES].reshape(-1, 2)
    return _assemble(
        pos, tet, signed.abs(), tri, cross / norm, 0.5 * norm, edges, fields
    )


def _assemble(pos, cells, volume, faces, normal, area, edges, fields):
    """Pair matching per-cell faces into interior/boundary faces.

    ``faces [C * k, m]`` lists the m nodes of each cell's k faces with outward
    ``normal`` and ``area``; ``edges [*, 2]`` gives the node message graph.
    """
    num_cells, per_cell = cells.shape
    owner = torch.arange(num_cells).repeat_interleave(faces.shape[0] // num_cells)
    key = torch.sort(faces, dim=1).values
    _, inverse, counts = torch.unique(
        key, dim=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        raise ValueError("non-manifold mesh: a face is shared by more than 2 cells")

    order = torch.argsort(inverse, stable=True)
    sorted_inv = inverse[order]
    first = torch.ones_like(sorted_inv, dtype=torch.bool)
    first[1:] = sorted_inv[1:] != sorted_inv[:-1]
    first_idx = order[first]
    shared = counts[sorted_inv[first]] == 2
    owner_half = first_idx[shared]
    neigh_half = order[torch.nonzero(first).flatten()[shared] + 1]
    boundary_half = first_idx[~shared]

    node_edges = torch.unique(torch.sort(edges, dim=1).values, dim=0).t()
    data = MeshData(
        pos=pos,
        edge_index=torch.cat([node_edges, node_edges.flip(0)], dim=1),
        cell_conn=cells.reshape(-1),
        cell_size=torch.full((num_cells,), per_cell, dtype=torch.long),
        cell_volume=volume.unsqueeze(1),
        cell_centroid=pos[cells].mean(dim=1),
        face_index=torch.stack([owner[owner_half], owner[neigh_half]]),
        face_normal=normal[owner_half],
        face_area=area[owner_half],
        face_nodes=faces[owner_half].t().contiguous(),
        bface_owner=owner[boundary_half],
        bface_normal=normal[boundary_half],
        bface_area=area[boundary_half],
        bface_tag=torch.zeros(boundary_half.numel(), dtype=torch.long),
        bface_nodes=faces[boundary_half].t().contiguous(),
        **fields,
    )
    mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    mask[faces[boundary_half].reshape(-1)] = True
    data.boundary_node_mask = mask
    return data


def unit_cube_mesh(nx, ny, nz, dtype=torch.float64, **fields):
    """Structured tetrahedralization of ``[0, 1]^3`` with ``6 * nx * ny * nz`` cells."""
    axes = [torch.linspace(0.0, 1.0, n + 1, dtype=dtype) for n in (nx, ny, nz)]
    grid = torch.meshgrid(*axes, indexing="ij")
    pos = torch.stack([g.reshape(-1) for g in grid], dim=1)
    i, j, k = (
        g.reshape(-1)
        for g in torch.meshgrid(
            torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij"
        )
    )

    def vid(di, dj, dk):
        return ((i + di) * (ny + 1) + (j + dj)) * (nz + 1) + (k + dk)

    # Kuhn split of each hexahedron along its main diagonal v000 -> v111.
    paths = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets = []
    for path in paths:
        step = [0, 0, 0]
        verts = [vid(*step)]
        for axis in path:
            step[axis] = 1
            verts.append(vid(*step))
        tets.append(torch.stack(verts, dim=1))
    return from_tetrahedra(pos, torch.cat(tets), **fields)


_GRAPH_SCALARS = ("time", "dt", "fidelity")


def validate_mesh(data, atol=1e-10):
    """Check shapes, index ranges, orientation and closure of a :class:`MeshData`.

    Raises ``ValueError`` on the first violation and returns ``data``.
    Closure (sum of outward ``normal * area`` over each cell is zero) detects
    missing faces and partially flipped normals; the direction of each normal
    is checked against centroid displacements, which assumes convex cells.
    """

    def fail(msg):
        raise ValueError(f"invalid mesh: {msg}")

    n, d = data.pos.shape
    c = data.num_cells
    if "cell_size" in data:
        if int(data.cell_size.sum()) != data.cell_conn.numel():
            fail("sum(cell_size) != len(cell_conn)")
        if data.cell_conn.numel() and not (
            0 <= int(data.cell_conn.min()) and int(data.cell_conn.max()) < n
        ):
            fail("cell_conn index out of range")
    for name, shape in (
        ("cell_volume", (c, 1)),
        ("cell_centroid", (c, d)),
        ("material_id", (c,)),
    ):
        if name in data and tuple(data[name].shape) != shape:
            fail(f"{name} has shape {tuple(data[name].shape)}, expected {shape}")
    if "cell_volume" in data and not bool((data.cell_volume > 0).all()):
        fail("non-positive cell_volume")
    if "material_id" in data and (
        data.material_id.dtype != torch.long or data.material_id.dim() != 1
    ):
        fail("material_id must be a 1D long tensor")

    for prefix, index, lead in (
        ("face", "face_index", (2,)),
        ("bface", "bface_owner", ()),
    ):
        if index not in data:
            continue
        idx = data[index]
        if idx.dtype != torch.long or tuple(idx.shape[:-1]) != lead or idx.dim() == 0:
            fail(f"{index} must be a long tensor of shape {lead + ('n',)}")
        rows = idx.shape[-1]
        if idx.numel() and not (0 <= int(idx.min()) and int(idx.max()) < c):
            fail(f"{index} cell id out of range")
        for suffix, cols in (("normal", d), ("area", 1)):
            value = data[f"{prefix}_{suffix}"]
            if tuple(value.shape) != (rows, cols):
                fail(f"{prefix}_{suffix} must have shape {(rows, cols)}")
        if not bool((data[f"{prefix}_area"] > 0).all()):
            fail(f"non-positive {prefix}_area")
        unit = data[f"{prefix}_normal"].norm(dim=1)
        if not torch.allclose(unit, torch.ones_like(unit), atol=atol, rtol=0):
            fail(f"{prefix}_normal is not unit length")
        nodes = data.get(f"{prefix}_nodes")
        if nodes is not None and (
            nodes.shape[1] != rows
            or nodes.numel()
            and (int(nodes.min()) < 0 or int(nodes.max()) >= n)
        ):
            fail(f"{prefix}_nodes has wrong shape or out-of-range ids")
    if "face_index" in data and bool((data.face_index[0] == data.face_index[1]).any()):
        fail("face_index connects a cell to itself")
    if "bface_tag" in data and data.bface_tag.shape != data.bface_owner.shape:
        fail("bface_tag must have one entry per boundary face")
    if "query_cell" in data or "query_pos" in data:
        qc, qp = data.get("query_cell"), data.get("query_pos")
        if qc is None or qp is None:
            fail("query_pos and query_cell must be given together")
        if qc.dtype != torch.long or qc.dim() != 1:
            fail("query_cell must be a 1D long tensor")
        if tuple(qp.shape) != (qc.numel(), d):
            fail(f"query_pos must have shape {(qc.numel(), d)}")
        if qc.numel() and not (0 <= int(qc.min()) and int(qc.max()) < c):
            fail("query_cell cell id out of range")

    if "face_index" in data and "bface_owner" in data:
        dtype = data.face_normal.dtype
        total = torch.zeros(c, d, dtype=dtype, device=data.pos.device)
        flux = data.face_normal * data.face_area
        total.index_add_(0, data.face_index[0], flux)
        total.index_add_(0, data.face_index[1], -flux)
        total.index_add_(0, data.bface_owner, data.bface_normal * data.bface_area)
        areas = torch.cat([data.face_area, data.bface_area])
        if areas.numel() and float(total.abs().max()) > atol * max(
            float(areas.max()), 1.0
        ):
            fail("cells are not closed (check normal orientation)")

    if "cell_centroid" in data:
        cen = data.cell_centroid
        if "face_index" in data:
            o, nb = data.face_index
            dots = ((cen[nb] - cen[o]) * data.face_normal).sum(dim=1)
            if not bool((dots > 0).all()):
                fail("face_normal does not point from owner to neighbour")
        if "bface_nodes" in data and data.bface_owner.numel():
            mid = data.pos[data.bface_nodes].mean(dim=0)
            dots = ((mid - cen[data.bface_owner]) * data.bface_normal).sum(dim=1)
            if not bool((dots > 0).all()):
                fail("bface_normal does not point out of the owner cell")

    for name in _GRAPH_SCALARS:
        if name in data and tuple(data[name].shape) != (1, 1):
            fail(f"{name} must have shape (1, 1)")
    if "dt" in data and not bool((data.dt > 0).all()):
        fail("dt must be positive")
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
