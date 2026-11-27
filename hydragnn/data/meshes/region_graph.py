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
"""Region interaction graphs for RIGNO (Mousavi et al. 2025, arXiv:2501.19205).

Port of ``RegionInteractionGraphBuilder`` from camlab-ethz/rigno (non-periodic
branch). Coordinates are mapped to ``[-1, 1]`` with one isotropic scale, so
the aspect ratio of the domain is kept. Regional nodes are a random subset of
``int(N / subsample_factor)`` physical nodes. Each regional node gets the
minimum support radius that makes the union of the sub-regions cover the
convex hull of the regional nodes. Upstream computes this radius from the
medians of the adjacent Delaunay triangles.

Stored fields (all batch with the right offsets):

* ``rigno_x [N, d]``: normalized physical coordinates
* ``rigno_rpos [R, d]``, ``rigno_rradius [R]``, ``rigno_rgraph [R]``
* ``rigno_p2r_index [2, E]``: physical -> regional, ``|x_p - x_r| <= f_p2r r``
* ``rigno_r2r_index [2, E]``: Delaunay multimesh over ``rmesh_levels`` nested
  prefixes of the regional nodes
* ``rigno_r2p_index [2, E]``: regional -> physical, ``|x_p - x_r| <= f_r2p r``
* with ``query_pos``: ``rigno_qx [Q, d]``, ``rigno_qgraph [Q]`` and
  ``rigno_r2q_index`` (regional -> query, same radii as r2p)
"""

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree

from .mesh_data import MeshData

_ONE_PER_GRAPH = frozenset({"rigno_rgraph", "rigno_qgraph"})


class RegionGraph(MeshData):
    @property
    def num_rnodes(self):
        return int(self.rigno_rpos.shape[0])

    def __inc__(self, key, value, *args, **kwargs):
        # Rows of bipartite indices point into different node sets.
        if key == "rigno_p2r_index":
            return torch.tensor([[self.num_nodes], [self.num_rnodes]])
        if key == "rigno_r2p_index":
            return torch.tensor([[self.num_rnodes], [self.num_nodes]])
        if key == "rigno_r2q_index":
            return torch.tensor([[self.num_rnodes], [int(self.rigno_qx.shape[0])]])
        if key == "rigno_r2r_index":
            return self.num_rnodes
        if key in _ONE_PER_GRAPH:
            return 1
        return super().__inc__(key, value, *args, **kwargs)


def _triangulation_medians(tri):
    pts = tri.points[tri.simplices]  # [S, 3, 2]
    # edges[:, i] is the side opposite vertex i.
    edges = np.stack(
        [
            np.linalg.norm(pts[:, (i + 1) % 3] - pts[:, (i + 2) % 3], axis=1)
            for i in range(3)
        ],
        axis=1,
    )
    sq = edges**2
    return 0.67 * np.sqrt((2 * (sq.sum(axis=1, keepdims=True) - sq) - sq) / 4)


def minimum_support_radius(x):
    """Largest median over the Delaunay triangles adjacent to each point."""
    tri = Delaunay(x)
    medians = _triangulation_medians(tri)
    radii = np.zeros(x.shape[0])
    np.maximum.at(radii, tri.simplices.reshape(-1), medians.reshape(-1))
    return radii


def _supported(centers, points, radii, guard):
    """Edges ``[point, center]`` with ``|point - center| <= radius``."""
    if guard is not None:
        radii = np.where(radii < guard[0], radii, guard[1])
    lists = cKDTree(points).query_ball_point(centers, radii)
    center = np.repeat(np.arange(len(lists)), [len(v) for v in lists])
    point = np.concatenate([np.asarray(v, dtype=np.int64) for v in lists])
    return np.stack([point, center]).astype(np.int64)


def _decoder_edges(centers, points, radii, guard, ensure_coverage):
    """Regional -> point edges ``[center, point]``."""
    edges = _supported(centers, points, radii, guard)
    if ensure_coverage:
        missing = np.setdiff1d(np.arange(points.shape[0]), edges[0])
        if missing.size:
            _, nearest = cKDTree(centers).query(points[missing])
            edges = np.concatenate([edges, np.stack([missing, nearest])], axis=1)
    return edges[::-1].copy()


def _multimesh(x, levels, factor):
    edges = []
    for level in range(levels):
        size = int(x.shape[0] / factor**level)
        if size < 4:
            continue
        indptr, cols = Delaunay(x[:size]).vertex_neighbor_vertices
        rows = np.repeat(np.arange(size), np.diff(indptr))
        edges.append(np.stack([rows, cols]))
    edges = np.concatenate(edges, axis=1)
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    return np.unique(edges, axis=1)


def build_rigno_graph(
    data,
    subsample_factor=4.0,
    overlap_p2r=1.0,
    overlap_r2p=2.0,
    rmesh_levels=4,
    generator=None,
    radius_guard=(0.5, 0.2),
    ensure_coverage=True,
):
    """Return a :class:`RegionGraph` copy of ``data`` with the RIGNO graphs.

    ``generator`` (a ``torch.Generator``) picks the regional nodes; build
    again with a new seed to resample the regional mesh, as upstream does
    every epoch. ``radius_guard=(limit, fallback)`` is upstream's makeshift
    fix for long Delaunay edges across concave regions: support radii
    ``>= limit`` become ``fallback``. Points outside every support region
    (hull corners, or radii cut by the guard) get no decoder message
    upstream; ``ensure_coverage`` links them to their nearest regional
    node instead. 2D only.
    """
    pos = data.pos
    if pos.shape[1] != 2:
        raise ValueError("build_rigno_graph supports 2D coordinates only")
    points = [pos.detach().cpu().double()]
    query = getattr(data, "query_pos", None)
    if query is not None:
        points.append(query.detach().cpu().double()[:, :2])
    lo = torch.cat(points).min(dim=0).values
    scale = (torch.cat(points).max(dim=0).values - lo).max()
    x = (2 * (points[0] - lo) / scale - 1).numpy()

    n = x.shape[0]
    num_r = int(n / subsample_factor)
    if num_r < 4:
        raise ValueError(f"RIGNO needs at least 4 regional nodes, got {num_r}")
    pick = torch.randperm(n, generator=generator)[:num_r].numpy()
    rpos = x[pick]
    radius = minimum_support_radius(rpos)
    r_max = radius.max()

    out = RegionGraph(**{key: value for key, value in data})
    dtype = pos.dtype if pos.is_floating_point() else torch.get_default_dtype()

    def tensor(a, dt=torch.long):
        return torch.as_tensor(a, dtype=dt, device=pos.device)

    out.rigno_x = tensor(x, dtype)
    out.rigno_rpos = tensor(rpos, dtype)
    out.rigno_rradius = tensor(radius, dtype)
    out.rigno_rgraph = tensor(np.zeros(num_r))
    out.rigno_p2r_index = tensor(
        _supported(rpos, x, np.clip(overlap_p2r * radius, 0, r_max), radius_guard)
    )
    out.rigno_r2r_index = tensor(_multimesh(rpos, rmesh_levels, subsample_factor))
    r2p_radius = np.clip(overlap_r2p * radius, 0, r_max)
    out.rigno_r2p_index = tensor(
        _decoder_edges(rpos, x, r2p_radius, radius_guard, ensure_coverage)
    )
    if query is not None:
        qx = (2 * (points[1] - lo) / scale - 1).numpy()
        out.rigno_qx = tensor(qx, dtype)
        out.rigno_qgraph = tensor(np.zeros(qx.shape[0]))
        out.rigno_r2q_index = tensor(
            _decoder_edges(rpos, qx, r2p_radius, radius_guard, ensure_coverage)
        )
    return out
