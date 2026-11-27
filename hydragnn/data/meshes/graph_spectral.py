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
"""Graph Laplacian eigenbasis for the GSNO (offline, once per geometry).

Following the GSNO paper: edge weights ``A_ij = exp(-|x_i - x_j|^2 / s^2)`` on
the mesh edges, ``L = I - D^{-1/2} A D^{-1/2}`` and the ``k`` smallest
eigenpairs. The paper does not state ``s``; the default is the mean edge
length. Eigenvectors are orthonormal in the Euclidean inner product
(``Phi^T Phi = I``) and sign-fixed so their largest-magnitude entry (lowest
index on ties) is positive. Eigenvalues closer than ``degenerate_tol`` (relative
to ``max(1, |lambda|)``) form one eigenspace; ``groups[k]`` is the index of the
first mode of the eigenspace of mode ``k``.
"""

import numpy as np
import scipy.sparse as sp
import torch


def normalized_laplacian(pos, edge_index, sigma=None):
    """``scipy.sparse`` ``I - D^{-1/2} A D^{-1/2}`` with Gaussian edge weights."""
    pos = np.asarray(pos.detach().cpu(), dtype=np.float64)
    edges = edge_index.detach().cpu().numpy()
    edges = np.unique(np.sort(edges[:, edges[0] != edges[1]], axis=0), axis=1)
    length2 = ((pos[edges[0]] - pos[edges[1]]) ** 2).sum(axis=1)
    if sigma is None:
        sigma = float(np.sqrt(length2).mean())
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    weight = np.exp(-length2 / sigma**2)
    n = pos.shape[0]
    adjacency = sp.coo_matrix(
        (
            np.concatenate([weight, weight]),
            (np.r_[edges[0], edges[1]], np.r_[edges[1], edges[0]]),
        ),
        shape=(n, n),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("graph has isolated nodes")
    scale = sp.diags(1.0 / np.sqrt(degree))
    return sp.identity(n, format="csr") - scale @ adjacency @ scale, sigma


def _degenerate_groups(eigenvalues, tol):
    groups = np.arange(len(eigenvalues))
    for k in range(1, len(eigenvalues)):
        gap = eigenvalues[k] - eigenvalues[k - 1]
        if gap <= tol * max(1.0, abs(eigenvalues[k])):
            groups[k] = groups[k - 1]
    return groups


def graph_spectral_basis(
    pos,
    edge_index,
    num_modes,
    sigma=None,
    method="auto",
    degenerate_tol=1e-8,
    seed=0,
):
    """Return ``(eigenvalues [k], basis [N, k], groups [k], info)`` in float64.

    ``method``: ``"dense"`` (``numpy.linalg.eigh``), ``"lobpcg"`` (as in the
    paper; seeded start, ``k + 4`` vectors) or ``"auto"`` (dense up to 2000
    nodes). ``info["split_eigenspace"]`` is True when mode ``k + 1`` is
    degenerate with mode ``k``, i.e. the truncation cuts an eigenspace and the
    basis is not unique up to rotations within whole eigenspaces.
    """
    laplacian, sigma = normalized_laplacian(pos, edge_index, sigma)
    n = laplacian.shape[0]
    if not 1 <= num_modes < n:
        raise ValueError(f"num_modes must be in [1, {n - 1}], got {num_modes}")
    wanted = min(num_modes + 1, n)
    if method == "auto":
        method = "dense" if n <= 2000 else "lobpcg"
    if method == "dense":
        values, vectors = np.linalg.eigh(laplacian.toarray())
        values, vectors = values[:wanted], vectors[:, :wanted]
    elif method == "lobpcg":
        from scipy.sparse.linalg import lobpcg

        size = min(num_modes + 4, n - 1)
        start = np.random.default_rng(seed).standard_normal((n, size))
        values, vectors = lobpcg(
            laplacian, start, largest=False, tol=1e-10, maxiter=2000
        )
        order = np.argsort(values)[:wanted]
        values, vectors = values[order], vectors[:, order]
        vectors, _ = np.linalg.qr(vectors)
    else:
        raise ValueError("method must be 'auto', 'dense' or 'lobpcg'")
    groups = _degenerate_groups(values, degenerate_tol)
    split = len(values) > num_modes and groups[num_modes] < num_modes
    values, vectors, groups = (
        values[:num_modes],
        vectors[:, :num_modes],
        groups[:num_modes],
    )
    peak = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[peak, np.arange(num_modes)])
    vectors = vectors * np.where(signs == 0, 1.0, signs)
    info = {"sigma": sigma, "method": method, "split_eigenspace": bool(split)}
    return (
        torch.from_numpy(values),
        torch.from_numpy(np.ascontiguousarray(vectors)),
        torch.from_numpy(groups),
        info,
    )


def attach_graph_spectral_basis(data, num_modes, sigma=None, method="auto", dtype=None):
    """Store ``gsno_basis [N, k]``, ``gsno_eigenvalues [1, k]`` and
    ``gsno_groups [1, k]`` on ``data`` (graph-level tensors batch to ``[G, k]``)."""
    values, basis, groups, info = graph_spectral_basis(
        data.pos, data.edge_index, num_modes, sigma=sigma, method=method
    )
    dtype = dtype or data.pos.dtype
    data.gsno_basis = basis.to(dtype)
    data.gsno_eigenvalues = values.to(dtype).unsqueeze(0)
    data.gsno_groups = groups.unsqueeze(0)
    return info
