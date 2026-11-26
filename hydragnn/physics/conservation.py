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
"""Finite-volume bookkeeping on :class:`~hydragnn.data.meshes.MeshData`.

Cell-centred update for one step of length ``dt``::

    u_i^{n+1} = u_i^n - dt / V_i * (sum_f s_if A_f F_f + sum_b A_b F_b) + dt s_i

``F_f`` is the flux density through interior face ``f`` along its normal
(owner -> neighbour), so ``s_if = +1`` for the owner and ``-1`` for the
neighbour and every interior face cancels in the global sum. ``F_b`` is the
outward flux density through boundary face ``b``. Summing ``V_i`` times the
update over a mesh gives the discrete balance checked by
:func:`balance_residual`. Conservation alone does not imply positivity or
entropy stability.
"""

import torch

from hydragnn.data.meshes.mesh_data import cell_batch


def net_outflow(data, face_flux, bface_flux=None):
    """``sum_f s_if A_f F_f (+ sum_b A_b F_b)`` per cell, shape ``[C, k]``."""
    out = face_flux.new_zeros(data.num_cells, face_flux.shape[1])
    flux = data.face_area.to(face_flux.dtype) * face_flux
    owner, neighbour = data.face_index
    out.index_add_(0, owner, flux)
    out.index_add_(0, neighbour, -flux)
    if bface_flux is not None:
        out.index_add_(
            0, data.bface_owner, data.bface_area.to(bface_flux.dtype) * bface_flux
        )
    return out


def fv_increment(data, face_flux, bface_flux=None, source=None, dt=1.0):
    """``u^{n+1} - u^n`` of the explicit finite-volume update, shape ``[C, k]``."""
    volume = data.cell_volume.to(face_flux.dtype)
    du = -dt * net_outflow(data, face_flux, bface_flux) / volume
    if source is not None:
        du = du + dt * source
    return du


def balance_residual(data, u_old, u_new, bface_flux=None, source=None, dt=1.0):
    """Per-graph discrete balance error and its scale, both ``[B, k]``.

    ``residual = sum V (u_new - u_old) + dt sum A_b F_b - dt sum V s``, which
    is zero for any conservative update; ``scale = sum V |u_old|`` turns it
    into a relative error.
    """
    volume = data.cell_volume.to(u_old.dtype)
    batch = cell_batch(data)
    num_graphs = int(batch.max()) + 1 if batch.numel() else 0
    k = u_old.shape[1]
    residual = u_old.new_zeros(num_graphs, k)
    residual.index_add_(0, batch, volume * (u_new - u_old))
    if bface_flux is not None:
        flux = dt * data.bface_area.to(u_old.dtype) * bface_flux
        residual.index_add_(0, batch[data.bface_owner], flux)
    if source is not None:
        residual.index_add_(0, batch, -dt * volume * source)
    scale = u_old.new_zeros(num_graphs, k).index_add_(0, batch, volume * u_old.abs())
    return residual, scale
