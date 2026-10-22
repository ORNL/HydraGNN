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
"""Autograd derivatives of model energies."""

import torch


def compute_forces_and_hessian(
    energy, positions, *, compute_hessian=False, create_graph=False
):
    """Differentiate scalar graph energy into forces and a Cartesian Hessian."""
    forces = -torch.autograd.grad(
        energy,
        positions,
        grad_outputs=torch.ones_like(energy),
        retain_graph=create_graph or compute_hessian,
        create_graph=create_graph or compute_hessian,
    )[0]
    if not compute_hessian:
        return forces, None

    rows = []
    for component in forces.reshape(-1):
        force_gradient = torch.autograd.grad(
            component,
            positions,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        rows.append(-force_gradient.reshape(-1))
    return forces, torch.stack(rows)
