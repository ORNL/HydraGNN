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
"""Autoregressive rollout for one-step PDE surrogates."""

import torch

_INPUT_ATTR = {"node": "x", "cell": "cell_x"}


@torch.no_grad()
def rollout(model, data, num_steps, columns, level="node", head=0, residual=False):
    """Feed head ``head`` back into input ``columns`` for ``num_steps`` steps.

    ``level`` selects the compiled input tensor (``"node"`` -> ``data.x``,
    ``"cell"`` -> ``data.cell_x``). With ``residual`` the head predicts the
    increment, otherwise the next state. Returns the ``num_steps`` predicted
    states. ``data`` is not modified; call ``model.eval()`` first.
    """
    if level not in _INPUT_ATTR:
        raise ValueError(f"rollout level must be one of {sorted(_INPUT_ATTR)}")
    attr = _INPUT_ATTR[level]
    columns = list(columns)
    data = data.clone()
    states = []
    for _ in range(num_steps):
        out = model(data)
        if getattr(model, "var_output", False):
            out = out[0]
        out = out[head]
        x = getattr(data, attr).clone()
        if out.shape != x[:, columns].shape:
            raise ValueError(
                f"head {head} has shape {tuple(out.shape)}, but the rolled-out "
                f"columns have shape {tuple(x[:, columns].shape)}"
            )
        state = x[:, columns] + out if residual else out
        x[:, columns] = state
        setattr(data, attr, x)
        states.append(state)
    return states
