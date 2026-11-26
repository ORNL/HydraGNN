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
"""Time series of fields on a fixed mesh, sampled as training pairs."""

import torch

_RESERVED = frozenset({"time", "dt"})


class Trajectory:
    """Fields ``[T, rows, dim]`` on one static :class:`MeshData` at ``times [T]``.

    The mesh is shared, not copied, so a trajectory costs one mesh plus the
    field history. ``static`` fields (e.g. sources, coefficients) are attached
    to every sample unchanged.
    """

    def __init__(self, mesh, times, fields, static=None, target_suffix="_next"):
        times = torch.as_tensor(times, dtype=torch.float64)
        if times.dim() != 1 or times.numel() < 2:
            raise ValueError("times must be a 1D tensor with at least 2 entries")
        if not bool((times[1:] > times[:-1]).all()):
            raise ValueError("times must be strictly increasing")
        if not fields:
            raise ValueError("a trajectory needs at least one field")
        for name, value in fields.items():
            if name in _RESERVED or name in mesh:
                raise ValueError(f"field name {name!r} clashes with mesh/time keys")
            if value.dim() != 3 or value.shape[0] != times.numel():
                raise ValueError(
                    f"field {name!r} must have shape [T={times.numel()}, rows, dim]; "
                    f"got {tuple(value.shape)}"
                )
        static = dict(static or {})
        for name in static:
            if name in _RESERVED or name in mesh or name in fields:
                raise ValueError(
                    f"static name {name!r} clashes with a field or mesh/time key"
                )
        self.mesh = mesh
        self.times = times
        self.fields = dict(fields)
        self.static = static
        self.target_suffix = target_suffix
        self._target_names(target_suffix)

    def _target_names(self, suffix):
        if not isinstance(suffix, str) or not suffix:
            raise ValueError("target_suffix must be a non-empty string")
        taken = set(self.fields) | set(self.static) | _RESERVED | set(self.mesh.keys())
        targets = [name + suffix for name in self.fields]
        clash = sorted(set(targets) & taken)
        if clash:
            raise ValueError(f"target names {clash} clash with existing keys")
        return targets

    def __len__(self):
        return self.times.numel()

    def snapshot(self, t):
        """Mesh with every field at step ``t`` and graph-level ``time``."""
        data = self.mesh.clone()
        for name, value in self.fields.items():
            data[name] = value[t]
        for name, value in self.static.items():
            data[name] = value
        data.time = self.times[t].view(1, 1)
        return data

    def pairs(self, horizon=1, stride=1, target_suffix=None):
        """One sample per step ``t``: fields at ``t``, targets at ``t + horizon``.

        Targets are stored as ``<name><target_suffix>`` (default: the suffix
        given at construction); ``time`` is the input time and
        ``dt = times[t + horizon] - times[t]``.
        """
        if horizon < 1 or stride < 1:
            raise ValueError("horizon and stride must be positive")
        suffix = self.target_suffix if target_suffix is None else target_suffix
        targets = self._target_names(suffix)
        samples = []
        for t in range(0, len(self) - horizon, stride):
            data = self.snapshot(t)
            for target, value in zip(targets, self.fields.values()):
                data[target] = value[t + horizon]
            data.dt = (self.times[t + horizon] - self.times[t]).view(1, 1)
            samples.append(data)
        return samples
