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
"""Per-variable normalization of named fields.

Each variable with ``normalize`` other than ``"none"`` is mapped to
``(value - shift) / scale`` with one scalar shift and scale per field, so
vector components share a scale:

- ``minmax``: shift = min, scale = max - min
- ``standard``: shift = mean, scale = std
- ``scale_only``: shift = 0, scale = RMS (preserves zero, sign, antisymmetry)

Statistics are fitted on the training split only and reduced across MPI ranks,
which gives identical results for replicated and sharded datasets.
"""

import math

import torch
import torch.distributed as dist
from mpi4py import MPI

from hydragnn.config.variable_schema import VariableSchema


class FieldNormalizer:
    def __init__(self, stats):
        # stats: {name: {"policy": str, "shift": float, "scale": float}}
        self.stats = {name: dict(entry) for name, entry in stats.items()}

    @staticmethod
    def specs(schema: VariableSchema):
        # transform() is keyed by name, so a field listed as both input and
        # output must use a single policy (including "none").
        policies = {}
        specs = []
        for spec in (*schema.inputs, *schema.outputs):
            previous = policies.setdefault(spec.name, spec.normalize)
            if previous != spec.normalize:
                raise ValueError(
                    f"Variable '{spec.name}' has conflicting normalize policies "
                    f"'{previous}' and '{spec.normalize}'"
                )
            if spec.normalize == "none" or any(s.name == spec.name for s in specs):
                continue
            if spec.node_type is not None:
                raise ValueError(
                    f"normalize is not supported for heterogeneous variable '{spec.name}'"
                )
            specs.append(spec)
        return specs

    @classmethod
    def fit(cls, dataset, schema: VariableSchema, group=None):
        """Fit statistics on ``dataset``, reduced over ``group``.

        ``group`` is a torch ``ProcessGroup`` or an ``MPI.Comm``; ``None``
        means the whole world when torch.distributed is initialized.
        """
        specs = cls.specs(schema)
        # count, mean, centered sum of squares (M2), sum of squares, min, max.
        # Centered moments are merged with Chan's update to avoid the
        # cancellation of E[x^2] - E[x]^2.
        acc = {spec.name: _empty() for spec in specs}
        if specs:
            for data in dataset:
                for spec in specs:
                    value = data[spec.name].detach().to(torch.float64)
                    if value.numel() == 0:
                        continue
                    mean = float(value.mean())
                    local = [
                        value.numel(),
                        mean,
                        float(((value - mean) ** 2).sum()),
                        float((value * value).sum()),
                        float(value.min()),
                        float(value.max()),
                    ]
                    acc[spec.name] = _merge(acc[spec.name], local)
        acc = _reduce(acc, group)

        stats = {}
        for spec in specs:
            count, mean, m2, total_sq, vmin, vmax = acc[spec.name]
            if count == 0:
                raise ValueError(f"No training values found for '{spec.name}'")
            if spec.normalize == "minmax":
                shift, scale = vmin, vmax - vmin
            elif spec.normalize == "standard":
                shift, scale = mean, math.sqrt(max(m2 / count, 0.0))
            else:
                shift, scale = 0.0, math.sqrt(total_sq / count)
            stats[spec.name] = {
                "policy": spec.normalize,
                "shift": shift,
                "scale": scale if scale > 0.0 else 1.0,
            }
        return cls(stats)

    def transform(self, data):
        """Normalize the named fields of ``data`` in place and return it."""
        for name, entry in self.stats.items():
            data[name] = (data[name] - entry["shift"]) / entry["scale"]
        return data

    def inverse(self, name, value):
        """Map a normalized tensor of variable ``name`` back to physical units."""
        entry = self.stats.get(name)
        if entry is None:
            return value
        return value * entry["scale"] + entry["shift"]

    def state_dict(self):
        return {name: dict(entry) for name, entry in self.stats.items()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state)


def _empty():
    return [0, 0.0, 0.0, 0.0, math.inf, -math.inf]


def _merge(a, b):
    if a[0] == 0:
        return list(b)
    if b[0] == 0:
        return list(a)
    count = a[0] + b[0]
    delta = b[1] - a[1]
    return [
        count,
        a[1] + delta * b[0] / count,
        a[2] + b[2] + delta * delta * a[0] * b[0] / count,
        a[3] + b[3],
        min(a[4], b[4]),
        max(a[5], b[5]),
    ]


def _reduce(acc, group):
    if group is None:
        if not dist.is_initialized():
            return acc
        group = MPI.COMM_WORLD
    if isinstance(group, MPI.Comm):
        gathered = group.allgather(acc)
    else:
        gathered = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, acc, group=group)
    merged = {}
    for name in acc:
        total = _empty()
        for g in gathered:
            total = _merge(total, g[name])
        merged[name] = total
    return merged
