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
        acc = {
            spec.name: [0, 0.0, 0.0, math.inf, -math.inf] for spec in specs
        }  # count, sum, sum of squares, min, max
        if specs:
            for data in dataset:
                for spec in specs:
                    value = data[spec.name].detach().to(torch.float64)
                    if value.numel() == 0:
                        continue
                    a = acc[spec.name]
                    a[0] += value.numel()
                    a[1] += float(value.sum())
                    a[2] += float((value * value).sum())
                    a[3] = min(a[3], float(value.min()))
                    a[4] = max(a[4], float(value.max()))
        acc = _reduce(acc, group)

        stats = {}
        for spec in specs:
            count, total, total_sq, vmin, vmax = acc[spec.name]
            if count == 0:
                raise ValueError(f"No training values found for '{spec.name}'")
            mean = total / count
            if spec.normalize == "minmax":
                shift, scale = vmin, vmax - vmin
            elif spec.normalize == "standard":
                shift, scale = mean, math.sqrt(max(total_sq / count - mean**2, 0.0))
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
    return {
        name: [
            sum(g[name][0] for g in gathered),
            sum(g[name][1] for g in gathered),
            sum(g[name][2] for g in gathered),
            min(g[name][3] for g in gathered),
            max(g[name][4] for g in gathered),
        ]
        for name in acc
    }
