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
"""Graph-level pooling shared by HydraGNN models and decoders."""

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

_POOL_MAP = {
    "mean": (global_mean_pool, "mean"),
    "add": (global_add_pool, "sum"),
    "max": (global_max_pool, "max"),
}


def resolve_graph_pooling(graph_pooling: str):
    """Return ``(mode, (pool_fn, reduction))`` for a pooling name.

    ``"sum"`` is accepted as an alias for ``"add"``.
    """
    mode = graph_pooling.lower()
    if mode == "sum":
        mode = "add"
    if mode not in _POOL_MAP:
        raise ValueError("Unsupported graph_pooling: " + graph_pooling)
    return mode, _POOL_MAP[mode]


class GraphPoolingMixin:
    """Provides ``_pool_graph_features``.

    Requires ``graph_pool_fn`` and ``graph_pool_reduction`` attributes, as set
    by :meth:`_configure_graph_pooling`.
    """

    def _configure_graph_pooling(self, graph_pooling: str):
        self.graph_pooling, (
            self.graph_pool_fn,
            self.graph_pool_reduction,
        ) = resolve_graph_pooling(graph_pooling)

    def _pool_graph_features(self, x_tensor, batch_tensor, size=None):
        if batch_tensor is None:
            if self.graph_pool_reduction == "mean":
                return x_tensor.mean(dim=0, keepdim=True)
            if self.graph_pool_reduction == "max":
                return x_tensor.max(dim=0, keepdim=True).values
            return x_tensor.sum(dim=0, keepdim=True)
        return self.graph_pool_fn(x_tensor, batch_tensor.to(x_tensor.device), size=size)
