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

import torch


def complete_graph_edge_index(batch: torch.Tensor) -> torch.Tensor:
    """Build all ordered non-self node pairs independently in each graph.

    Args:
        batch: One-dimensional integer tensor assigning every node to a graph.
            Graph identifiers need not be contiguous.

    Returns:
        A standard PyTorch Geometric ``edge_index`` tensor with shape
        ``[2, E]``.  Row zero contains source nodes and row one contains target
        nodes.  Pairs are ordered by graph identifier, target node, then source
        node.  Singleton graphs contribute no pairs.

    This function deliberately excludes self-pairs: the equivariant
    Transformer handles a node's own state through its residual connection.
    """
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a torch.Tensor")
    if batch.ndim != 1:
        raise ValueError(
            f"batch must be one-dimensional, but has shape {tuple(batch.shape)}"
        )
    if batch.dtype != torch.long:
        raise TypeError(f"batch must have dtype torch.long, but has {batch.dtype}")
    if torch.any(batch < 0):
        raise ValueError("batch graph identifiers must be nonnegative")

    pairs = []
    for graph_id in torch.unique(batch, sorted=True):
        nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        target = nodes.repeat_interleave(nodes.numel())
        source = nodes.repeat(nodes.numel())
        non_self = source != target
        pairs.append(torch.stack((source[non_self], target[non_self])))

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(pairs, dim=1)
