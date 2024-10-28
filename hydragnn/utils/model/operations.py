##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
from typing import List, Tuple, Optional
import numpy as np
import torch


###########################################################################################
# Function for the computation of edge vectors and lengths (MIT License (see MIT.md))
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
###########################################################################################
def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: Optional[torch.Tensor] = None,  # [n_edges, 3], optional
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]

    # If shifts is None, set it to a zero tensor of the appropriate shape
    if shifts is None:
        shifts = torch.zeros_like(positions[receiver] - positions[sender])

    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def get_pbc_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    supercell_size: torch.Tensor,  # [n_graphs*3, 3], supercell matrices per graph
    batch: torch.Tensor,  # [n_nodes], node-to-graph assignment
    shifts: Optional[torch.Tensor] = None,  # [n_edges, 3], optional
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance vectors between nodes, adjusting for periodic boundary conditions (PBC),
    when each graph in a batch has its own supercell matrix.

    Parameters:
    - positions: Tensor of shape [n_nodes, 3], node positions.
    - edge_index: Tensor of shape [2, n_edges], edge indices.
    - supercell_size: Tensor of shape [n_graphs, 3, 3], supercell matrices per graph.
    - batch: Tensor of shape [n_nodes], node-to-graph assignment.
    - shifts: Optional tensor of shape [n_edges, 3], shifts to apply to the vectors.
    - normalize: Whether to normalize the distance vectors.
    - eps: Small value to prevent division by zero in normalization.

    Returns:
    - vectors: Tensor of shape [n_edges, 3], adjusted for PBCs.
    - lengths: Tensor of shape [n_edges, 1], containing the lengths of the distance vectors.
    """

    # Validate and process supercell_size
    if supercell_size is None:
        raise ValueError("Supercell size must be provided.")

    n_graphs = batch.max().item() + 1
    expected_shape = (n_graphs * 3, 3)
    if supercell_size.shape != expected_shape:
        raise ValueError(
            f"Expected supercell_size to have shape {expected_shape}, but got {supercell_size.shape}"
        )

    # Reshape supercell_size from [n_graphs*3, 3] to [n_graphs, 3, 3]
    supercell_size = supercell_size.view(n_graphs, 3, 3)  # Shape: [n_graphs, 3, 3]
    # Extract cell sizes along each dimension (diagonal elements)
    supercell_size = torch.diagonal(
        supercell_size, dim1=1, dim2=2
    )  # Shape: [n_graphs, 3]

    # Ensure tensors are on the same device
    positions = positions.to(edge_index.device)
    supercell_size = supercell_size.to(positions.device)
    batch = batch.to(positions.device)

    # Setup Data
    sender = edge_index[0]  # Indices of source nodes
    receiver = edge_index[1]  # Indices of target nodes
    vectors = positions[receiver] - positions[sender]  # Shape: (n_edges, 3)
    # Get the graph assignments for each edge based on sender nodes
    edge_batch = batch[
        sender
    ]  # Shape: (n_edges,)   # This is very important to use the correct pbc matrix for each edge!
    supercell_per_edge = supercell_size[edge_batch]  # Shape: (n_edges, 3, 3)
    half_sizes = supercell_per_edge / 2  # Shape: (n_edges, 3)

    # Apply PBCs
    for dim in range(3):
        dim_size = supercell_per_edge[:, dim]
        over_half = vectors[:, dim] > half_sizes[:, dim]
        under_half = vectors[:, dim] < -half_sizes[:, dim]
        vectors[over_half, dim] -= dim_size[over_half]
        vectors[under_half, dim] += dim_size[under_half]

    # Apply shifts if provided
    if shifts is not None:
        vectors = vectors + shifts  # Shape: (n_edges, 3)

    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # Shape: (n_edges, 1)

    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths
