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

import math

import torch
from e3nn import o3
from torch_geometric.utils import softmax


class EquivariantAllToAllAttention(torch.nn.Module):
    """SE(3)-equivariant attention over an explicitly supplied edge set.

    Queries and keys retain the input irreps. Their inner product is invariant
    and therefore suitable as an attention logit. Values couple source-node
    features to spherical harmonics of ``position[source] - position[target]``
    through a fully connected tensor product. Invariant attention weights are
    normalized over the sources of each target node.

    Pair construction is intentionally separate so this kernel can later be
    used by both dense and chunked execution paths.
    """

    def __init__(
        self,
        irreps: o3.Irreps | str,
        heads: int = 1,
        lmax: int = 1,
        num_radial: int = 16,
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if self.irreps.dim == 0:
            raise ValueError("irreps must not be empty")
        if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
            raise ValueError("heads must be a positive integer")
        if not isinstance(lmax, int) or isinstance(lmax, bool) or lmax < 0:
            raise ValueError("lmax must be a nonnegative integer")
        if (
            not isinstance(num_radial, int)
            or isinstance(num_radial, bool)
            or num_radial <= 0
        ):
            raise ValueError("num_radial must be a positive integer")

        self.heads = heads
        self.num_radial = num_radial
        self.head_irreps = o3.Irreps(" + ".join([str(self.irreps)] * heads))
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)

        self.query = o3.Linear(self.irreps, self.head_irreps, biases=False)
        self.key = o3.Linear(self.irreps, self.head_irreps, biases=False)
        self.value_tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.sh_irreps,
            self.irreps,
            shared_weights=False,
        )
        self.radial_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_radial, num_radial),
            torch.nn.SiLU(),
            torch.nn.Linear(num_radial, heads * self.value_tensor_product.weight_numel),
        )
        self.output = o3.Linear(self.head_irreps, self.irreps, biases=False)
        self.register_buffer("radial_centers", torch.linspace(0.0, 1.0, num_radial))

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply attention, returning one feature tensor per input node."""
        self._validate_inputs(node_features, positions, edge_index)
        num_nodes = node_features.shape[0]
        source, target = edge_index

        if edge_index.shape[1] == 0:
            result = node_features.new_zeros((num_nodes, self.irreps.dim))
            if return_attention_weights:
                return result, node_features.new_zeros((0, self.heads))
            return result

        queries = self.query(node_features).reshape(
            num_nodes, self.heads, self.irreps.dim
        )
        keys = self.key(node_features).reshape(num_nodes, self.heads, self.irreps.dim)
        logits = (queries[target] * keys[source]).sum(dim=-1)
        logits = logits / math.sqrt(self.irreps.dim)
        attention = softmax(logits, index=target, num_nodes=num_nodes)

        relative = positions[source] - positions[target]
        distances = torch.linalg.vector_norm(relative, dim=-1)
        spherical_harmonics = o3.spherical_harmonics(
            self.sh_irreps,
            relative,
            normalize=True,
            normalization="component",
        )
        radial = self._radial_basis(distances)
        weights = self.radial_mlp(radial).reshape(
            edge_index.shape[1], self.heads, self.value_tensor_product.weight_numel
        )

        source_features = node_features[source]
        values = torch.stack(
            [
                self.value_tensor_product(
                    source_features, spherical_harmonics, weights[:, head]
                )
                for head in range(self.heads)
            ],
            dim=1,
        )
        messages = attention.unsqueeze(-1) * values
        aggregated = node_features.new_zeros((num_nodes, self.heads, self.irreps.dim))
        aggregated.index_add_(0, target, messages)
        result = self.output(aggregated.reshape(num_nodes, self.head_irreps.dim))

        if return_attention_weights:
            return result, attention
        return result

    def _radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        normalized = distances / (1.0 + distances)
        width = max(self.num_radial - 1, 1)
        return torch.exp(
            -(width**2) * (normalized.unsqueeze(-1) - self.radial_centers) ** 2
        )

    def _validate_inputs(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        num_nodes = node_features.shape[0] if node_features.ndim > 0 else 0
        if node_features.ndim != 2 or node_features.shape[1] != self.irreps.dim:
            raise ValueError(f"node_features must have shape [N, {self.irreps.dim}]")
        if positions.shape != (num_nodes, 3):
            raise ValueError(f"positions must have shape ({num_nodes}, 3)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if (
            node_features.device != positions.device
            or positions.device != edge_index.device
        ):
            raise ValueError("features, positions, and edge_index must share a device")
        if node_features.dtype != positions.dtype:
            raise ValueError("features and positions must have the same dtype")
        if edge_index.numel() and (
            torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes)
        ):
            raise ValueError("edge_index contains an out-of-range node index")
