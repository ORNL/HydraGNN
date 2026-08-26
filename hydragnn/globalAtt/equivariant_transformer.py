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
from e3nn import nn, o3

from hydragnn.globalAtt.complete_graph import complete_graph_edge_index
from hydragnn.globalAtt.equivariant_attention import EquivariantAllToAllAttention
from hydragnn.globalAtt.periodic_supercell import build_periodic_attention_sources


class EquivariantRMSNorm(torch.nn.Module):
    """Normalize each node by an invariant RMS with a shared scalar gain.

    A conventional LayerNorm would independently normalize Cartesian tensor
    components and would therefore break rotation equivariance.  The squared
    norm of a complete irrep representation is invariant, so using it as one
    scale per node preserves every declared irrep.
    """

    def __init__(self, irreps: o3.Irreps | str, epsilon: float = 1.0e-8):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if self.irreps.dim == 0:
            raise ValueError("irreps must not be empty")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.irreps.dim:
            raise ValueError(f"features must have shape [N, {self.irreps.dim}]")
        inverse_rms = torch.rsqrt(
            features.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return self.gain * features * inverse_rms


class EquivariantTransformerLayer(torch.nn.Module):
    """Equivariant all-to-all attention and feed-forward residual block.

    The layer uses pre-normalization around both sublayers.  Its feed-forward
    nonlinearity acts on invariant irrep norms and rescales complete irrep
    blocks, rather than applying an elementwise activation to tensor
    components.  Consequently, attention, normalization, feed-forward paths,
    and residual additions all preserve the configured representation.
    """

    def __init__(
        self,
        irreps: o3.Irreps | str,
        heads: int = 1,
        lmax: int = 1,
        num_radial: int = 16,
        feedforward_multiplier: int = 2,
        require_tensor_coupling: bool = True,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        has_tensor_features = any(irrep.l > 0 for _, irrep in self.irreps)
        if require_tensor_coupling and not has_tensor_features:
            raise ValueError(
                "tensor-valued local/global coupling requires at least one "
                "non-scalar input irrep; use require_tensor_coupling=False "
                "only for the acknowledged SchNet/DimeNet scalar-only mode"
            )
        if (
            not isinstance(feedforward_multiplier, int)
            or isinstance(feedforward_multiplier, bool)
            or feedforward_multiplier <= 0
        ):
            raise ValueError("feedforward_multiplier must be a positive integer")
        if chunk_size is not None and (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer or None")
        self.chunk_size = chunk_size

        hidden_irreps = o3.Irreps(
            [
                (multiplicity * feedforward_multiplier, irrep)
                for multiplicity, irrep in self.irreps
            ]
        )
        self.attention_norm = EquivariantRMSNorm(self.irreps)
        self.attention = EquivariantAllToAllAttention(
            self.irreps,
            heads=heads,
            lmax=lmax,
            num_radial=num_radial,
        )
        self.feedforward_norm = EquivariantRMSNorm(self.irreps)
        self.feedforward = torch.nn.Sequential(
            o3.Linear(self.irreps, hidden_irreps, biases=False),
            nn.NormActivation(
                hidden_irreps,
                scalar_nonlinearity=torch.nn.functional.silu,
                normalize=True,
                epsilon=1.0e-8,
                bias=False,
            ),
            o3.Linear(hidden_irreps, self.irreps, biases=False),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update node features using pairs within each graph in ``batch``."""
        node_features = self.forward_attention(
            node_features, positions, batch, edge_index=edge_index
        )
        return self.forward_feedforward(node_features)

    def forward_attention(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the normalized attention residual sublayer."""
        if batch.ndim != 1 or batch.shape[0] != node_features.shape[0]:
            raise ValueError("batch must contain one graph identifier per node")
        if batch.dtype != torch.long:
            raise TypeError("batch must have dtype torch.long")
        if batch.device != node_features.device:
            raise ValueError("batch and node features must share a device")
        if edge_index is None:
            if self.chunk_size is None:
                edge_index = complete_graph_edge_index(batch)
        elif edge_index.ndim == 2 and edge_index.shape[0] == 2:
            if edge_index.dtype != torch.long:
                raise TypeError("edge_index must have dtype torch.long")
            if edge_index.device != batch.device:
                raise ValueError("edge_index and batch must share a device")
            if edge_index.numel() and (
                torch.any(edge_index < 0)
                or torch.any(edge_index >= node_features.shape[0])
            ):
                raise ValueError("edge_index contains an out-of-range node index")
            source, target = edge_index
            if edge_index.numel() and torch.any(source == target):
                raise ValueError("edge_index must not contain self-pairs")
            if edge_index.numel() and torch.any(batch[source] != batch[target]):
                raise ValueError("edge_index must not contain cross-graph pairs")

        normalized = self.attention_norm(node_features)
        if edge_index is None:
            attention_output = self.attention.forward_chunked(
                normalized, positions, batch, self.chunk_size
            )
        else:
            attention_output = self.attention(normalized, positions, edge_index)
        return node_features + attention_output

    def forward_periodic_attention(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        replication: int | tuple[int, int, int] | list[int] = 1,
    ) -> torch.Tensor:
        """Apply central-query attention over explicit periodic images.

        This is intentionally not minimum-image attention. The selected image
        extent defines a finite supercell; every image is a separate softmax
        source, while only original-cell nodes receive residual updates.
        """
        if batch.ndim != 1 or batch.shape[0] != node_features.shape[0]:
            raise ValueError("batch must contain one graph identifier per node")
        normalized = self.attention_norm(node_features)
        sources = build_periodic_attention_sources(
            normalized, positions, batch, cell, pbc, replication
        )
        chunk_size = self.chunk_size or max(node_features.shape[0], 1)
        attention_output = self.attention.forward_bipartite_chunked(
            normalized,
            positions,
            batch,
            sources.features,
            sources.positions,
            sources.batch,
            sources.base_index,
            sources.is_central_image,
            chunk_size,
        )
        return node_features + attention_output

    def forward_feedforward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Apply the normalized equivariant feed-forward residual sublayer."""
        return node_features + self.feedforward(self.feedforward_norm(node_features))
