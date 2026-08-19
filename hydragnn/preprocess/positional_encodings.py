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

from typing import Dict, Optional

import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE


class AddEmptyPE:
    """Attach an empty positional encoding tensor when pe_dim is disabled."""

    def __init__(self, attr_name: str = "pe"):
        self.attr_name = attr_name

    def __call__(self, data):
        num_nodes = data.num_nodes
        setattr(data, self.attr_name, torch.empty((num_nodes, 0), dtype=torch.float32))
        return data


class AddCommunicabilityPE:
    """Compute communicability-inspired node positional encodings.

    Two methods are supported:
    - "katz": multi-scale Katz-style centrality vectors with linearly spaced alpha.
    - "adjacency_powers": repeated powers of normalized adjacency applied to ones.
    """

    def __init__(
        self,
        k: int,
        attr_name: str = "pe",
        method: str = "katz",
        alpha_min: float = 0.05,
        alpha_max: float = 0.5,
        degree_normalize: bool = True,
        eps: float = 1.0e-8,
    ):
        self.k = int(k)
        self.attr_name = attr_name
        self.method = method
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.degree_normalize = bool(degree_normalize)
        self.eps = float(eps)

    def _build_adjacency(self, data) -> torch.Tensor:
        num_nodes = int(data.num_nodes)
        edge_index = data.edge_index
        device = edge_index.device
        dtype = torch.float32

        adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
        if edge_index.numel() == 0:
            return adjacency

        src, dst = edge_index
        weight = torch.ones(src.shape[0], dtype=dtype, device=device)
        adjacency.index_put_((src, dst), weight, accumulate=True)

        # Enforce undirected connectivity for a stable graph-level PE.
        adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
        adjacency.fill_diagonal_(0.0)

        if self.degree_normalize:
            deg = adjacency.sum(dim=1)
            inv_sqrt = torch.rsqrt(torch.clamp(deg, min=self.eps))
            adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]

        return adjacency

    def _spectral_radius(self, adjacency: torch.Tensor) -> float:
        if adjacency.numel() == 0 or adjacency.shape[0] == 0:
            return 0.0
        eigvals = torch.linalg.eigvals(adjacency)
        return float(torch.max(torch.abs(eigvals)).real)

    def _katz_features(self, adjacency: torch.Tensor) -> torch.Tensor:
        num_nodes = adjacency.shape[0]
        pe = torch.zeros((num_nodes, self.k), dtype=adjacency.dtype, device=adjacency.device)
        ones = torch.ones((num_nodes,), dtype=adjacency.dtype, device=adjacency.device)
        eye = torch.eye(num_nodes, dtype=adjacency.dtype, device=adjacency.device)

        rho = max(self._spectral_radius(adjacency), self.eps)
        alpha_ceiling = 0.99 / rho
        alphas = torch.linspace(
            self.alpha_min,
            self.alpha_max,
            self.k,
            dtype=adjacency.dtype,
            device=adjacency.device,
        )

        for idx, alpha in enumerate(alphas):
            alpha_eff = min(float(alpha), alpha_ceiling)
            system = eye - alpha_eff * adjacency
            try:
                feature = torch.linalg.solve(system, ones)
            except RuntimeError:
                # Fallback to a short Neumann series if solve is numerically unstable.
                term = ones.clone()
                feature = ones.clone()
                for hop in range(1, 25):
                    term = (alpha_eff / float(hop)) * (adjacency @ term)
                    feature = feature + term
            pe[:, idx] = feature

        return pe

    def _adjacency_power_features(self, adjacency: torch.Tensor) -> torch.Tensor:
        num_nodes = adjacency.shape[0]
        pe = torch.zeros((num_nodes, self.k), dtype=adjacency.dtype, device=adjacency.device)
        signal = torch.ones((num_nodes,), dtype=adjacency.dtype, device=adjacency.device)
        for idx in range(self.k):
            signal = adjacency @ signal
            pe[:, idx] = signal
        return pe

    def __call__(self, data):
        num_nodes = int(data.num_nodes)
        if self.k <= 0:
            setattr(
                data,
                self.attr_name,
                torch.empty((num_nodes, 0), dtype=torch.float32, device=data.edge_index.device),
            )
            return data

        adjacency = self._build_adjacency(data)
        if num_nodes == 0:
            setattr(data, self.attr_name, torch.empty((0, self.k), dtype=torch.float32))
            return data

        if self.method == "katz":
            pe = self._katz_features(adjacency)
        elif self.method == "adjacency_powers":
            pe = self._adjacency_power_features(adjacency)
        else:
            raise ValueError(
                f"Unknown communicability method '{self.method}'. Expected one of ['katz', 'adjacency_powers']."
            )

        # Normalize feature scales to improve optimizer stability across datasets.
        pe = pe - pe.mean(dim=0, keepdim=True)
        pe = pe / torch.clamp(pe.std(dim=0, keepdim=True), min=self.eps)
        setattr(data, self.attr_name, pe)
        return data


def create_positional_encoder(architecture_config: Optional[Dict]) -> object:
    """Factory for node positional encoders used during preprocessing."""

    arch = architecture_config or {}
    pe_dim = int(arch.get("pe_dim", 0))
    pe_encoder = str(arch.get("pe_encoder", "laplacian")).lower()

    if pe_dim <= 0 or pe_encoder in {"none", "disabled"}:
        return AddEmptyPE(attr_name="pe")

    if pe_encoder == "laplacian":
        return AddLaplacianEigenvectorPE(
            k=pe_dim,
            attr_name="pe",
            is_undirected=True,
        )

    if pe_encoder == "communicability":
        return AddCommunicabilityPE(
            k=pe_dim,
            attr_name="pe",
            method=str(arch.get("communicability_method", "katz")).lower(),
            alpha_min=float(arch.get("communicability_alpha_min", 0.05)),
            alpha_max=float(arch.get("communicability_alpha_max", 0.5)),
            degree_normalize=bool(arch.get("communicability_degree_normalize", True)),
            eps=float(arch.get("communicability_eps", 1.0e-8)),
        )

    raise ValueError(
        f"Unsupported pe_encoder '{pe_encoder}'. Expected one of ['laplacian', 'communicability', 'none']."
    )


def add_relative_pe(data, pe_attr_name: str = "pe", rel_attr_name: str = "rel_pe"):
    """Attach edge-wise relative positional encodings from a node-level PE tensor."""

    pe = getattr(data, pe_attr_name)
    source_pe = pe[data.edge_index[0]]
    target_pe = pe[data.edge_index[1]]
    setattr(data, rel_attr_name, torch.abs(source_pe - target_pe))
    return data