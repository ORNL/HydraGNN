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
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from hydragnn.utils.model.allscaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from hydragnn.utils.model.allscaip.custom_types import GraphAttentionData

from typing import Literal

from hydragnn.utils.model.escaip.utils.nn_utils import (
    Activation,
    NormalizationType,
    get_linear,
    get_normalization_layer,
)


class ChgSpinEmbedding(nn.Module):
    """Vendored from ``fairchem.core.models.uma.nn.embedding.ChgSpinEmbedding``."""

    def __init__(
        self,
        embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"],
        embedding_target: Literal["charge", "spin"],
        embedding_size: int,
        grad: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type
        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target
        assert embedding_size % 2 == 0, f"{embedding_size=} must be even"

        if self.embedding_target == "charge":
            self._index_offset = 100
            self._num_embeddings = 201
        elif self.embedding_target == "spin":
            self._index_offset = 0
            self._num_embeddings = 101

        if self.embedding_type == "pos_emb":
            self.W = nn.Parameter(
                torch.randn(embedding_size // 2) * scale, requires_grad=grad
            )
        elif self.embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(in_features=1, out_features=embedding_size)
            if not grad:
                for param in self.lin_emb.parameters():
                    param.requires_grad = False
        elif self.embedding_type == "rand_emb":
            self.rand_emb = nn.Embedding(self._num_embeddings, embedding_size)
            if not grad:
                for param in self.rand_emb.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"embedding type {self.embedding_type} not implemented")

    def forward(self, x):
        if self.embedding_type == "pos_emb":
            x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
            if self.embedding_target == "charge":
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            elif self.embedding_target == "spin":
                zero_idxs = torch.where(x == 0)[0]
                emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
                emb[zero_idxs] = 0
                return emb
        elif self.embedding_type == "lin_emb":
            if self.embedding_target == "spin":
                x[x == 0] = -100
            return self.lin_emb(x.unsqueeze(-1).float())
        elif self.embedding_type == "rand_emb":
            indices = x + self._index_offset
            return self.rand_emb(indices.long())
        raise ValueError(f"embedding type {self.embedding_type} not implemented")


class InputBlock(nn.Module):
    """
    Featurize the input data into edge and global embeddings.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Atomic number embeddings
        # ref: escn https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/escn/escn.py#L823
        self.atomic_embedding = nn.Embedding(
            molecular_graph_cfg.max_num_elements, global_cfg.hidden_size
        )
        nn.init.uniform_(self.atomic_embedding.weight.data, -0.001, 0.001)

        # Node direction embedding
        self.node_direction_embedding = get_linear(
            in_features=gnn_cfg.node_direction_expansion_size,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.node_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.node_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size * 2)

        # Edge attribute linear
        self.edge_attr_linear = get_linear(
            in_features=gnn_cfg.edge_distance_expansion_size
            + (gnn_cfg.edge_direction_expansion_size) ** 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.edge_feature_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )

        # charge / spin embedding
        self.charge_embedding = ChgSpinEmbedding(
            "rand_emb",
            "charge",
            global_cfg.hidden_size,
            grad=True,
        )
        self.spin_embedding = ChgSpinEmbedding(
            "rand_emb",
            "spin",
            global_cfg.hidden_size,
            grad=True,
        )
        self.charge_spin_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )

    def forward(self, data: GraphAttentionData):
        # neighbor embeddings
        atomic_embedding = self.atomic_embedding(data.atomic_numbers)
        node_direction_embedding = self.node_direction_embedding(
            data.node_direction_expansion
        )
        node_embeddings = torch.cat(
            [atomic_embedding, node_direction_embedding], dim=-1
        )
        node_embeddings = self.node_linear(self.node_norm(node_embeddings))
        # node_embeddings: (num_nodes, hidden_dim)

        # charge / spin embedding
        charge_embedding = self.charge_embedding(data.charge)
        spin_embedding = self.spin_embedding(data.spin)
        charge_spin_embeddings = self.charge_spin_linear(
            torch.cat([charge_embedding, spin_embedding], dim=-1)
        )
        # charge_spin_embeddings: (num_graphs, hidden_dim)
        node_embeddings = node_embeddings + charge_spin_embeddings[data.node_batch]

        # neighbor embedding
        edge_attr = self.edge_attr_linear(
            torch.cat(
                [data.edge_distance_expansion, data.edge_direction_expansion], dim=-1
            )
        )
        neighbor_embeddings = self.edge_feature_linear(
            torch.cat([node_embeddings[data.neighbor_index[0]], edge_attr], dim=-1)
        )

        return neighbor_embeddings
