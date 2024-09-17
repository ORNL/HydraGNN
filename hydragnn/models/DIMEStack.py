##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

from typing import Callable, Optional, Tuple
from torch_geometric.typing import SparseTensor

import torch
from torch import Tensor
from torch.nn import Identity, SiLU

from torch_geometric.nn import Linear, Sequential
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    EmbeddingBlock,
    InteractionPPBlock,
    OutputPPBlock,
    SphericalBasisLayer,
)
from torch_geometric.utils import scatter

from .Base import Base


class DIMEStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        basis_emb_size,
        envelope_exponent,
        int_emb_size,
        out_emb_size,
        num_after_skip,
        num_before_skip,
        num_radial,
        num_spherical,
        radius,
        *args,
        max_neighbours: Optional[int] = None,
        **kwargs
    ):
        self.basis_emb_size = basis_emb_size
        self.int_emb_size = int_emb_size
        self.out_emb_size = out_emb_size
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.radius = radius

        super().__init__(*args, **kwargs)

        self.rbf = BesselBasisLayer(num_radial, radius, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, radius, envelope_exponent
        )

        pass

    def _init_conv(self):
        self.graph_convs.append(self.get_conv(self.input_dim, self.hidden_dim))
        self.feature_layers.append(Identity())
        for _ in range(self.num_conv_layers - 1):
            conv = self.get_conv(self.hidden_dim, self.hidden_dim)
            self.graph_convs.append(conv)
            self.feature_layers.append(Identity())

    def get_conv(self, input_dim, output_dim):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        assert (
            hidden_dim > 1
        ), "DimeNet requires more than one hidden dimension between input_dim and output_dim."
        lin = Linear(input_dim, hidden_dim)
        emb = HydraEmbeddingBlock(
            num_radial=self.num_radial, hidden_channels=hidden_dim, act=SiLU()
        )
        inter = InteractionPPBlock(
            hidden_channels=hidden_dim,
            int_emb_size=self.int_emb_size,
            basis_emb_size=self.basis_emb_size,
            num_spherical=self.num_spherical,
            num_radial=self.num_radial,
            num_before_skip=self.num_before_skip,
            num_after_skip=self.num_after_skip,
            act=SiLU(),
        )
        dec = OutputPPBlock(
            num_radial=self.num_radial,
            hidden_channels=hidden_dim,
            out_emb_channels=self.out_emb_size,
            out_channels=output_dim,
            num_layers=1,
            act=SiLU(),
            output_initializer="glorot_orthogonal",
        )
        return Sequential(
            "x, pos, rbf, sbf, i, j, idx_kj, idx_ji",
            [
                (lin, "x -> x"),
                (emb, "x, rbf, i, j -> x1"),
                (inter, "x1, rbf, sbf, idx_kj, idx_ji -> x2"),
                (dec, "x2, rbf, i -> c"),
                (lambda x, pos: [x, pos], "c, pos -> c, pos"),
            ],
        )

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "DimeNet requires node positions (data.pos) to be set."
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            data.edge_index, num_nodes=data.x.size(0)
        )
        dist = (data.pos[i] - data.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = data.pos[idx_i]
        pos_ji, pos_ki = data.pos[idx_j] - pos_i, data.pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        conv_args = {
            "rbf": rbf,
            "sbf": sbf,
            "i": i,
            "j": j,
            "idx_kj": idx_kj,
            "idx_ji": idx_ji,
        }

        return conv_args


"""
PyG Adapted Codes
------------------
The following code is adapted from
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py

"""


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class HydraEmbeddingBlock(EmbeddingBlock):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__(
            num_radial=num_radial, hidden_channels=hidden_channels, act=act
        )
        del self.emb  # Atomic embeddings are handled by Hydra.
        self.reset_parameters()

    def reset_parameters(self):
        # self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        # x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
