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

from typing import Callable, Optional, Tuple
from torch_geometric.typing import SparseTensor

import pdb

import torch
from torch import Tensor
from torch.nn import Identity, SiLU

from torch_geometric.nn import Linear, Sequential
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    InteractionPPBlock,
    OutputPPBlock,
    SphericalBasisLayer,
)
from torch_geometric.utils import scatter

from .Base import Base
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths


class DIMEStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        input_args,
        conv_args,
        basis_emb_size,
        envelope_exponent,
        int_emb_size,
        out_emb_size,
        num_after_skip,
        num_before_skip,
        num_radial,
        num_spherical,
        edge_dim,
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
        self.edge_dim = edge_dim
        self.radius = radius
        self.is_edge_model = True  # specify that mpnn can handle edge features
        super().__init__(input_args, conv_args, *args, **kwargs)

        self.rbf = BesselBasisLayer(num_radial, radius, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, radius, envelope_exponent
        )

        pass

    def _init_conv(self):
        self.graph_convs.append(
            self._apply_global_attn(
                self.get_conv(
                    self.embed_dim, self.hidden_dim, edge_dim=self.edge_embed_dim
                )
            )
        )
        self.feature_layers.append(Identity())
        for _ in range(self.num_conv_layers - 1):
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.hidden_dim, self.hidden_dim, edge_dim=self.edge_embed_dim
                    )
                )
            )
            self.feature_layers.append(Identity())

    def get_conv(self, input_dim, output_dim, edge_dim=None):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        assert (
            hidden_dim > 1
        ), "DimeNet requires more than one hidden dimension between input_dim and output_dim."
        lin = Linear(input_dim, hidden_dim)
        emb = HydraEmbeddingBlock(
            num_radial=self.num_radial,
            hidden_channels=hidden_dim,
            act=SiLU(),
            edge_dim=edge_dim,
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

        if self.use_edge_attr or (
            self.use_global_attn and self.is_edge_model
        ):  # check if gps is being used and mpnn can handle edge feats
            return Sequential(
                self.input_args,
                [
                    (lin, "inv_node_feat -> inv_node_feat"),
                    (emb, "inv_node_feat, rbf, i, j, edge_attr -> x1"),
                    (inter, "x1, rbf, sbf, idx_kj, idx_ji -> x2"),
                    (dec, "x2, rbf, i -> inv_node_feat"),
                    (
                        lambda inv_node_feat, equiv_node_feat: [
                            inv_node_feat,
                            equiv_node_feat,
                        ],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )
        else:
            return Sequential(
                self.input_args,
                [
                    (lin, "inv_node_feat -> inv_node_feat"),
                    (emb, "inv_node_feat, rbf, i, j -> x1"),
                    (inter, "x1, rbf, sbf, idx_kj, idx_ji -> x2"),
                    (dec, "x2, rbf, i -> inv_node_feat"),
                    (
                        lambda x, pos: [x, pos],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )

    def _embedding(self, data):
        super()._embedding(data)

        assert (
            data.pos is not None
        ), "DimeNet requires node positions (data.pos) to be set."

        # Calculate triplet indices
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            data.edge_index, num_nodes=data.x.size(0)
        )

        # Calculate edge_vec and edge_dist
        edge_vec, edge_dist = get_edge_vectors_and_lengths(
            data.pos, data.edge_index, data.edge_shifts
        )

        # Calculate angles
        pos_ji = edge_vec[idx_ji]
        pos_kj = edge_vec[idx_kj]
        pos_ki = (
            pos_kj + pos_ji
        )  # It's important to calculate the vectors separately and then add in case of periodic boundary conditions
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.linalg.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(edge_dist.squeeze())
        sbf = self.sbf(edge_dist.squeeze(), angle, idx_kj)

        conv_args = {
            "rbf": rbf,
            "sbf": sbf,
            "i": i,
            "j": j,
            "idx_kj": idx_kj,
            "idx_ji": idx_ji,
        }

        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_attr})

        if self.use_global_attn:
            x = self.pos_emb(data.pe)
            if self.input_dim:
                x = torch.cat((self.node_emb(data.x.float()), x), 1)
                x = self.node_lin(x)
            if self.is_edge_model:
                e = self.rel_pos_emb(data.rel_pe)
                if self.use_edge_attr:
                    e = torch.cat((self.edge_emb(conv_args["edge_attr"]), e), 1)
                    e = self.edge_lin(e)
                conv_args.update({"edge_attr": e})
            return x, data.pos, conv_args
        else:
            return data.x, data.pos, conv_args


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
    src, dst = edge_index  # j -> i

    # Workaround for torch_sparse bug on AMD GPUs (OLCF Frontier with ROCm):
    # avoid SparseTensor indexing while keeping memory linear in number of triplets.
    num_edges = src.size(0)
    if num_edges == 0:
        empty = src.new_empty(0)
        return dst, src, empty, empty, empty, empty, empty

    # Group edges by destination node to enumerate incoming edges k -> j.
    counts_in = torch.bincount(dst, minlength=num_nodes)
    perm_in = torch.argsort(dst)
    ptr_in = torch.zeros(num_nodes + 1, device=src.device, dtype=torch.long)
    ptr_in[1:] = torch.cumsum(counts_in, dim=0)

    # For each edge j -> i (index ji), pair it with all edges k -> j.
    deg_per_ji = counts_in[src]
    idx_ji_all = torch.repeat_interleave(
        torch.arange(num_edges, device=src.device, dtype=torch.long), deg_per_ji
    )

    if idx_ji_all.numel() == 0:
        empty = src.new_empty(0)
        return dst, src, empty, empty, empty, empty, empty

    seg_offsets = torch.cumsum(deg_per_ji, dim=0) - deg_per_ji
    local_pos = torch.arange(idx_ji_all.numel(), device=src.device, dtype=torch.long)
    local_pos = local_pos - torch.repeat_interleave(seg_offsets, deg_per_ji)

    j_for_all = src[idx_ji_all]
    start_for_all = ptr_in[j_for_all]
    idx_kj_all = perm_in[start_for_all + local_pos]

    # Exclude triplets with k == i.
    valid = src[idx_kj_all] != dst[idx_ji_all]
    idx_ji = idx_ji_all[valid]
    idx_kj = idx_kj_all[valid]

    idx_i = dst[idx_ji]
    idx_j = src[idx_ji]
    idx_k = src[idx_kj]

    return dst, src, idx_i, idx_j, idx_k, idx_kj, idx_ji


class HydraEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        act: Callable,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.act = act

        # self.emb = Embedding(95, hidden_channels)  # Atomic embeddings are handled by HYDRA
        self.lin_rbf = Linear(num_radial, hidden_channels)
        if edge_dim is not None:  # Optional edge features
            self.edge_lin = Linear(edge_dim, hidden_channels)
            self.lin = Linear(4 * hidden_channels, hidden_channels)
        else:
            self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()
        if hasattr(self, "edge_lin"):
            self.edge_lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        i: Tensor,
        j: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        # x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))

        # Include edge features if they are provided
        if edge_attr is not None and hasattr(self, "edge_lin"):
            edge_attr = self.act(self.edge_lin(edge_attr))
            out = torch.cat([x[i], x[j], rbf, edge_attr], dim=-1)
        else:
            out = torch.cat([x[i], x[j], rbf], dim=-1)

        return self.act(self.lin(out))
