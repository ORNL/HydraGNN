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

# Adapted From the Following:
# Github: https://github.com/nityasagarjena/PaiNN-model/blob/main/PaiNN/model.py
# Paper: https://arxiv.org/pdf/2102.03150

# To-Do:
## Maybe do PNA aggregation for vectorial? To maintain equivariance, aggregation could only the Identity, but all scalers are valid.

from typing import Any, Callable, Dict, List, Optional, Union
import pdb

# Torch
import torch
from torch import nn, Tensor
from torch.nn import ModuleList
from torch.utils.checkpoint import checkpoint

# Torch Geo
from torch_geometric import nn as geom_nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.dense.linear import Linear as geom_Linear
from torch_geometric.nn.aggr.scaler import DegreeScalerAggregation
from torch_geometric.typing import Adj, OptTensor

# HydraGNN
from .Base import Base
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths


class PNAEqStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        input_args,
        conv_args,
        deg: list,
        edge_dim: int,
        num_radial: int,
        radius: float,
        *args,
        **kwargs,
    ):

        self.x_aggregators = ["mean", "min", "max", "std"]
        self.x_scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
            "inverse_linear",
        ]
        self.deg = self._sanitize_degree(torch.Tensor(deg))
        self.edge_dim = edge_dim
        self.num_radial = num_radial
        self.radius = radius
        self.is_edge_model = True  # specify that mpnn can handle edge features
        super().__init__(input_args, conv_args, *args, **kwargs)

        self.rbf = rbf_BasisLayer(self.num_radial, self.radius)

    @staticmethod
    def _sanitize_degree(deg: torch.Tensor) -> torch.Tensor:
        if deg.numel() == 0:
            return deg.new_ones((1,), dtype=torch.float32)

        deg = deg.to(dtype=torch.float32)

        # Compute max over finite values (this is used to replace +inf)
        finite = torch.isfinite(deg)
        max_finite = deg[finite].max() if finite.any() else deg.new_tensor(1.0)

        # Replace NaN/-inf with 1, +inf with max_finite
        deg = torch.nan_to_num(deg, nan=1.0, neginf=1.0, posinf=max_finite.item())

        return deg.clamp_min(1.0)

    def _init_conv(self):
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(
            self._apply_global_attn(
                self.get_conv(
                    self.embed_dim,
                    self.hidden_dim,
                    last_layer,
                    edge_dim=self.edge_embed_dim,
                )
            )
        )
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.hidden_dim,
                        self.hidden_dim,
                        last_layer,
                        edge_dim=self.edge_embed_dim,
                    )
                )
            )
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer=False, edge_dim=None):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        assert (
            hidden_dim > 1
        ), "PNAEq requires more than one hidden dimension between input_dim and output_dim."
        message = PainnMessage(
            node_size=input_dim,
            x_aggregators=self.x_aggregators,
            x_scalers=self.x_scalers,
            deg=self.deg,
            edge_dim=edge_dim,
            num_radial=self.num_radial,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        update = PainnUpdate(node_size=input_dim, last_layer=last_layer)
        """
        The following linear layers are to get the correct sizing of embeddings. This is
        necessary to use the hidden_dim, output_dim of HYDRAGNN's stacked conv layers correctly
        because node_scalar and node-vector are updated through an additive skip connection.
        """
        # Embed down to output size
        node_embed_out = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),  # Promotes stability to avoid exploding gradients
            nn.Linear(output_dim, output_dim),
        )
        vec_embed_out = (
            geom_nn.Linear(input_dim, output_dim) if not last_layer else None
        )

        if not last_layer:
            return geom_nn.Sequential(
                self.input_args,
                [
                    (message, self.conv_args + " -> inv_node_feat, equiv_node_feat"),
                    (
                        update,
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                    (node_embed_out, "inv_node_feat -> inv_node_feat"),
                    (vec_embed_out, "equiv_node_feat -> equiv_node_feat"),
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
            return geom_nn.Sequential(
                self.input_args,
                [
                    (message, self.conv_args + " -> inv_node_feat, equiv_node_feat"),
                    (
                        update,
                        "inv_node_feat, equiv_node_feat -> inv_node_feat",
                    ),  # v is not updated in the last layer to avoid hanging gradients
                    (
                        node_embed_out,
                        "inv_node_feat -> inv_node_feat",
                    ),  # No need to embed down v because it's not used anymore
                    (
                        lambda inv_node_feat, equiv_node_feat: [
                            inv_node_feat,
                            equiv_node_feat,
                        ],
                        "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                    ),
                ],
            )

    def _embedding(self, data):
        super()._embedding(data)

        assert (
            data.pos is not None
        ), "PNAEq requires node positions (data.pos) to be set."

        # Edge vector and distance features
        norm_edge_vec, edge_dist = get_edge_vectors_and_lengths(
            data.pos, data.edge_index, data.edge_shifts, normalize=True
        )
        rbf = self.rbf(edge_dist.squeeze())

        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),
            "edge_rbf": rbf,
            "edge_vec": norm_edge_vec,
        }

        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_attr})

        if self.use_global_attn:
            # encode node positional embeddings
            x = self.pos_emb(data.pe)
            # if node features are available, genrate mebeddings, concatenate with positional embeddings and map to hidden dim
            if self.input_dim:
                x = torch.cat((self.node_emb(data.x.float()), x), 1)
                x = self.node_lin(x)
            # repeat for edge features and relative edge encodings
            if self.is_edge_model:
                e = self.rel_pos_emb(data.rel_pe)
                if self.use_edge_attr:
                    e = torch.cat((self.edge_emb(conv_args["edge_attr"]), e), 1)
                    e = self.edge_lin(e)
                conv_args.update({"edge_attr": e})
        else:
            x = data.x
        # Instantiate tensor to hold equivariant traits
        v = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        return x, v, conv_args


class PainnMessage(MessagePassing):
    """Message function"""

    def __init__(
        self,
        node_size: int,
        x_aggregators: List[str],
        x_scalers: List[str],
        deg: Tensor,
        edge_dim: int,
        num_radial: int,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "tanh",
        act_kwargs: Optional[Dict[str, Any]] = None,
        # train_norm: bool = False,
        **kwargs,
    ):

        degree_scaler_aggregation = DegreeScalerAggregation(
            aggr=x_aggregators, scaler=x_scalers, deg=deg
        )

        super().__init__(aggr=degree_scaler_aggregation, node_dim=0, **kwargs)

        assert node_size % towers == 0

        self.node_size = node_size  # We keep input and output dim the same here because of the skip connection
        self.num_radial = num_radial
        self.edge_dim = edge_dim

        self.towers = towers
        self.divide_input = divide_input

        self.F_in = node_size // towers if divide_input else node_size
        self.F_out = self.node_size // towers

        # Pre and post MLPs
        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = (
                [geom_Linear(4 * self.F_in, self.F_in)]
                if self.edge_dim
                else [geom_Linear(3 * self.F_in, self.F_in)]
            )
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [geom_Linear(self.F_in, self.F_in)]
            self.pre_nns.append(nn.Sequential(*modules))

            modules = [
                geom_Linear(
                    (len(x_aggregators) * len(x_scalers) + 1) * self.F_in, self.F_out
                )
            ]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [geom_Linear(self.F_out, self.F_out)]
            self.post_nns.append(nn.Sequential(*modules))

        # Embedding rbf for making m_ij
        self.rbf_emb = nn.Sequential(
            nn.Linear(num_radial, self.F_in),
            activation_resolver(
                act, **(act_kwargs or {})
            ),  # embedded rbf to concat with edge_attr
        )
        # Embedding edge_attr for making m_ij
        if self.edge_dim is not None:
            self.edge_encoder = geom_Linear(edge_dim, self.F_in)

        # Projection of rbf for pointwise-product with m_ij
        self.rbf_lin = nn.Linear(num_radial, self.F_in * 3, bias=False)

        # MLP for scalar messages to split among x,v operations
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(self.F_in, self.F_in),
            nn.Tanh(),  # Promotes stability to avoid exploding gradients
            nn.Linear(self.F_in, self.F_in),
            nn.SiLU(),
            nn.Linear(self.F_in, self.F_in * 3),
        )

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        edge_index: Adj,
        edge_rbf: Tensor,
        edge_vec: Tensor,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        src, dst = edge_index.t()

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # Create message_scalar using an MLP on concatenated node scalar, neighbor scalar, edge_rbf, and edge_attr(optional)
        if edge_attr is not None:
            rbf_attr = self.rbf_emb(edge_rbf)
            rbf_attr = rbf_attr.view(-1, 1, self.F_in)
            rbf_attr = rbf_attr.repeat(1, self.towers, 1)
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            message_scalar = torch.cat([x[src], x[dst], rbf_attr, edge_attr], dim=-1)
        else:
            rbf_attr = self.rbf_emb(edge_rbf)
            rbf_attr = rbf_attr.view(-1, 1, self.F_in)
            rbf_attr = rbf_attr.repeat(1, self.towers, 1)
            message_scalar = torch.cat([x[src], x[dst], rbf_attr], dim=-1)

        # Pass the concatenated features through pre_nns
        message_scalar = [nn(message_scalar[:, i]) for i, nn in enumerate(self.pre_nns)]
        # message_scalar = torch.stack(message_scalar, dim=1).squeeze(1)
        message_scalar = torch.stack(message_scalar, dim=1)
        scalar_out = self.scalar_message_mlp(message_scalar)  # Expand for PAINN

        # Apply distance filtering with pointwise product
        # Put rbf through a linear layer
        rbf = self.rbf_lin(edge_rbf)
        # Repeat distance embedding for each tower
        rbf = rbf.view(-1, 1, 3 * self.F_in)
        rbf = rbf.repeat(1, self.towers, 1)
        # Perform Hadamard (element-wise) product
        filter_out = scalar_out * rbf

        # Split for x,v tasks
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim=-1,
        )

        # Create message_vector
        message_vector = v[dst] * gate_state_vector
        edge_vector = gate_edge_vector * edge_vec.unsqueeze(-1)
        message_vector = message_vector + edge_vector

        # Aggregate and scale message_scalar
        message_scalar = self.aggr_module(
            message_scalar.squeeze(1), index=src, dim_size=x.shape[0]
        ).unsqueeze(
            1
        )  # degree scalar aggregation expects shape(num_nodes, feature_dim)
        message_scalar = torch.cat([x, message_scalar], dim=-1)
        delta_x = [nn(message_scalar[:, i]) for i, nn in enumerate(self.post_nns)]
        delta_x = torch.stack(delta_x, dim=1)

        # Aggregate message_vector
        delta_v = torch.zeros_like(v)
        delta_v.index_add_(0, src, message_vector)

        # Update with skip connection
        x = x.squeeze(1) + delta_x.squeeze(1)
        v = v + delta_v

        return x, v

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.in_channels}, towers={self.towers}, "
            f"edge_dim={self.edge_dim})"
        )


class PainnUpdate(MessagePassing):
    """Update function"""

    def __init__(self, node_size: int, last_layer=False):
        super().__init__()

        self.update_X = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        self.last_layer = last_layer

        if not self.last_layer:
            self.update_mlp = nn.Sequential(
                nn.Linear(node_size * 2, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 3),
            )
        else:
            self.update_mlp = nn.Sequential(
                nn.Linear(node_size * 2, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 2),
            )

    def forward(self, x, v):
        Xv = self.update_X(v)
        Vv = self.update_V(v)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, x), dim=-1)
        mlp_output = self.update_mlp(mlp_input)

        if not self.last_layer:
            a_vv, a_xv, a_xx = torch.split(
                mlp_output,
                x.shape[-1],
                dim=-1,
            )

            delta_v = a_vv.unsqueeze(1) * Xv
            inner_prod = torch.sum(Xv * Vv, dim=1)
            delta_x = a_xv * inner_prod + a_xx

            return x + delta_x, v + delta_v
        else:
            a_xv, a_xx = torch.split(
                mlp_output,
                v.shape[-1],
                dim=-1,
            )

            inner_prod = torch.sum(Xv * Vv, dim=1)
            delta_x = a_xv * inner_prod + a_xx

            return x + delta_x


class rbf_BasisLayer(nn.Module):
    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff

    def sinc_expansion(self, edge_dist: torch.Tensor) -> torch.Tensor:
        """
        Calculate sinc radial basis function:

        sin(n * pi * d / d_cut) / d
        """
        n = (
            torch.arange(
                self.num_radial, device=edge_dist.device, dtype=edge_dist.dtype
            )
            + 1
        )
        scaled = edge_dist.unsqueeze(-1) * n * torch.pi / self.cutoff

        # Avoid division-by-zero when edges have zero length (e.g., self-loops or overlapping nodes).
        eps = 1e-9
        safe_edge_dist = edge_dist.unsqueeze(-1).clamp_min(eps)
        sinc = torch.sin(scaled) / safe_edge_dist

        # For very small distances, use the analytic limit sin(x)/x -> 1 so the ratio goes to (n*pi/d_cut).
        small_mask = edge_dist.unsqueeze(-1).abs() < eps
        if small_mask.any():
            sinc = torch.where(
                small_mask,
                (n * torch.pi / self.cutoff),
                sinc,
            )

        return sinc

    def cosine_cutoff(self, edge_dist: torch.Tensor) -> torch.Tensor:
        """
        Calculate cutoff value based on distance.
        This uses the cosine Behler-Parinello cutoff function:

        f(d) = 0.5 * (cos(pi * d / d_cut) + 1) for d < d_cut and 0 otherwise
        """
        return torch.where(
            edge_dist < self.cutoff,
            0.5 * (torch.cos(torch.pi * edge_dist / self.cutoff) + 1),
            torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
        )

    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        # Calculate sinc expansion
        sinc_out = self.sinc_expansion(edge_dist)

        # Calculate cosine cutoff
        cosine_out = self.cosine_cutoff(edge_dist).unsqueeze(-1)

        # Apply filter weights
        filter_weight = sinc_out * cosine_out

        return filter_weight
