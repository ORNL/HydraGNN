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
## PNA aggregation for vec?
## Add rbf in formulation of message?
## Add edge_features
## Get Scaling actually from PNA   from torch_geometric.nn.aggr import DegreeScalerAggregation

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# Torch
import torch
from torch import nn, ModuleList, Tensor
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import torch_scatter
from torch_scatter import scatter

# Torch Geo
from torch_geometric import nn as geom_nn
from geom_nn import activation_resolver
from geom_nn.dense.linear import Linear as geom_Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree
from torch_geometric.typing import Adj, OptTensor

from .Base import Base


class PAINNStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        deg: list,
        edge_dim: int,   # To-Do: Add edge_features
        num_radial: int,
        radius: float,
        *args,
        **kwargs
    ):

        self.x_aggregators = ["mean", "min", "max", "std"]
        self.x_scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.v_aggregators = ["mean"]
        self.v_scalers = ["identity"]
        self.deg = torch.Tensor(deg)
        self.edge_dim = edge_dim
        self.num_radial = num_radial
        self.radius = radius
        self.activation_function = activation_resolver(kwargs.get("activation", "relu"))
        
        self.rbf = rbf_BasisLayer(
            self.num_radial, self.radius
        )
        
        super().__init__(*args, **kwargs)

    def _init_conv(self):
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(self.get_conv(self.input_dim, self.hidden_dim))
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer=False):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        assert (
            hidden_dim > 1
        ), "PainnNet requires more than one hidden dimension between input_dim and output_dim."
        self_inter = PainnMessage(
            node_size=input_dim, 
            x_aggregators=self.x_aggregators,
            x_scalers=self.x_scalers,
            v_aggregators=self.v_aggregators,
            v_scalers=self.v_scalers,
            deg=self.deg,
            edge_dim=self.edge_dim,
            num_radial=self.num_radial, 
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        cross_inter = PainnUpdate(node_size=input_dim, last_layer=last_layer)
        """
        The following linear layers are to get the correct sizing of embeddings. This is 
        necessary to use the hidden_dim, output_dim of HYDRAGNN's stacked conv layers correctly 
        because node_scalar and node-vector are updated through an additive skip connection.
        """
        # Embed down to output size
        node_embed_out = self.activation_function(nn.Linear(input_dim, output_dim))
        vec_embed_out = nn.Linear(input_dim, output_dim) if not last_layer else None
        
        
        input_args = "x, v, pos, edge_index, edge_rbf, edge_vec"
        conv_args = "x, v, edge_index, edge_rbf, edge_vec"
        
        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        if not last_layer:
            return geom_nn.Sequential(
                input_args,
                [
                    # (embed_in, "x, v -> x, v"),
                    (self_inter, conv_args + " -> x, v"),
                    (cross_inter, "x, v -> x, v"),
                    (node_embed_out, "x -> x"),
                    (vec_embed_out, "v -> v"),
                    (lambda x, v, pos: [x, v, pos], "x, v, pos -> x, v, pos"),
                ],
            )
        else:
            return geom_nn.Sequential(
                "x, v, pos, edge_index, diff, dist",
                [
                    # (embed_in, "x, v -> x, v"),
                    (self_inter, conv_args + " -> x, v"),
                    (
                        cross_inter,
                        "x, v -> x",
                    ),  # v is not updated in the last layer to avoid hanging gradients
                    (
                        node_embed_out,
                        "x -> x",
                    ),  # No need to embed down v because it's not used anymore
                    (lambda x, v, pos: [x, v, pos], "x, v, pos -> x, v, pos"),
                ],
            )

    def forward(self, data):
        data, conv_args = self._conv_args(
            data
        )  # Added v to data here (necessary for PAINN Stack)
        x = data.x
        v = data.v
        pos = data.pos

        ### encoder part ####
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                c, v, pos = conv(x=x, v=v, pos=pos, **conv_args)  # Added v here
            else:
                c, v, pos = checkpoint(  # Added v here
                    conv, use_reentrant=False, x=x, v=v, pos=pos, **conv_args
                )
            x = self.activation_function(feat_layer(c))

        #### multi-head decoder part####
        # shared dense layers for graph level output
        if data.batch is None:
            x_graph = x.mean(dim=0, keepdim=True)
        else:
            x_graph = geom_nn.global_mean_pool(x, data.batch.to(x.device))
        outputs = []
        outputs_var = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                x_graph_head = self.graph_shared(x_graph)
                output_head = headloc(x_graph_head)
                outputs.append(output_head[:, :head_dim])
                outputs_var.append(output_head[:, head_dim:] ** 2)
            else:
                if self.node_NN_type == "conv":
                    for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                        c, v, pos = conv(x=x, v=v, pos=pos, **conv_args)
                        c = batch_norm(c)
                        x = self.activation_function(c)
                    x_node = x
                else:
                    x_node = headloc(x=x, batch=data.batch)
                outputs.append(x_node[:, :head_dim])
                outputs_var.append(x_node[:, head_dim:] ** 2)
        if self.var_output:
            return outputs, outputs_var
        return outputs

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "PAINNNet requires node positions (data.pos) to be set."

        # Calculate relative vectors and distances
        i, j = data.edge_index[0], data.edge_index[1]
        diff = data.pos[i] - data.pos[j]
        dist = diff.pow(2).sum(dim=-1).sqrt()
        rbf = self.rbf(dist)
        norm_diff = diff / dist.unsqueeze(-1)

        # Instantiate tensor to hold equivariant traits
        v = torch.zeros(data.x.size(0), 3, data.x.size(1), device=data.x.device)
        data.v = v

        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),
            "edge_rbf": rbf,
            "edge_vec": norm_diff,
        }

        return data, conv_args

class PainnMessage(nn.Module):
    """Message function"""

    def __init__(
        self, 
        node_size: int,
        x_aggregators: List[str],
        x_scalers: List[str],
        v_aggregators: List[str],
        v_scalers: List[str],
        deg: Tensor,
        edge_dim: int,
        num_radial: int,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        # train_norm: bool = False,
        **kwargs,
    ):
    
        super().__init__()
        
        assert node_size % towers == 0

        self.node_size = node_size  # We keep input and output dim the same here because of the skip connection
        self.x_aggregators = x_aggregators
        self.x_scalers = x_scalers
        self.v_aggregators = v_aggregators
        self.v_scalers = v_scalers
        self.deg = deg
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
            modules = [geom_Linear(4 * self.F_in, self.F_in)] if self.edge_dim else [geom_Linear(3 * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [geom_Linear(self.F_in, self.F_in)]
            self.pre_nns.append(nn.Sequential(*modules))

            modules = [geom_Linear((len(x_aggregators) * len(x_scalers) + 1) * self.F_in, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [geom_Linear(self.F_out, self.F_out)]
            self.post_nns.append(nn.Sequential(*modules))
            
        # Filtering rbf and edge_attr for making mij
        self.rbf_emb = nn.Sequential(
            nn.Linear(num_radial, self.F_in),
            activation_resolver(
                act, **(act_kwargs or {})
            ),  # embedded rbf to concat with edge_attr
        )
        if self.edge_dim is not None:
            self.edge_encoder = geom_Linear(edge_dim, self.F_in)
            
        # Projection of rbf for pointwise-product with m_ij
        self.rbf_lin = nn.Linear(
            num_radial, self.F_in*3, bias=False
        )
        
        # MLP for scalar messages to split among x,v operations
        self.scalar_message_mlp = nn.Sequential(
                nn.Linear(self.F_in, self.F_in),
                nn.SiLU(),
                nn.Linear(self.F_in, self.F_in * 3),
            )

        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()
    
    def forward(
        self, 
        x: Tensor, 
        v: Tensor, 
        edge_index: Adj, 
        edge_rbf: Tensor,
        edge_vec: Tensor, 
        edge_attr: OptTensor = None,
    ) -> Tensor :
        
        src, dst = edge_index
        
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
        
        # Create message_scalar using an MLP on concatenated node scalar, neighbor scalar, edge_rbf, and edge_attr(optional)
        # h: Tensor = x[src]  # Dummy.
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
        message_scalar = torch.stack(message_scalar, dim=1)
        scalar_out = self.scalar_message_mlp(message_scalar)  # Expand for PAINN
        
        # Apply distance filtering with pointwise product
        filter_out = scalar_out * self.rbf_lin(edge_rbf)
        
        # Split for x,v tasks
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim=1,
        )
        
        # Create message_vector
        message_vector = x[dst] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_vec).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # Aggregate and scale message_scalar
        message_scalar = aggregate_and_scale(self.x_aggregators, self.x_scalers, message_scalar, edge_index, self.deg)
        message_scalar = torch.cat([x, message_scalar], dim=-1)
        delta_x = [nn(message_scalar[:, i]) for i, nn in enumerate(self.post_nns)]
        
        # Aggregate message_vector
        delta_v = torch.zeros_like(v)
        delta_v.index_add_(0, edge_index[:, 0], message_vector)
        
        # Update node state
        x = x + delta_x
        v = v + delta_v

        return x, v
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.in_channels}, towers={self.towers}, "
            f"edge_dim={self.edge_dim})"
        )

class PainnUpdate(nn.Module):
    """Update function"""

    def __init__(self, node_size: int, last_layer=False):
        super().__init__()

        self.update_U = nn.Linear(node_size, node_size)
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

    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        if not self.last_layer:
            a_vv, a_sv, a_ss = torch.split(
                mlp_output,
                node_vector.shape[-1],
                dim=1,
            )

            delta_v = a_vv.unsqueeze(1) * Uv
            inner_prod = torch.sum(Uv * Vv, dim=1)
            delta_s = a_sv * inner_prod + a_ss

            return node_scalar + delta_s, node_vector + delta_v
        else:
            a_sv, a_ss = torch.split(
                mlp_output,
                node_vector.shape[-1],
                dim=1,
            )

            inner_prod = torch.sum(Uv * Vv, dim=1)
            delta_s = a_sv * inner_prod + a_ss

            return node_scalar + delta_s


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
        n = torch.arange(self.num_radial, device=edge_dist.device) + 1
        return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / self.cutoff) / edge_dist.unsqueeze(-1)

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


def aggregate_and_scale(aggregators, scalers, messages, index, degree):
    """
    Perform the aggregating and scaling operations of PNA on the messages for type 2 tensors.
    """
    aggregated = []
    for aggregator in aggregators:
        if aggregator == 'mean':
            aggregated.append(scatter(messages, index, dim=0, reduce='mean'))
        elif aggregator == 'min':
            aggregated.append(scatter(messages, index, dim=0, reduce='min'))
        elif aggregator == 'max':
            aggregated.append(scatter(messages, index, dim=0, reduce='max'))
        elif aggregator == 'std':
            aggregated.append(scatter(messages, index, dim=0, reduce='std'))

    aggregated = Tensor(torch.cat(aggregated, dim=-1))  # [len(aggregators) * message_size]

    deg = degree(index, num_nodes=aggregated.size(0), dtype=aggregated.dtype)
    size = [1] * len(aggregated.size())
    size[0] = -1
    deg = deg.view(size)

    agg_scaled = []
    for scaler in scalers:
        if scaler == 'identity':
            agg_scaled.append(aggregated)
        elif scaler == 'amplification':
            avg_deg_log = float(torch.log(deg + 1).mean())
            agg_scaled.append(aggregated * (torch.log(deg + 1) / avg_deg_log))
        elif scaler == 'attenuation':
            avg_deg_log = float(torch.log(deg + 1).mean())
            agg_scaled.append(aggregated * (avg_deg_log / torch.log(deg.clamp(min=1) + 1)))
        elif scaler == 'linear':
            avg_deg_lin = float(deg.mean())
            agg_scaled.append(aggregated * (deg / avg_deg_lin))
        elif scaler == 'inverse_linear':
            avg_deg_lin = float(deg.mean())
            agg_scaled.append(aggregated * (avg_deg_lin / deg.clamp(min=1)))  # Clamp prevents division by zero

    agg_scaled = Tensor(torch.cat(agg_scaled, dim=-1))  # [len(scalers) * len(aggregators) * message_size]
    return agg_scaled
