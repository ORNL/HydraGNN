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

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# Torch
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential
from torch_geometric.nn import (
    Sequential as PyGSequential,
)  # This naming is because there is torch.nn.Sequential and torch_geometric.nn.Sequential

# Torch Geo
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_geometric.typing import Adj

# HydraGNN
from .Base import Base


class PNAPlusStack(Base):
    def __init__(
        self,
        deg: list,
        edge_dim: int,
        envelope_exponent: int,
        num_radial: int,
        radius: float,
        *args,
        **kwargs,
    ):

        self.aggregators = ["mean", "min", "max", "std"]
        self.scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.deg = torch.Tensor(deg)
        self.edge_dim = edge_dim
        self.envelope_exponent = envelope_exponent
        self.num_radial = num_radial
        self.radius = radius

        super().__init__(*args, **kwargs)

        self.rbf = BesselBasisLayer(
            self.num_radial, self.radius, self.envelope_exponent
        )

    def get_conv(self, input_dim, output_dim):
        pna = PNAConv(
            in_channels=input_dim,
            out_channels=output_dim,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            edge_dim=self.edge_dim,
            num_radial=self.num_radial,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

        input_args = "x, pos, edge_index, rbf"
        conv_args = "x, edge_index, rbf"

        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        return PyGSequential(
            input_args,
            [
                (pna, conv_args + " -> x"),
                (lambda x, pos: [x, pos], "x, pos -> x, pos"),
            ],
        )

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "PNA+ requires node positions (data.pos) to be set."

        j, i = data.edge_index  # j->i
        dist = (data.pos[i] - data.pos[j]).pow(2).sum(dim=-1).sqrt()
        rbf = self.rbf(dist)
        # rbf = dist.unsqueeze(-1)
        conv_args = {"edge_index": data.edge_index.to(torch.long), "rbf": rbf}

        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_attr})

        return conv_args

    def __str__(self):
        return "PNAStack"


class PNAConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        num_radial: int = 5,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        train_norm: bool = False,
        **kwargs,
    ):

        aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear(3 * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)
        self.rbf_lin = Linear(
            num_radial, self.F_in, bias=False
        )  # projection of rbf for Hadamard with m_ij
        self.rbf_emb = Sequential(
            Linear(num_radial, self.F_in),
            activation_resolver(
                act, **(act_kwargs or {})
            ),  # embedded rbf to concat with edge_attr
        )

        if self.edge_dim is not None:
            self.edge_encoder = Linear(self.F_in + edge_dim, self.F_in)

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

    def propagate(self, edge_index: Adj, size: Tuple[int, int] = None, **kwargs):
        kwargs["rbf"] = kwargs.get(
            "rbf", None
        )  # Necessary to include rbf in message-passsing now
        return super().propagate(edge_index, size=size, **kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        rbf: Tensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None, rbf=rbf)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(
        self, x_i: Tensor, x_j: Tensor, rbf: Tensor = None, edge_attr: OptTensor = None
    ) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, self.rbf_emb(rbf)], dim=-1)
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            rbf_attr = self.rbf_emb(rbf)
            rbf_attr = rbf_attr.view(-1, 1, self.F_in)
            rbf_attr = rbf_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, rbf_attr], dim=-1)

        # Pass the concatenated embeddings through the pre_nns
        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        hs = torch.stack(hs, dim=1)

        # Put rbf through a linear layer
        rbf = self.rbf_lin(rbf)
        # Repeat distance embedding for each tower
        rbf = rbf.view(-1, 1, self.F_in)
        rbf = rbf.repeat(1, self.towers, 1)
        # Perform Hadamard product
        hs = hs * rbf

        return hs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, towers={self.towers}, "
            f"edge_dim={self.edge_dim})"
        )

    @staticmethod
    def get_degree_histogram(loader: DataLoader) -> Tensor:
        r"""Returns the degree histogram to be used as input for the :obj:`deg`
        argument in :class:`PNAConv`."""
        deg_histogram = torch.zeros(1, dtype=torch.long)
        for data in loader:
            deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg_bincount = torch.bincount(deg, minlength=deg_histogram.numel())
            deg_histogram = deg_histogram.to(deg_bincount.device)
            if deg_bincount.numel() > deg_histogram.numel():
                deg_bincount[: deg_histogram.size(0)] += deg_histogram
                deg_histogram = deg_bincount
            else:
                assert deg_bincount.numel() == deg_histogram.numel()
                deg_histogram += deg_bincount

        return deg_histogram
