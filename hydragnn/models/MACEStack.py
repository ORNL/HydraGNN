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
# Adapted From:
# GitHub: https://github.com/ACEsuit/mace
# ArXiV: https://arxiv.org/pdf/2206.07697
# Date: August 27, 2024  |  12:37 (EST)
###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

# NOTE MACE Architecture:
## There are two key ideas of MACE:
### (1) Message passing and interaction blocks are equivariant to the O(3) group. And invariant to the T(3) group (translations).
### (2) Predictions are made in an n-body expansion, where n is the numnber of convolutional layers+1. This is done by creating
###     multi-body interactions, then decoding them. Decoding before anything else with 1-body interactions, Interaction Layer 1
###     will decode 2-body interactions, layer 2 will decode 3-body interactions,and so on. So, for a 3-convolutional-layer model
###     predicting energy, there are 4 outputs for energy: 1 before convolution + 3*(1 after each layer). These outputs are summed
###     at the end. This requires some adjustment to the behavior from Base.py

# from typing import Any, Callable, Dict, List, Optional, Type, Union
import warnings

# Torch
import torch
from torch.nn import ModuleList, Sequential
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter

# Torch Geo
from torch_geometric.nn import (
    Sequential as PyGSequential,
)  # This naming is because there is torch.nn.Sequential and torch_geometric.nn.Sequential
from torch_geometric.nn import global_mean_pool

# MACE
from hydragnn.utils.model.mace_utils.modules.blocks import (
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
)
from hydragnn.utils.model.operations import (
    get_edge_vectors_and_lengths,
)

# E3NN
from e3nn import nn, o3
from e3nn.util.jit import compile_mode


# HydraGNN
from .Base import Base

# Etc
import numpy as np
import math


@compile_mode("script")
class MACEStack(Base):
    def __init__(
        self,
        r_max: float,  # The cutoff radius for the radial basis functions and edge_index
        radial_type: str,  # The type of radial basis function to use
        distance_transform: str,  # The distance transform to use
        num_bessel: int,  # The number of radial bessel functions. This dictates the richness of radial information in message-passing.
        edge_dim: int,  # The dimension of HYDRA's optional edge attributes
        max_ell: int,  # Max l-type for CG-tensor product. Theoretically, there is no max l-type, but in practice, we need to truncate the CG-tensor product to keep tractible computation
        node_max_ell: int,  # Max l-type for node features
        avg_num_neighbors: float,
        num_polynomial_cutoff,  # The polynomial cutoff function ensures that the function goes to zero at the cutoff radius smoothly. Same as envelope_exponent for DimeNet
        correlation,  # Used in the product basis block and *roughly* determines the richness of interaction in the n-body interaction of layer 'n'.
        *args,
        **kwargs,
    ):
        """Notes On MACEStack Arguments:"""
        # MACE args that we have given definitions for and the reasons why:
        ## Note: These can be changed in the future if the desired argument options change
        ## interaction_cls / interaction_cls_first: The choice of interaction block type should not make much of a difference and would require more imports in create.py and/or string handling
        ## Atomic Energies: This is not agnostic to what we're predicting, which is a requirement of HYDRA. We also don't have base atomic energies to load, so we simply one-hot encode the atomic numbers and train.
        ## Atomic Numbers / num_elements: It's more robust in preventing errors to just cover the entire periodic table (1-118)

        # MACE args that we have dropped and the resons why:
        ## pair repulsion, distance_transform, compute_virials, etc: HYDRA's framework is meant to compute based on graph or node type, so must be agnostic to these property specific types of computations

        # MACE args constructed by HYDRA args
        ## Reasoning: Oftentimes, MACE arguments show similarity to HYDRA arguments, but are labelled differently
        ## num_interactions is represented by num_conv_layers
        ## radial_MLP uses ceil(hidden_dim/3) for its layer sizes
        ## hidden_irreps and MLP_irreps are constructed from hidden_dim
        ## - Note that this is a nontrivial choice... reconstructing irreps allows users to be unfamiliar with the e3nn library, and is more attached to the HYDRA framework, but limits customization slightly
        ## - I use a hidden_max_ell argument to allow the user to set max ell in the hidden dimensions as well
        """"""

        # Init Args
        ## Passed
        self.node_max_ell = node_max_ell
        num_interactions = kwargs["num_conv_layers"]
        self.edge_dim = edge_dim
        self.avg_num_neighbors = avg_num_neighbors
        ## Defined
        self.interaction_cls = RealAgnosticAttResidualInteractionBlock
        self.interaction_cls_first = RealAgnosticAttResidualInteractionBlock
        atomic_numbers = list(range(1, 119))  # 118 elements in the periodic table
        self.num_elements = len(atomic_numbers)
        # Optional
        num_polynomial_cutoff = (
            5 if num_polynomial_cutoff is None else num_polynomial_cutoff
        )
        self.correlation = [2] if correlation is None else correlation
        radial_type = "bessel" if radial_type is None else radial_type

        # Making Irreps
        self.sh_irreps = o3.Irreps.spherical_harmonics(
            max_ell
        )  # This makes the irreps string
        self.edge_feats_irreps = o3.Irreps(f"{num_bessel}x0e")

        super().__init__(*args, **kwargs)

        self.spherical_harmonics = o3.SphericalHarmonics(
            self.sh_irreps,
            normalize=True,
            normalization="component",  # This makes the spherical harmonic class to be called with forward
        )

        # Register buffers are made when parameters need to be saved and transferred with the model, but not trained.
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            self.correlation = [self.correlation] * self.num_interactions
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=self.node_attr_irreps,
            irreps_out=create_irreps_string(
                self.hidden_dim, 0
            ),  # Changed this to hidden_dim because no longer had node_feats_irreps
        )

    def _init_conv(self):
        # Multihead Decoders
        ## This integrates HYDRA multihead nature with MACE's layer-wise readouts
        ## NOTE Norm techniques (feature_layers in HYDRA) are not advised for use in equivariant models as it can break equivariance
        self.multihead_decoders = ModuleList()
        # attr_irreps for node and edges are created here because we need input_dim, which requires super(base) to be called, which calls _init_conv
        self.node_attr_irreps = o3.Irreps([(self.num_elements, (0, 1))])
        # Edge Attributes are by default the spherical harmoncis but should be extended to include HYDRA's edge_attr is desired
        if self.use_edge_attr:
            self.edge_attrs_irreps = (
                o3.Irreps(f"{self.edge_dim}x0e") + self.sh_irreps
            ).simplify()  # Simplify combines irreps of the same type
        else:
            self.edge_attrs_irreps = self.sh_irreps
        hidden_irreps = o3.Irreps(
            create_irreps_string(self.hidden_dim, self.node_max_ell)
        )
        final_hidden_irreps = o3.Irreps(
            create_irreps_string(self.hidden_dim, 0)
        )  # Only scalars are outputted in the last layer

        last_layer = 1 == self.num_conv_layers

        self.multihead_decoders.append(
            MultiheadDecoderBlock(
                self.node_attr_irreps,
                self.node_max_ell,
                self.config_heads,
                self.head_dims,
                self.head_type,
                self.num_heads,
                self.activation_function,
                self.num_nodes,
                nonlinear=True,
            )
        )  # For base-node traits
        self.graph_convs.append(
            self.get_conv(self.input_dim, self.hidden_dim, first_layer=True)
        )
        irreps = hidden_irreps if not last_layer else final_hidden_irreps
        self.multihead_decoders.append(
            MultiheadDecoderBlock(
                irreps,
                self.node_max_ell,
                self.config_heads,
                self.head_dims,
                self.head_type,
                self.num_heads,
                self.activation_function,
                self.num_nodes,
                nonlinear=last_layer,
            )
        )
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(
                self.hidden_dim, self.hidden_dim, last_layer=last_layer
            )
            self.graph_convs.append(conv)
            irreps = hidden_irreps if not last_layer else final_hidden_irreps
            self.multihead_decoders.append(
                MultiheadDecoderBlock(
                    irreps,
                    self.node_max_ell,
                    self.config_heads,
                    self.head_dims,
                    self.head_type,
                    self.num_heads,
                    self.activation_function,
                    self.num_nodes,
                    nonlinear=last_layer,
                )
            )  # Last layer will be nonlinear node decoding

    def get_conv(self, input_dim, output_dim, first_layer=False, last_layer=False):
        hidden_dim = output_dim if input_dim == 1 else input_dim

        # All of these should be constructed with HYDRA dimensional arguments
        ## Radial
        radial_MLP_dim = math.ceil(
            float(hidden_dim) / 3
        )  # Go based off hidden_dim for radial_MLP
        radial_MLP = [radial_MLP_dim, radial_MLP_dim, radial_MLP_dim]
        ## Input, Hidden, and Output irreps sizing (this is usually just hidden in MACE)
        ### Input dimensions are handled implicitly
        ### Hidden
        hidden_irreps = create_irreps_string(hidden_dim, self.node_max_ell)
        hidden_irreps = o3.Irreps(hidden_irreps)
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        num_features = hidden_irreps.count(
            o3.Irrep(0, 1)
        )  # Multiple copies of spherical harmonics for multiple interactions. They are 'combined' in a certain way during .simplify()  ## This makes it a requirement that hidden irreps all have the same number of channels
        interaction_irreps = (
            (self.sh_irreps * num_features)
            .sort()[0]
            .simplify()  # Kept as sh_irreps for the output of reshape irreps, whether or not edge_attr irreps are added from HYDRA functionality
        )  # .sort() is a tuple, so we need the [0] element for the sorted result
        ### Output
        output_irreps = create_irreps_string(output_dim, self.node_max_ell)
        output_irreps = o3.Irreps(output_irreps)

        # Constructing convolutional layers
        if first_layer:
            hidden_irreps_out = hidden_irreps
            inter = self.interaction_cls_first(
                node_attrs_irreps=self.node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=self.edge_attrs_irreps,
                edge_feats_irreps=self.edge_feats_irreps,
                target_irreps=interaction_irreps,  # Replace with output?
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=self.avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            # Use the appropriate self connection at the first layer for proper E0
            use_sc_first = False
            if "Residual" in str(self.interaction_cls_first):
                use_sc_first = True
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps,
                correlation=self.correlation[0],
                num_elements=self.num_elements,
                use_sc=use_sc_first,
            )
            sizing = o3.Linear(
                hidden_irreps_out, output_irreps
            )  # Change sizing to output_irreps
        elif last_layer:
            # Select only scalars output for last layer
            hidden_irreps_out = str(hidden_irreps[0])
            output_irreps = str(output_irreps[0])
            inter = self.interaction_cls(
                node_attrs_irreps=self.node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=self.edge_attrs_irreps,
                edge_feats_irreps=self.edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=self.avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=self.correlation[0],
                num_elements=self.num_elements,
                use_sc=True,
            )
            sizing = o3.Linear(
                hidden_irreps_out, output_irreps
            )  # Change sizing to output_irreps
        else:
            hidden_irreps_out = hidden_irreps
            inter = self.interaction_cls(
                node_attrs_irreps=self.node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=self.edge_attrs_irreps,
                edge_feats_irreps=self.edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=self.avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=self.correlation[0],  # Should this be i+1?
                num_elements=self.num_elements,
                use_sc=True,
            )
            sizing = o3.Linear(
                hidden_irreps_out, output_irreps
            )  # Change sizing to output_irreps

        input_args = "node_attributes, pos, node_features, edge_attributes, edge_features, edge_index"
        conv_args = "node_attributes, edge_attributes, edge_features, edge_index"  # node_features is not used here because it's passed through in the forward

        if not last_layer:
            return PyGSequential(
                input_args,
                [
                    (inter, "node_features, " + conv_args + " -> node_features, sc"),
                    (prod, "node_features, sc, node_attributes -> node_features"),
                    (sizing, "node_features -> node_features"),
                    (
                        lambda node_features, pos: [node_features, pos],
                        "node_features, pos -> node_features, pos",
                    ),
                ],
            )
        else:
            return PyGSequential(
                input_args,
                [
                    (inter, "node_features, " + conv_args + " -> node_features, sc"),
                    (prod, "node_features, sc, node_attributes -> node_features"),
                    (sizing, "node_features -> node_features"),
                    (
                        lambda node_features, pos: [node_features, pos],
                        "node_features, pos -> node_features, pos",
                    ),
                ],
            )

    def forward(self, data):
        data, conv_args = self._conv_args(data)
        node_features = data.node_features
        node_attributes = data.node_attributes
        pos = data.pos

        ### encoder / decoder part ####
        ## NOTE Norm techniques (feature_layers in HYDRA) are not advised for use in equivariant models as it can break equivariance

        ### There is a readout before the first convolution layer ###
        outputs = []
        output = self.multihead_decoders[0](
            data, node_attributes
        )  # [index][n_output, size_output]
        # Create outputs first
        outputs = output

        ### Do conv --> readout --> repeat for each convolution layer ###
        for conv, readout in zip(self.graph_convs, self.multihead_decoders[1:]):
            if not self.conv_checkpointing:
                node_features, pos = conv(
                    node_features=node_features, pos=pos, **conv_args
                )
                output = readout(data, node_features)  # [index][n_output, size_output]
            else:
                node_features, pos = checkpoint(
                    conv,
                    use_reentrant=False,
                    node_features=node_features,
                    pos=pos,
                    **conv_args,
                )
                output = readout(
                    data, node_features
                )  # output is a list of tensors with [index][n_output, size_output]
            # Sum predictions for each index, taking care of size differences
            for idx, prediction in enumerate(output):
                outputs[idx] = outputs[idx] + prediction

        return outputs

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "MACE requires node positions (data.pos) to be set."

        # Center positions at 0 per graph. This is a requirement for equivariant models that
        # initialize the spherical harmonics, since the initial spherical harmonic projection
        # uses the nodal position vector  x/||x|| as the input to the spherical harmonics.
        # If we didn't center at 0, these models wouldn't even be invariant to translation.
        mean_pos = scatter(data.pos, data.batch, dim=0, reduce="mean")
        data.pos = data.pos - mean_pos[data.batch]

        # Create node_attrs from atomic numbers. Later on it may contain more information
        ## Node attrs are intrinsic properties of the atoms. Currently, MACE only supports atomic number node attributes
        ## data.node_attrs is already used in another place, so has been renamed to data.node_attributes from MACE and same with other data variable names
        data.node_attributes = process_node_attributes(data["x"], self.num_elements)
        data.shifts = torch.zeros(
            (data.edge_index.shape[1], 3), dtype=data.pos.dtype, device=data.pos.device
        )  # Shifts takes into account pbc conditions, but I believe we already generate data.pos to take it into account

        # Embeddings
        node_feats = self.node_embedding(data["node_attributes"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["pos"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attributes = self.spherical_harmonics(vectors)
        if self.use_edge_attr:
            edge_attributes = torch.cat([data.edge_attr, edge_attributes], dim=1)
        edge_features = self.radial_embedding(
            lengths, data["node_attributes"], data["edge_index"], self.atomic_numbers
        )

        # Variable names
        data.node_features = node_feats
        data.edge_attributes = edge_attributes
        data.edge_features = edge_features

        conv_args = {
            "node_attributes": data.node_attributes,
            "edge_attributes": data.edge_attributes,
            "edge_features": data.edge_features,
            "edge_index": data.edge_index,
        }

        return data, conv_args

    def _multihead(self):
        # NOTE Multihead is skipped as it's an integral part of MACE's architecture to have a decoder after every layer,
        # and a convolutional layer in decoding is not supported. Therefore, this final step is not necessary for MACE.
        # However, various parts of multihead are applied in the MultiheadLinearBlock and MultiheadNonLinearBlock classes.
        pass

    def __str__(self):
        return "MACEStack"


def create_irreps_string(
    n: int, ell: int
):  # Custom function to allow for use of HYDRA arguments in creating irreps
    irreps = [f"{n}x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(ell + 1)]
    return " + ".join(irreps)


def process_node_attributes(node_attributes, num_elements):
    # Check that node attributes are atomic numbers and process accordingly
    node_attributes = node_attributes.squeeze()  # Squeeze all unnecessary dimensions
    assert (
        node_attributes.dim() == 1
    ), "MACE only supports raw atomic numbers as node_attributes. Your data.x \
        isn't a 1D tensor after squeezing, are you using vector features?"

    # Check that all elements are integers or integer-like (e.g., 1.0, 2.0), not floats like 1.1
    # This is only a warning so that we don't enforce this requirement on the tests.
    if not torch.all(node_attributes == node_attributes.round()):
        warnings.warn(
            "MACE only supports raw atomic numbers as node_attributes. Your data.x \
            contains floats, which does not align with atomic numbers."
        )

    # Check that all atomic numbers are within the valid range (1 to num_elements)
    # This is only a warning so that we don't enforce this requirement on the tests.
    if not torch.all((node_attributes >= 1) & (node_attributes <= num_elements)):
        warnings.warn(
            "MACE only supports raw atomic numbers as node_attributes. Your data.x \
            is not in the range 1-118, which does not align with atomic numbers."
        )
        node_attributes = torch.clamp(node_attributes, min=1, max=118)

    # Perform one-hot encoding
    one_hot = torch.nn.functional.one_hot(
        (node_attributes - 1).long(),
        num_classes=num_elements,  # Subtract 1 to make atomic numbers 0-indexed for one-hot encoding
    ).float()  # [n_atoms, 118]

    return one_hot


@compile_mode("script")
class MultiheadDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        input_irreps,
        node_max_ell,
        config_heads,
        head_dims,
        head_type,
        num_heads,
        activation_function,
        num_nodes,
        nonlinear=False,
    ):
        super(MultiheadDecoderBlock, self).__init__()
        self.input_irreps = input_irreps
        self.node_max_ell = node_max_ell if not nonlinear else 0
        self.config_heads = config_heads
        self.head_dims = head_dims
        self.head_type = head_type
        self.num_heads = num_heads
        self.activation_function = activation_function
        self.num_nodes = num_nodes

        self.graph_shared = None
        self.node_NN_type = None
        self.heads = ModuleList()

        # Create shared dense layers for graph-level output if applicable
        if "graph" in self.config_heads:
            graph_input_irreps = o3.Irreps(
                f"{self.input_irreps.count(o3.Irrep(0, 1))}x0e"
            )
            dim_sharedlayers = self.config_heads["graph"]["dim_sharedlayers"]
            sharedlayers_irreps = o3.Irreps(f"{dim_sharedlayers}x0e")
            denselayers = []
            denselayers.append(o3.Linear(graph_input_irreps, sharedlayers_irreps))
            denselayers.append(
                nn.Activation(
                    irreps_in=sharedlayers_irreps, acts=[self.activation_function]
                )
            )
            for _ in range(self.config_heads["graph"]["num_sharedlayers"] - 1):
                denselayers.append(o3.Linear(sharedlayers_irreps, sharedlayers_irreps))
                denselayers.append(
                    nn.Activation(
                        irreps_in=sharedlayers_irreps, acts=[self.activation_function]
                    )
                )
            self.graph_shared = Sequential(*denselayers)

        # Create layers for each head
        for ihead in range(self.num_heads):
            if self.head_type[ihead] == "graph":
                num_layers_graph = self.config_heads["graph"]["num_headlayers"]
                hidden_dim_graph = self.config_heads["graph"]["dim_headlayers"]
                denselayers = []
                head_hidden_irreps = o3.Irreps(f"{hidden_dim_graph[0]}x0e")
                denselayers.append(o3.Linear(sharedlayers_irreps, head_hidden_irreps))
                denselayers.append(
                    nn.Activation(
                        irreps_in=head_hidden_irreps, acts=[self.activation_function]
                    )
                )
                for ilayer in range(num_layers_graph - 1):
                    input_irreps = o3.Irreps(f"{hidden_dim_graph[ilayer]}x0e")
                    output_irreps = o3.Irreps(f"{hidden_dim_graph[ilayer + 1]}x0e")
                    denselayers.append(o3.Linear(input_irreps, output_irreps))
                    denselayers.append(
                        nn.Activation(
                            irreps_in=output_irreps, acts=[self.activation_function]
                        )
                    )
                input_irreps = o3.Irreps(f"{hidden_dim_graph[-1]}x0e")
                output_irreps = o3.Irreps(f"{self.head_dims[ihead]}x0e")
                denselayers.append(o3.Linear(input_irreps, output_irreps))
                self.heads.append(Sequential(*denselayers))
            elif self.head_type[ihead] == "node":
                self.node_NN_type = self.config_heads["node"]["type"]
                head = ModuleList()
                if self.node_NN_type == "mlp" or self.node_NN_type == "mlp_per_node":
                    self.num_mlp = 1 if self.node_NN_type == "mlp" else self.num_nodes
                    assert (
                        self.num_nodes is not None
                    ), "num_nodes must be a positive integer for MLP"
                    num_layers_node = self.config_heads["node"]["num_headlayers"]
                    hidden_dim_node = self.config_heads["node"]["dim_headlayers"]
                    head = MLPNode(
                        self.input_irreps,
                        self.node_max_ell,
                        self.config_heads,
                        num_layers_node,
                        hidden_dim_node,
                        self.head_dims[ihead],
                        self.num_mlp,
                        self.num_nodes,
                        self.config_heads["node"]["type"],
                        self.activation_function,
                        nonlinear=nonlinear,
                    )
                    self.heads.append(head)
                else:
                    raise ValueError(
                        f"Unknown head NN structure for node features: {self.node_NN_type}"
                    )
            else:
                raise ValueError(
                    f"Unknown head type: {self.head_type[ihead]}; supported types are 'graph' or 'node'"
                )

    def forward(self, data, node_features):
        if data.batch is None:
            graph_features = node_features[:, : self.hidden_dim].mean(
                dim=0, keepdim=True
            )  # Need to take only the type-0 irreps for aggregation
        else:
            graph_features = global_mean_pool(
                node_features[:, : self.input_irreps.count(o3.Irrep(0, 1))],
                data.batch.to(node_features.device),
            )
        outputs = []
        for headloc, type_head in zip(self.heads, self.head_type):
            if type_head == "graph":
                x_graph_head = self.graph_shared(graph_features)
                outputs.append(headloc(x_graph_head))
            else:  # Node-level output
                if self.node_NN_type == "conv":
                    raise ValueError(
                        "Node-level convolutional layers are not supported in MACE"
                    )
                else:
                    x_node = headloc(node_features, data.batch)
                    outputs.append(x_node)
        return outputs


@compile_mode("script")
class MLPNode(torch.nn.Module):
    def __init__(
        self,
        input_irreps,
        node_max_ell,
        config_heads,
        num_layers,
        hidden_dims,
        output_dim,
        num_mlp,
        num_nodes,
        node_type,
        activation_function,
        nonlinear=False,
    ):
        super().__init__()
        self.input_irreps = input_irreps
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.node_max_ell = node_max_ell if not nonlinear else 0
        self.config_heads = config_heads
        self.num_layers = num_layers
        self.node_type = node_type
        self.num_mlp = num_mlp
        self.num_nodes = num_nodes
        self.activation_function = activation_function

        self.mlp = ModuleList()

        # Create dense layers for each MLP based on node_type ("mlp" or "mlp_per_node")
        for _ in range(self.num_mlp):
            denselayers = []

            # Input and hidden irreps for each MLP layer
            input_irreps = input_irreps
            hidden_irreps = o3.Irreps(f"{hidden_dims[0]}x0e")  # Hidden irreps

            denselayers.append(o3.Linear(input_irreps, hidden_irreps))
            denselayers.append(
                nn.Activation(irreps_in=hidden_irreps, acts=[self.activation_function])
            )

            # Add intermediate layers
            for ilayer in range(self.num_layers - 1):
                input_irreps = o3.Irreps(f"{hidden_dims[ilayer]}x0e")
                hidden_irreps = o3.Irreps(f"{hidden_dims[ilayer + 1]}x0e")
                denselayers.append(o3.Linear(input_irreps, hidden_irreps))
                denselayers.append(
                    nn.Activation(
                        irreps_in=hidden_irreps, acts=[self.activation_function]
                    )
                )

            # Last layer
            hidden_irreps = o3.Irreps(f"{hidden_dims[-1]}x0e")
            output_irreps = o3.Irreps(
                f"{self.output_dim}x0e"
            )  # Assuming head_dims has been passed for the final output
            denselayers.append(o3.Linear(hidden_irreps, output_irreps))

            # Append to MLP
            self.mlp.append(Sequential(*denselayers))

    def node_features_reshape(self, node_features, batch):
        """Reshape node_features from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = node_features.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=node_features.dtype,
            device=node_features.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = node_features[inode_index, :]
        return out

    def forward(self, node_features: torch.Tensor, batch: torch.Tensor):
        if self.node_type == "mlp":
            outs = self.mlp[0](node_features)
        else:
            outs = torch.zeros(
                (
                    node_features.shape[0],
                    self.head_dims[0],
                ),  # Assuming `head_dims` defines the final output dimension
                dtype=node_features.dtype,
                device=node_features.device,
            )
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"
