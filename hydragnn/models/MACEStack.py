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
from torch.nn import ModuleList, Sequential, Linear
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

# E3NN
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

# HydraGNN
from .Base import Base
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths
from hydragnn.utils.model.irreps_tools import create_irreps_string
from hydragnn.utils.model.mace_utils.modules.blocks import (
    CombineBlock,
    SplitBlock,
    NonLinearMultiheadDecoderBlock,
    LinearMultiheadDecoderBlock,
)

# Etc
import numpy as np
import math


@compile_mode("script")
class MACEStack(Base):
    def __init__(
        self,
        input_args,
        conv_args,
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

        ############################ Prior to Inheritance ############################
        # Init Args
        ## Passed
        self.max_ell = max_ell
        self.node_max_ell = node_max_ell
        num_interactions = kwargs["num_conv_layers"]
        self.edge_dim = edge_dim
        self.avg_num_neighbors = avg_num_neighbors
        ## Defined
        self.interaction_cls = RealAgnosticAttResidualInteractionBlock
        self.interaction_cls_first = RealAgnosticAttResidualInteractionBlock
        atomic_numbers = list(
            range(1, 119)
        )  # 118 elements in the periodic table. Simpler to not expose this to the user
        self.num_elements = len(atomic_numbers)
        ## Optional
        num_polynomial_cutoff = (
            5 if num_polynomial_cutoff is None else num_polynomial_cutoff
        )
        self.correlation = [2] if correlation is None else correlation
        radial_type = "bessel" if radial_type is None else radial_type

        # Making Irreps
        self.edge_feats_irreps = o3.Irreps(f"{num_bessel}x0e")
        self.node_attr_irreps = o3.Irreps([(self.num_elements, (0, 1))])
        ##############################################################################

        # NOTE the super() call is done at this point because some of the arguments are needed for the initialization of the
        # Base class. For example, _init_ calls _init_conv, which requires self.edge_attr_irreps, self.node_attr_irreps, etc.
        # Other arguments such as the radial type may be moved before the super() call just for streamlining the code.
        super().__init__(input_args, conv_args, *args, **kwargs)

        ############################ Post Inheritance ############################
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.sh_irreps,
            normalize=True,
            normalization="component",
        )  # Called to embed the edge_vectors into spherical harmonics

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
            ),  # Going from one-hot to hidden_dim
        )
        ##############################################################################

    def _init_conv(self):
        # Multihead Decoders
        ## This integrates HYDRA multihead nature with MACE's layer-wise readouts
        ## NOTE Norm techniques (called feature_layers in HYDRA) such as BatchNorm are
        ##      not advised for use in equivariant models as it can break equivariance
        self.multihead_decoders = ModuleList()

        # Making Irreps
        ## Edge attributes (must be done here in order to have self.use_edge_attr)
        self.sh_irreps = o3.Irreps.spherical_harmonics(
            self.max_ell
        )  # This makes the irreps string
        if self.use_edge_attr:
            self.edge_attrs_irreps = (
                o3.Irreps(f"{self.edge_dim}x0e") + self.sh_irreps
            ).simplify()  # Simplify combines irreps of the same type (e.g., 2x0e + 2x0e = 4x0e)
        else:
            self.edge_attrs_irreps = self.sh_irreps
        ## Node features after convolution
        hidden_irreps = o3.Irreps(
            create_irreps_string(self.hidden_dim, self.node_max_ell)
        )
        final_hidden_irreps = o3.Irreps(
            create_irreps_string(self.hidden_dim, 0)
        )  # Only scalars are output in the last layer

        last_layer = 1 == self.num_conv_layers

        # Decoder before convolutions based on node_attributes
        self.multihead_decoders.append(
            get_multihead_decoder(
                nonlinear=last_layer,
                input_irreps=self.node_attr_irreps,
                config_heads=self.config_heads,
                head_dims=self.head_dims,
                head_type=self.head_type,
                num_heads=self.num_heads,
                activation_function=self.activation_function,
                num_nodes=self.num_nodes,
            )
        )

        # First Conv and Decoder
        self.graph_convs.append(
            self.get_conv(
                self.hidden_dim,
                self.hidden_dim,
                first_layer=True,
                last_layer=last_layer,
            )  # Node features are already converted to hidden_dim via one-hot embedding
        )
        irreps = hidden_irreps if not last_layer else final_hidden_irreps
        self.multihead_decoders.append(
            get_multihead_decoder(
                nonlinear=last_layer,
                input_irreps=irreps,
                config_heads=self.config_heads,
                head_dims=self.head_dims,
                head_type=self.head_type,
                num_heads=self.num_heads,
                activation_function=self.activation_function,
                num_nodes=self.num_nodes,
            )
        )

        # Variable number of convolutions and decoders
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            self.graph_convs.append(
                self.get_conv(self.hidden_dim, self.hidden_dim, last_layer=last_layer)
            )
            irreps = hidden_irreps if not last_layer else final_hidden_irreps
            self.multihead_decoders.append(
                get_multihead_decoder(
                    nonlinear=last_layer,
                    input_irreps=irreps,
                    config_heads=self.config_heads,
                    head_dims=self.head_dims,
                    head_type=self.head_type,
                    num_heads=self.num_heads,
                    activation_function=self.activation_function,
                    num_nodes=self.num_nodes,
                )
            )  # Last layer will be nonlinear node decoding

    def get_conv(self, input_dim, output_dim, first_layer=False, last_layer=False):
        hidden_dim = output_dim if input_dim == 1 else input_dim

        # NOTE All of these should be constructed with HYDRA dimensional arguments

        # Radial
        radial_MLP_dim = math.ceil(
            float(hidden_dim) / 3
        )  # Go based off hidden_dim for radial_MLP
        radial_MLP = [radial_MLP_dim, radial_MLP_dim, radial_MLP_dim]

        # Input, Hidden, and Output irreps sizing (this is usually just hidden in MACE)
        ## Input
        if first_layer:
            node_feats_irreps = o3.Irreps(create_irreps_string(input_dim, 0))
        else:
            node_feats_irreps = o3.Irreps(
                create_irreps_string(input_dim, self.node_max_ell)
            )
        ## Hidden
        hidden_irreps = o3.Irreps(create_irreps_string(hidden_dim, self.node_max_ell))
        num_features = hidden_dim  # Multiple copies of spherical harmonics for multiple interactions. They are 'combined' during .simplify()  ## This makes it a requirement that different irrep types in hidden irreps all have the same number of channels.
        interaction_irreps = (
            (self.sh_irreps * num_features)
            .sort()[0]
            .simplify()  # Kept as sh_irreps for the output of reshape irreps, whether or not edge_attr irreps are added from HYDRA functionality
        )  # .sort() is a tuple, so we need the [0] element for the sorted result
        ## Output
        output_irreps = o3.Irreps(create_irreps_string(output_dim, self.node_max_ell))

        # Combine the inv_node_feat and equiv_node_feat into irreps
        combine = CombineBlock()

        # Scalars output for last layer
        if last_layer:
            # Convert to irreps here for countability in the splitblock
            hidden_irreps = o3.Irreps(str(hidden_irreps[0]))
            output_irreps = o3.Irreps(str(output_irreps[0]))

        # Interaction
        if first_layer:
            interaction_cls = self.interaction_cls_first
        else:
            interaction_cls = self.interaction_cls
        inter = interaction_cls(
            node_attrs_irreps=self.node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=self.edge_attrs_irreps,
            edge_feats_irreps=self.edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_MLP=radial_MLP,
        )

        # Product
        if first_layer:
            use_sc = "Residual" in str(self.interaction_cls_first)
        else:
            use_sc = True  # True for non-first layers
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=interaction_irreps,
            target_irreps=hidden_irreps,
            correlation=self.correlation[0],  # Currently uniform across all layers
            num_elements=self.num_elements,
            use_sc=use_sc,
        )

        # Post-processing for conv layer output size
        sizing = o3.Linear(hidden_irreps, output_irreps)

        # Split irreps into inv_node_feat and equiv_node_feat
        split = SplitBlock(output_irreps)

        return PyGSequential(
            self.input_args,
            [
                (combine, "inv_node_feat, equiv_node_feat -> node_features"),
                (
                    inter,
                    "node_features, " + self.conv_args + " -> node_features, sc",
                ),
                (prod, "node_features, sc, node_attributes -> node_features"),
                (sizing, "node_features -> node_features"),
                (
                    lambda node_features, equiv_node_feat: [
                        node_features,
                        equiv_node_feat,
                    ],
                    "node_features, equiv_node_feat -> node_features, equiv_node_feat",
                ),
                (split, "node_features -> inv_node_feat, equiv_node_feat"),
            ],
        )  # NOTE An if/else is not needed here because MACE's interaction layers already contract down to purely scalars in the last layer

    def forward(self, data):
        inv_node_feat, equiv_node_feat, conv_args = self._embedding(data)

        ### MACE has a readout block before convolutions ###
        output = self.multihead_decoders[0](
            data, data.node_attributes
        )  # [index][n_output, size_output]
        outputs = output

        ### Do conv --> readout --> repeat for each convolution layer ###
        for conv, readout in zip(self.graph_convs, self.multihead_decoders[1:]):
            if not self.conv_checkpointing:
                inv_node_feat, equiv_node_feat = conv(
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
                output = readout(
                    data, torch.cat([inv_node_feat, equiv_node_feat], dim=1)
                )  # [index][n_output, size_output]
            else:
                inv_node_feat, equiv_node_feat = checkpoint(
                    conv,
                    use_reentrant=False,
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
                output = readout(
                    data, torch.cat([inv_node_feat, equiv_node_feat], dim=1)
                )  # output is a list of tensors with [index][n_output, size_output]
            # Sum predictions for each index, taking care of size differences
            for idx, prediction in enumerate(output):
                outputs[idx] = outputs[idx] + prediction

        return outputs

    def _embedding(self, data):
        super()._embedding(data)

        assert (
            data.pos is not None
        ), "MACE requires node positions (data.pos) to be set."

        # Center positions at 0 per graph. This is a requirement for equivariant models that
        # initialize the spherical harmonics, since the initial spherical harmonic projection
        # uses the nodal position vector  x/||x|| as the input to the spherical harmonics.
        # If we didn't center at 0, these models wouldn't even be invariant to translation.
        if data.batch is None:
            mean_pos = data.pos.mean(dim=0, keepdim=True)
            data.pos = data.pos - mean_pos
        else:
            mean_pos = scatter(data.pos, data.batch, dim=0, reduce="mean")
            data.pos = data.pos - mean_pos[data.batch]

        # Get edge vectors and distances
        edge_vec, edge_dist = get_edge_vectors_and_lengths(
            data.pos, data.edge_index, data.edge_shifts
        )

        # Create node_attrs from atomic numbers. Later on it may contain more information
        ## Node attrs are intrinsic properties of the atoms. Currently, MACE only supports atomic number node attributes
        ## data.node_attrs is already used in another place, so has been renamed to data.node_attributes from MACE and same with other data variable names
        data.node_attributes = process_node_attributes(data["x"], self.num_elements)

        # Embeddings
        node_feats = self.node_embedding(data["node_attributes"])
        edge_attributes = self.spherical_harmonics(edge_vec)
        if self.use_edge_attr:
            edge_attributes = torch.cat([data.edge_attr, edge_attributes], dim=1)
        edge_features = self.radial_embedding(
            edge_dist, data["node_attributes"], data["edge_index"], self.atomic_numbers
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

        return (
            data.node_features[:, : self.hidden_dim],
            data.node_features[:, self.hidden_dim :],
            conv_args,
        )

    def _multihead(self):
        # NOTE Multihead is skipped as it's an integral part of MACE's architecture to have a decoder after every layer,
        # and a convolutional layer in decoding is not supported. Therefore, this final step is not necessary for MACE.
        # However, various parts of multihead are applied in the LinearMultiheadBlock and NonLinearMultiheadBlock classes.
        pass

    def __str__(self):
        return "MACEStack"


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


def get_multihead_decoder(
    nonlinear: bool,
    input_irreps,
    config_heads,
    head_dims,
    head_type,
    num_heads,
    activation_function,
    num_nodes,
):
    if nonlinear:
        return NonLinearMultiheadDecoderBlock(
            input_irreps,
            config_heads,
            head_dims,
            head_type,
            num_heads,
            activation_function,
            num_nodes,
        )
    else:
        return LinearMultiheadDecoderBlock(
            input_irreps,
            config_heads,
            head_dims,
            head_type,
            num_heads,
            activation_function,
            num_nodes,
        )
