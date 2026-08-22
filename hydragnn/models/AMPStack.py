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

# Anisotropic Message Passing (AMP), version 3 -- short-range ML potential.
# Adapted from the authors' MIT-licensed AMP-BMS implementation; its license is
# retained in hydragnn/utils/model/amp/LICENSE.
# Cartesian (Burnham-English) multipole message passing.
#   Github: https://github.com/rinikerlab/amp_bms  (source/amp/AMP.py)
#   Paper:  Thuerlemann et al., J. Am. Chem. Soc. 2026, DOI 10.1021/jacs.6c00217
#
# This stack implements V_AMP from eqs. 3-8. D4 dispersion, reaction-field
# electrostatics, ZBL repulsion, and ML/MM electrostatic embedding are separate
# physical energy terms in the paper and are not part of this encoder.

import torch
import torch.nn as nn
import torch_scatter

from .Base import Base
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths
from hydragnn.utils.model.amp import (
    ff_module,
    build_Rx2,
    build_poles,
    aniso_features,
    BesselKernel,
)


class AMPStack(Base):
    """AMP ML-zone stack: Cartesian-multipole anisotropic message passing with a
    per-atom energy readout wired into HydraGNN's node/graph heads."""

    def __init__(
        self,
        input_args,
        conv_args,
        num_radial: int,
        radius: float,
        edge_size: int,
        num_channels: int,
        envelope_p: float,
        max_z: int,
        *args,
        trainable_bessel: bool = False,
        **kwargs,
    ):
        if num_radial <= 0:
            raise ValueError("AMP requires num_radial > 0.")
        if radius <= 0:
            raise ValueError("AMP requires radius > 0.")
        if edge_size <= 0:
            raise ValueError("AMP requires amp_edge_size > 0.")
        if num_channels <= 0:
            raise ValueError("AMP requires amp_num_channels > 0.")
        if max_z <= 0:
            raise ValueError("AMP requires amp_max_z > 0.")

        self.num_radial = int(num_radial)
        self.radius = float(radius)
        self.edge_size = int(edge_size)
        self.num_channels = int(num_channels)
        self.envelope_p = float(envelope_p)
        self.max_z = int(max_z)
        self.trainable_bessel = bool(trainable_bessel)

        # AMP constructs its own edge representation. ``is_edge_model`` in Base
        # refers specifically to user-provided data.edge_attr, which AMP ignores.
        self.is_edge_model = False
        super().__init__(input_args, conv_args, *args, **kwargs)

        if self.use_global_attn:
            raise ValueError("AMP does not support HydraGNN global attention.")

        # Node/edge embeddings (see reference AMP._embed). Node features come from
        # an atomic-number embedding of width node_size == hidden_dim.
        self.node_embedding = nn.Embedding(self.max_z, self.hidden_dim)
        self.radial_embedding = BesselKernel(
            cutoff=self.radius,
            n_bessel=self.num_radial,
            trainable=self.trainable_bessel,
            p=self.envelope_p,
        )
        # Static edge embedding: [bessel | node_i | node_j] -> edge_size.
        self.edge_embedding = ff_module(
            self.edge_size,
            1,
            self.num_radial + 2 * self.hidden_dim,
            output_size=self.edge_size,
            activation=self.activation_function,
        )

    def _init_conv(self):
        # AMP uses residual node updates, so node width is constant (= hidden_dim)
        # across all message-passing steps.
        for i in range(self.num_conv_layers):
            first_layer = i == 0
            self.graph_convs.append(
                self.get_conv(self.hidden_dim, self.hidden_dim, first_layer=first_layer)
            )
            self.feature_layers.append(nn.Identity())

    def get_conv(
        self, input_dim, output_dim, first_layer=False, last_layer=False, edge_dim=None
    ):
        # AMP conv layers operate at fixed node width; input_dim/output_dim are
        # accepted for interface compatibility with Base but must equal hidden_dim.
        # In particular, AMP uses the paper's MLP readout rather than HydraGNN's
        # optional convolutional output heads.
        if input_dim != self.hidden_dim or output_dim != self.hidden_dim:
            raise ValueError(
                "AMP requires fixed-width message passing and does not support "
                "convolutional output heads; use an 'mlp' node or graph head."
            )
        return AMPMessagePassingLayer(
            node_size=self.hidden_dim,
            num_channels=self.num_channels,
            edge_size=self.edge_size,
            first_layer=first_layer,
            activation=self.activation_function,
        )

    def _post_conv(self, inv_node_feat, feature_layer):
        # Equation 7 already contains the complete residual update. The reference
        # implementation does not apply normalization or another activation here.
        return feature_layer(inv_node_feat)

    def _atomic_numbers(self, data):
        """Return a validated one-dimensional atomic-number tensor.

        HydraGNN atomistic datasets commonly retain ``atomic_numbers`` even when
        ``x`` is sliced to selected input features. PyG datasets often use ``z``.
        Falling back to the first column of ``x`` retains HydraGNN's conventional
        NUM_OF_PROTONS=0 layout.
        """
        if hasattr(data, "atomic_numbers") and data.atomic_numbers is not None:
            atomic_numbers = data.atomic_numbers
        elif hasattr(data, "z") and data.z is not None:
            atomic_numbers = data.z
        elif hasattr(data, "x") and data.x is not None:
            atomic_numbers = data.x[:, 0] if data.x.dim() > 1 else data.x
        else:
            raise ValueError(
                "AMP requires atomic numbers in data.atomic_numbers, data.z, "
                "or the first column of data.x."
            )

        if atomic_numbers.dim() == 2 and atomic_numbers.size(-1) == 1:
            atomic_numbers = atomic_numbers.squeeze(-1)
        if atomic_numbers.dim() != 1:
            raise ValueError(
                "AMP atomic numbers must have shape [num_nodes] or [num_nodes, 1]."
            )
        if atomic_numbers.size(0) != data.pos.size(0):
            raise ValueError(
                "AMP received a different number of atomic numbers and positions."
            )
        if torch.is_floating_point(atomic_numbers):
            rounded = atomic_numbers.round()
            if not torch.equal(atomic_numbers, rounded):
                raise ValueError("AMP atomic numbers must be integer-valued.")
            atomic_numbers = rounded

        atomic_numbers = atomic_numbers.to(device=data.pos.device, dtype=torch.long)
        if atomic_numbers.numel() > 0:
            min_z = int(atomic_numbers.min().item())
            max_z = int(atomic_numbers.max().item())
            if min_z < 0 or max_z >= self.max_z:
                raise ValueError(
                    f"AMP atomic numbers must satisfy 0 <= Z < amp_max_z "
                    f"({self.max_z}); received range [{min_z}, {max_z}]."
                )
        return atomic_numbers

    def _embedding(self, data):
        if not hasattr(data, "pos") or data.pos is None:
            raise ValueError("AMP requires node positions in data.pos.")
        if data.pos.dim() != 2 or data.pos.size(-1) != 3:
            raise ValueError("AMP positions must have shape [num_nodes, 3].")
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("AMP requires data.edge_index.")
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise ValueError("AMP edge_index must have shape [2, num_edges].")

        edge_index = data.edge_index.to(device=data.pos.device, dtype=torch.long)
        if not hasattr(data, "edge_shifts") or data.edge_shifts is None:
            edge_shifts = torch.zeros(
                (edge_index.size(1), 3),
                device=data.pos.device,
                dtype=data.pos.dtype,
            )
        else:
            edge_shifts = data.edge_shifts.to(
                device=data.pos.device, dtype=data.pos.dtype
            )
            if edge_shifts.shape != (edge_index.size(1), 3):
                raise ValueError(
                    "AMP edge_shifts must have shape [num_edges, 3]."
                )

        # Local geometry: unit bond vectors R^1 and traceless bond tensors R^2.
        edge_vectors, dist = get_edge_vectors_and_lengths(
            data.pos, edge_index, edge_shifts, normalize=False
        )
        # The reference implementation constructs its short-range graph with a
        # strict r < r_SR test. Enforce that cutoff even if a caller supplies a
        # larger precomputed graph.
        cutoff_mask = dist.squeeze(-1) < self.radius
        edge_index = edge_index[:, cutoff_mask]
        edge_vectors = edge_vectors[cutoff_mask]
        dist = dist[cutoff_mask]
        if dist.numel() > 0 and torch.any(dist <= 0):
            raise ValueError("AMP does not support zero-length edges/self loops.")

        # Cast geometric features to the parameter dtype while retaining their
        # differentiable connection to data.pos for force training.
        model_dtype = self.node_embedding.weight.dtype
        dist = dist.to(dtype=model_dtype)
        rx1 = edge_vectors.to(dtype=model_dtype) / dist
        rx2 = build_Rx2(rx1)
        bessel, envelope = self.radial_embedding(dist)

        nodes = self.node_embedding(self._atomic_numbers(data))

        # Static edge features from Bessel distances + endpoint element embeddings.
        senders = edge_index[0]
        receivers = edge_index[1]
        edge_in = torch.cat((bessel, nodes[senders], nodes[receivers]), dim=-1)
        edge_attr = self.edge_embedding(edge_in)

        conv_args = {
            "edge_index": edge_index,
            "rx1": rx1,
            "rx2": rx2,
            "envelope": envelope,
            "edge_attr": edge_attr,
        }
        # equiv_node_feat slot carries the previous step's anisotropic features
        # (None before the first step).
        return nodes, None, conv_args


class AMPMessagePassingLayer(nn.Module):
    """One AMP message-passing step (paper eqs. 6-7).

    Builds multipoles from the current node features (via learned per-edge
    coefficients scaling the bond bases), contracts them into invariant
    anisotropic features g_0..g_8, forms invariant messages, and applies a
    residual update to the invariant node features. Returns the updated node
    features and the anisotropic edge features for the next step.
    """

    def __init__(
        self,
        node_size: int,
        num_channels: int,
        edge_size: int,
        first_layer: bool,
        activation,
    ):
        super().__init__()
        self.node_size = node_size
        self.num_channels = num_channels
        self.first_layer = first_layer
        aniso_size = 9 * num_channels

        # Equivariant-coefficient generator: outputs a scalar per (channel, order)
        # used to build the dipole and quadrupole multipoles. The first layer has
        # no anisotropic features yet, so its input omits them.
        eq_in = 2 * node_size + edge_size + (0 if first_layer else aniso_size)
        self.eq_message_layer = ff_module(
            node_size, 2, eq_in, output_size=2 * num_channels, activation=activation
        )
        # Scalar message network (consumes the invariant anisotropic features).
        msg_in = 2 * node_size + aniso_size + edge_size
        self.in_message_layer = ff_module(node_size, 2, msg_in, activation=activation)
        # Residual node-state update.
        self.in_update_layer = ff_module(
            node_size, 2, 2 * node_size, activation=activation
        )

    def forward(
        self,
        inv_node_feat,
        equiv_node_feat,
        edge_index,
        rx1,
        rx2,
        envelope,
        edge_attr,
    ):
        senders = edge_index[0]
        receivers = edge_index[1]
        n_nodes = inv_node_feat.shape[0]

        f_i = inv_node_feat[senders]
        f_j = inv_node_feat[receivers]

        if self.first_layer:
            eq_in = torch.cat((f_i, f_j, edge_attr), dim=-1)
        else:
            # equiv_node_feat holds the previous step's anisotropic edge features.
            if equiv_node_feat is None:
                raise ValueError(
                    "AMP layers after the first require anisotropic edge features."
                )
            eq_in = torch.cat((f_i, f_j, equiv_node_feat, edge_attr), dim=-1)

        coefficients = self.eq_message_layer(eq_in)
        dipos, quads = build_poles(
            coefficients, envelope, rx1, rx2, receivers, n_nodes
        )
        aniso = aniso_features(dipos, quads, rx1, rx2, senders, receivers)

        messages = self.in_message_layer(
            torch.cat((f_i, f_j, aniso, edge_attr), dim=-1)
        )
        messages = messages * envelope
        agg = torch_scatter.scatter_add(messages, receivers, dim=0, dim_size=n_nodes)
        new_inv = inv_node_feat + self.in_update_layer(
            torch.cat((inv_node_feat, agg), dim=-1)
        )
        return new_inv, aniso
