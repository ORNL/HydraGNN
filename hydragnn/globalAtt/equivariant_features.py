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

import torch
from e3nn import o3


class ScalarVectorIrrepsAdapter(torch.nn.Module):
    """Convert HydraGNN scalar/vector features to an e3nn representation.

    PaiNN and PNAEq store ``channels`` invariant features as ``[N, C]`` and
    the same number of Cartesian vector features as ``[N, 3, C]``.  e3nn
    represents these features as a single ``C x 0e + C x 1o`` tensor with
    shape ``[N, 4 * C]``.  This adapter defines the conversion at the boundary
    of the equivariant global-attention implementation.

    The vector features use odd parity (``1o``), as appropriate for polar
    vectors.  Encoding and decoding only rearrange tensor entries; they do not
    contain trainable parameters.
    """

    def __init__(self, channels: int):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")

        self.channels = channels
        self.irreps = o3.Irreps(f"{channels}x0e + {channels}x1o")

    def forward(
        self, inv_node_feat: torch.Tensor, equiv_node_feat: torch.Tensor
    ) -> torch.Tensor:
        """Encode scalar and vector features as one e3nn irrep tensor."""
        self._validate_hydragnn_features(inv_node_feat, equiv_node_feat)
        vectors = equiv_node_feat.transpose(1, 2).reshape(
            inv_node_feat.shape[0], 3 * self.channels
        )
        return torch.cat((inv_node_feat, vectors), dim=-1)

    def decode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode an e3nn irrep tensor into HydraGNN scalar/vector layouts."""
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [N, {self.irreps.dim}], "
                f"but has shape {tuple(features.shape)}"
            )
        if features.shape[1] != self.irreps.dim:
            raise ValueError(
                f"features must have {self.irreps.dim} entries per node, "
                f"but has {features.shape[1]}"
            )

        scalars = features[:, : self.channels]
        vectors = features[:, self.channels :].reshape(
            features.shape[0], self.channels, 3
        )
        return scalars, vectors.transpose(1, 2).contiguous()

    def _validate_hydragnn_features(
        self, inv_node_feat: torch.Tensor, equiv_node_feat: torch.Tensor
    ) -> None:
        if inv_node_feat.ndim != 2 or inv_node_feat.shape[1] != self.channels:
            raise ValueError(
                f"inv_node_feat must have shape [N, {self.channels}], "
                f"but has shape {tuple(inv_node_feat.shape)}"
            )
        expected_vector_shape = (
            inv_node_feat.shape[0],
            3,
            self.channels,
        )
        if tuple(equiv_node_feat.shape) != expected_vector_shape:
            raise ValueError(
                "equiv_node_feat must have shape "
                f"{expected_vector_shape}, but has shape "
                f"{tuple(equiv_node_feat.shape)}"
            )
        if inv_node_feat.device != equiv_node_feat.device:
            raise ValueError("scalar and vector features must be on the same device")
        if inv_node_feat.dtype != equiv_node_feat.dtype:
            raise ValueError("scalar and vector features must have the same dtype")
