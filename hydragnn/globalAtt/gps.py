##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# Copyright (c) 2023, PyG Team <team@pyg.org>                                #
#                                                                            #
# Adapted from torch_geometric.nn.conv.GPSConv (PyG 2.8.0), distributed      #
# under the MIT License. HydraGNN modifications are distributed under the    #
# BSD 3-clause license. See LICENSE and LICENSES/PYG-MIT.txt.                 #
#                                                                            #
# SPDX-License-Identifier: MIT AND BSD-3-Clause                              #
##############################################################################

import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.utils import to_dense_batch


class HydraGPSConv(torch.nn.Module):
    r"""PyG ``GPSConv`` adapted to HydraGNN's two-stream feature interface.

    The local message-passing layer consumes and returns invariant and
    equivariant node features. Global attention is applied only to invariant
    features, while the updated equivariant features are propagated unchanged
    by the global branch.

    This implementation tracks ``torch_geometric.nn.conv.GPSConv`` from PyG
    2.8.0. The differences are intentionally limited to the local-convolution
    call signature and the tuple return value.
    """

    pyg_source_version = "2.8.0"

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == "multihead":
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == "performer":
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f"{attn_type} is not supported")

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        self._performer_steps_since_redraw = 0
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = "batch" in signature.parameters

    def _maybe_redraw_performer_projection(
        self, redraw_interval: Optional[int]
    ) -> bool:
        """Redraw Performer's random-feature projection when it is due.

        Performer uses random features to approximate softmax attention. A
        periodic redraw during training prevents optimization from depending
        on only the initial random approximation. ``redraw_interval`` counts
        calls to this method, with one call expected per training batch.

        This method is intentionally called by the training loop rather than
        from :meth:`forward`, since a forward may be repeated by activation
        checkpointing or other execution strategies.

        Args:
            redraw_interval: Number of training batches between redraws. A
                value of ``None`` keeps the initial projection fixed.

        Returns:
            ``True`` if the projection was redrawn, otherwise ``False``. No
            redraw occurs for non-Performer attention or in evaluation mode.

        Raises:
            ValueError: If ``redraw_interval`` is not positive or ``None``.
        """
        if redraw_interval is None or self.attn_type != "performer":
            return False
        if redraw_interval <= 0:
            raise ValueError("Performer redraw interval must be positive or None")
        if not self.training:
            return False

        self._performer_steps_since_redraw += 1
        if self._performer_steps_since_redraw < redraw_interval:
            return False

        self.attn.redraw_projection_matrix()
        self._performer_steps_since_redraw = 0
        return True

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        inv_node_feat: Tensor,
        equiv_node_feat: Tensor,
        graph_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h, equiv_node_feat = self.conv(
                inv_node_feat=inv_node_feat, equiv_node_feat=equiv_node_feat, **kwargs
            )
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + inv_node_feat
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=graph_batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(inv_node_feat, graph_batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + inv_node_feat  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=graph_batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=graph_batch)
            else:
                out = self.norm3(out)

        return out, equiv_node_feat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"attn_type={self.attn_type})"
        )


def redraw_performer_projections(
    model: torch.nn.Module, redraw_interval: Optional[int]
) -> int:
    """Advance the redraw schedule for every ``HydraGPSConv`` in ``model``.

    Call this exactly once per training batch, immediately before the model
    forward. The return value is the number of layers whose Performer random
    projection was redrawn by this call.
    """
    return sum(
        module._maybe_redraw_performer_projection(redraw_interval)
        for module in model.modules()
        if isinstance(module, HydraGPSConv)
    )
