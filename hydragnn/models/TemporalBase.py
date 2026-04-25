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

"""
TemporalBase — adds a GRU/LSTM temporal backbone on top of any HydraGNN spatial
stack while leaving Base.py completely untouched.

Design
------
TemporalBase inherits from Base. Concrete temporal models are created by
combining TemporalBase with any existing spatial stack via Python's MRO:

    class TemporalGINStack(TemporalBase, GINStack): ...
    class TemporalPNAStack(TemporalBase, PNAStack): ...

MRO:  TemporalGINStack → TemporalBase → GINStack → Base → Module

This means:
  * TemporalBase.forward()     — handles the temporal loop
  * GINStack.get_conv()        — supplies the spatial convolution implementation
  * Base._init_conv()          — populates self.graph_convs / self.feature_layers
  * Base._embedding()          — called at each timestep inside _spatial_encode_step()
  * Base._multihead()          — builds all output heads (used in _temporal_decode)

Data contract
-------------
Temporal mode is activated when data.x_seq is present:

    data.x_seq      : Tensor [N_total, T, F]   node feature sequences
    data.edge_index : Tensor [2, E]             static topology (shared across T)
    data.x          : Tensor [N_total, F]       single-step snapshot (fallback path)

If data.x_seq is absent, forward() delegates to Base.forward() unchanged.

temporal_mode choices
---------------------
post_gcn   (default, T-GCN style)
    For each timestep t: run the full spatial GCN stack → node embedding h_t.
    Stack [h_1,...,h_T] and run RNN → last hidden state → decoder.

pre_gcn
    Run RNN over raw feature sequence [N, T, F] → per-node summary.
    Project summary to input_dim and run the full GCN stack once → decoder.

interleaved
    At each GCN layer l: apply GCN_l across all T embeddings, collapse with
    a dedicated RNN_l, broadcast the result back to T copies for the next layer.
    Requires one RNN per GCN layer (self.temporal_rnns ModuleList).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.models.Base import Base


class TemporalBase(Base):
    """Abstract temporal base — must be combined with a spatial stack via MRO."""

    def __init__(
        self,
        *args,
        temporal_backbone: str = "gru",
        temporal_hidden_dim: int = None,
        temporal_num_layers: int = 1,
        temporal_mode: str = "post_gcn",
        **kwargs,
    ):
        # ------------------------------------------------------------------ #
        # 1. Fully initialise the spatial stack first (GINStack → Base).      #
        #    graph_convs, feature_layers, heads_NN, etc. are all ready after  #
        #    this call.                                                        #
        # ------------------------------------------------------------------ #
        super().__init__(*args, **kwargs)

        # ------------------------------------------------------------------ #
        # 2. Validate and store temporal hyperparameters.                     #
        # ------------------------------------------------------------------ #
        backbone = temporal_backbone.lower()
        if backbone not in ("gru", "lstm"):
            raise ValueError(
                f"temporal_backbone must be 'gru' or 'lstm', got '{temporal_backbone}'"
            )
        self._backbone_type = backbone

        mode = temporal_mode.lower()
        if mode not in ("post_gcn", "pre_gcn", "interleaved"):
            raise ValueError(
                f"temporal_mode must be 'post_gcn', 'pre_gcn', or 'interleaved', "
                f"got '{temporal_mode}'"
            )
        self._temporal_mode = mode

        # ------------------------------------------------------------------ #
        # 3. Determine RNN input/output sizes.                                #
        #                                                                     #
        #   post_gcn / interleaved : RNN sees GCN output  → size hidden_dim  #
        #   pre_gcn               : RNN sees raw features → size input_dim   #
        # ------------------------------------------------------------------ #
        rnn_input_size = self.input_dim if mode == "pre_gcn" else self.hidden_dim

        # t_hidden defaults to match what the RNN will feed into the decoder:
        #   pre_gcn  → project back to input_dim so the GCN can run normally
        #   post_gcn / interleaved → project to hidden_dim for the decoder
        proj_output_dim = self.input_dim if mode == "pre_gcn" else self.hidden_dim
        t_hidden = (
            temporal_hidden_dim if temporal_hidden_dim is not None else proj_output_dim
        )

        # ------------------------------------------------------------------ #
        # 4. Build the RNN module(s).                                         #
        # ------------------------------------------------------------------ #
        rnn_cls = nn.GRU if backbone == "gru" else nn.LSTM

        if mode == "interleaved":
            # One RNN per GCN layer; each takes hidden_dim → t_hidden.
            self.temporal_rnns = nn.ModuleList(
                [
                    rnn_cls(
                        self.hidden_dim,
                        t_hidden,
                        num_layers=temporal_num_layers,
                        batch_first=True,
                        dropout=self.dropout if temporal_num_layers > 1 else 0.0,
                    )
                    for _ in range(self.num_conv_layers)
                ]
            )
            self.temporal_rnn = None
        else:
            self.temporal_rnn = rnn_cls(
                rnn_input_size,
                t_hidden,
                num_layers=temporal_num_layers,
                batch_first=True,
                dropout=self.dropout if temporal_num_layers > 1 else 0.0,
            )
            self.temporal_rnns = None

        # Optional linear projection: maps t_hidden → proj_output_dim.
        # Uses Identity when the sizes already match to avoid wasted parameters.
        self.temporal_proj = (
            nn.Linear(t_hidden, proj_output_dim)
            if t_hidden != proj_output_dim
            else nn.Identity()
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _rnn_last_hidden(self, rnn: nn.Module, H: torch.Tensor) -> torch.Tensor:
        """Run *rnn* on H [N, T, input] and return the last-layer, last-step
        hidden state [N, t_hidden]."""
        if isinstance(rnn, nn.LSTM):
            _, (h_n, _) = rnn(H)
        else:
            _, h_n = rnn(H)
        return h_n[-1]  # [N, t_hidden]

    def _spatial_encode_step(self, data, x_t: torch.Tensor, conv_args: dict = None):
        """Run Base's full spatial encoder for a single timestep.

        Temporarily replaces data.x with x_t so that Base._embedding() and
        every downstream child-class get_conv() receive the correct features.
        data.x is restored before returning.

        Returns
        -------
        inv   : [N_total, hidden_dim]  node embeddings after all GCN layers
        equiv : equivariant node features (data.pos for non-equivariant stacks)
        conv_args : dict forwarded to GCN layers (edge_index, edge_attr, …)
        """
        saved_x = data.x
        data.x = x_t.float()
        inv, equiv, c_args = self._embedding(data)
        data.x = saved_x

        # conv_args (edge_index, optionally edge_attr) is static across timesteps;
        # accept a pre-computed version or use the freshly computed one.
        if conv_args is None:
            conv_args = c_args

        batch_fc = (
            data.batch if hasattr(data, "batch") and data.batch is not None else None
        )

        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                inv, equiv = conv(inv_node_feat=inv, equiv_node_feat=equiv, **conv_args)
            else:
                inv, equiv = checkpoint(
                    conv,
                    use_reentrant=False,
                    inv_node_feat=inv,
                    equiv_node_feat=equiv,
                    **conv_args,
                )
            inv = self._apply_graph_conditioning(inv, batch_fc, data)
            inv = self.activation_function(feat_layer(inv))

        return inv, equiv, conv_args

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(self, data):
        """If data.x_seq is present, run the temporal path; otherwise fall back
        to the standard static Base.forward() path unchanged."""
        if not (hasattr(data, "x_seq") and data.x_seq is not None):
            return super().forward(data)
        return self._forward_temporal(data)

    def _forward_temporal(self, data):
        x_seq = data.x_seq  # [N_total, T, F]
        T = x_seq.shape[1]

        tr.start("enc_forward")

        # ------------------------------------------------------------------ #
        # post_gcn: GCN at every timestep → RNN over resulting embeddings.   #
        # ------------------------------------------------------------------ #
        if self._temporal_mode == "post_gcn":
            # Compute static conv_args once from the first timestep.
            saved_x = data.x
            data.x = x_seq[:, 0, :].float()
            _, _, conv_args = self._embedding(data)
            data.x = saved_x

            steps = []
            for t in range(T):
                inv, equiv, conv_args = self._spatial_encode_step(
                    data, x_seq[:, t, :], conv_args
                )
                steps.append(inv)  # [N_total, hidden]

            H = torch.stack(steps, dim=1)  # [N_total, T, hidden]
            h = self._rnn_last_hidden(self.temporal_rnn, H)  # [N_total, t_hidden]
            x = self.temporal_proj(h)  # [N_total, hidden]
            equiv_final = equiv

        # ------------------------------------------------------------------ #
        # pre_gcn: RNN over raw features → summary → GCN once.               #
        # ------------------------------------------------------------------ #
        elif self._temporal_mode == "pre_gcn":
            H_in = x_seq.float()  # [N_total, T, F]
            h = self._rnn_last_hidden(self.temporal_rnn, H_in)  # [N_total, t_hidden]
            x_summary = self.temporal_proj(h)  # [N_total, input_dim]
            x, equiv_final, conv_args = self._spatial_encode_step(data, x_summary)

        # ------------------------------------------------------------------ #
        # interleaved: per-layer GCN → RNN alternation.                      #
        #                                                                     #
        # Layer 0 GCN  ─→ RNN_0 ─→ Layer 1 GCN ─→ RNN_1 ─→ … → decoder    #
        #                                                                     #
        # At each GCN layer, all T embeddings are processed, the T outputs   #
        # are collapsed by RNN_l into one vector, and broadcast back to T     #
        # copies as input to the next GCN layer.                              #
        # ------------------------------------------------------------------ #
        elif self._temporal_mode == "interleaved":
            # Compute conv_args from static topology (edge_index, edge_attr).
            saved_x = data.x
            data.x = x_seq[:, 0, :].float()
            _, _, conv_args = self._embedding(data)
            data.x = saved_x

            # Embed raw node features for every timestep.
            step_inputs = []
            for t in range(T):
                saved_x = data.x
                data.x = x_seq[:, t, :].float()
                inv_t, _, _ = self._embedding(data)  # [N_total, embed_dim]
                data.x = saved_x
                step_inputs.append(inv_t)

            batch_fc = (
                data.batch
                if hasattr(data, "batch") and data.batch is not None
                else None
            )

            current = step_inputs  # list of T tensors, each [N_total, *]
            for conv, feat_layer, rnn_l in zip(
                self.graph_convs, self.feature_layers, self.temporal_rnns
            ):
                gcn_outs = []
                for inv_t in current:
                    if not self.conv_checkpointing:
                        inv_t, _ = conv(
                            inv_node_feat=inv_t,
                            equiv_node_feat=data.pos,
                            **conv_args,
                        )
                    else:
                        inv_t, _ = checkpoint(
                            conv,
                            use_reentrant=False,
                            inv_node_feat=inv_t,
                            equiv_node_feat=data.pos,
                            **conv_args,
                        )
                    inv_t = self._apply_graph_conditioning(inv_t, batch_fc, data)
                    inv_t = self.activation_function(feat_layer(inv_t))
                    gcn_outs.append(inv_t)

                H_l = torch.stack(gcn_outs, dim=1)  # [N_total, T, hidden]
                h_l = self._rnn_last_hidden(rnn_l, H_l)  # [N_total, t_hidden]
                x_l = self.temporal_proj(h_l)  # [N_total, hidden]
                # Broadcast single summary back to T copies for the next layer.
                current = [x_l] * T

            x = current[0]  # [N_total, hidden]
            equiv_final = data.pos

        tr.stop("enc_forward")

        return self._temporal_decode(x, equiv_final, conv_args, data)

    # ---------------------------------------------------------------------- #
    # Decoder                                                                 #
    # ---------------------------------------------------------------------- #

    def _temporal_decode(self, x, equiv_node_feat, conv_args, data):
        """Multi-head decoder that accepts a pre-computed node embedding x.

        Mirrors the decoder section of Base.forward() so that the temporal
        embedding is fed directly into the output heads without re-running
        the spatial encoder.

        NOTE: If the decoder logic in Base.forward() is updated in a future
        version, this method must be synchronised accordingly.
        """
        tr.start("branch_forward")

        if data.batch is None:
            x_graph = self._pool_graph_features(x, None)
            data.batch = data.x * 0
        else:
            x_graph = self._pool_graph_features(x, data.batch)

        x_graph = self._apply_graph_pool_conditioning(x_graph, data)

        outputs, outputs_var = [], []

        if not hasattr(data, "dataset_name"):
            setattr(data, "dataset_name", data.batch.unique() * 0)
        datasetIDs = data.dataset_name.unique()
        unique, node_counts = torch.unique_consecutive(data.batch, return_counts=True)

        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                out_dtype = x_graph.dtype
                head = torch.zeros(
                    (len(data.dataset_name), head_dim),
                    device=x.device,
                    dtype=out_dtype,
                )
                headvar = torch.zeros(
                    (len(data.dataset_name), head_dim * self.var_output),
                    device=x.device,
                    dtype=out_dtype,
                )
                if self.num_branches == 1:
                    x_graph_head = self.graph_shared["branch-0"](x_graph)
                    output_head = headloc["branch-0"](x_graph_head)
                    head = output_head[:, :head_dim]
                    headvar = output_head[:, head_dim:] ** 2
                else:
                    for ID in datasetIDs:
                        mask = (data.dataset_name == ID)[:, 0]
                        branchtype = f"branch-{ID.item()}"
                        x_graph_head = self.graph_shared[branchtype](x_graph[mask, :])
                        output_head = headloc[branchtype](x_graph_head)
                        head[mask] = output_head[:, :head_dim]
                        headvar[mask] = (output_head[:, head_dim:] ** 2).to(
                            dtype=out_dtype
                        )
                outputs.append(head)
                outputs_var.append(headvar)

            else:  # "node"
                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                out_dtype = x.dtype
                head = torch.zeros(
                    (x.shape[0], head_dim), device=x.device, dtype=out_dtype
                )
                headvar = torch.zeros(
                    (x.shape[0], head_dim * self.var_output),
                    device=x.device,
                    dtype=out_dtype,
                )
                if self.num_branches == 1:
                    branchtype = "branch-0"
                    if node_NN_type == "conv":
                        inv = x
                        equiv = equiv_node_feat
                        for conv, batch_norm in zip(
                            headloc[branchtype][0::2],
                            headloc[branchtype][1::2],
                        ):
                            inv, equiv = conv(
                                inv_node_feat=inv,
                                equiv_node_feat=equiv,
                                **conv_args,
                            )
                            inv = batch_norm(inv)
                            inv = self.activation_function(inv)
                        x_node = inv
                    else:
                        x_node = headloc[branchtype](x=x, batch=data.batch)
                    head = x_node[:, :head_dim]
                    headvar = x_node[:, head_dim:] ** 2
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask_nodes = torch.repeat_interleave(mask, node_counts)
                        branchtype = f"branch-{ID.item()}"
                        if node_NN_type == "conv":
                            inv = x[mask_nodes, :]
                            equiv = equiv_node_feat[mask_nodes, :]
                            for conv, batch_norm in zip(
                                headloc[branchtype][0::2],
                                headloc[branchtype][1::2],
                            ):
                                inv, equiv = conv(
                                    inv_node_feat=inv,
                                    equiv_node_feat=equiv,
                                    **conv_args,
                                )
                                inv = batch_norm(inv)
                                inv = self.activation_function(inv)
                            x_node = inv
                        else:
                            x_node = headloc[branchtype](
                                x=x[mask_nodes, :],
                                batch=data.batch[mask_nodes],
                            )
                        head[mask_nodes] = x_node[:, :head_dim]
                        headvar[mask_nodes] = x_node[:, head_dim:] ** 2
                outputs.append(head)
                outputs_var.append(headvar)

        tr.stop("branch_forward")

        if self.var_output:
            return outputs, outputs_var
        return outputs
