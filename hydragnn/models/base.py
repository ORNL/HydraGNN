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

import inspect

import torch
import torch.nn.functional as F
from torch.nn import Embedding, ModuleList, Sequential, Linear, Module, ModuleDict
from torch_geometric.nn import BatchNorm
from torch.utils.checkpoint import checkpoint

from hydragnn.models.utils import activation_function_selection
from hydragnn.distributed import get_device
from hydragnn.models.attention.gps import HydraGPSConv
from hydragnn.models.attention.equivariant_local_global import (
    EquivariantLocalGlobalConv,
)
from hydragnn.models.common.graph_conditioning import (
    GraphConditioningMixin,
    validate_graph_attr_conditioning_mode,
)
from hydragnn.models.common.losses import MultiHeadLossMixin
from hydragnn.models.common.mlp_node import MLPNode
from hydragnn.models.common.model_controls import ModelControlsMixin
from hydragnn.models.common.pooling import GraphPoolingMixin
import hydragnn.utils.profiling_and_tracing.tracer as tr


class Base(
    GraphPoolingMixin,
    GraphConditioningMixin,
    MultiHeadLossMixin,
    ModelControlsMixin,
    Module,
):
    def __init__(
        self,
        input_args: str,
        conv_args: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: list,
        pe_dim: int,
        global_attn_engine: str,
        global_attn_type: str,
        global_attn_heads: int,
        output_type: list,
        config_heads: dict,
        activation_function_type: str,
        loss_function_type: str,
        equivariance: bool,
        loss_weights: list | None = None,  # per-head weights; default uniform
        freeze_conv=False,
        initial_bias=None,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_nodes: int = None,
        graph_pooling: str = "mean",
        use_graph_attr_conditioning: bool = False,
        graph_attr_conditioning_mode: str = "concat_node",
        equivariant_attn_lmax: int = 1,
        equivariant_attn_num_radial: int = 16,
        equivariant_attn_feedforward_multiplier: int = 2,
        equivariant_attn_allow_scalar_only: bool = False,
        equivariant_attn_require_tensor_coupling: bool = True,
        equivariant_attn_chunk_size: int | None = 512,
        equivariant_attn_coupling_mode: str = "parallel",
        equivariant_attn_periodic: bool = False,
        equivariant_attn_periodic_replication: int | list[int] = 1,
        attn_only: bool = False,
        *,
        equivariant_attn_num_hidden_layers: int = 1,
    ):
        super().__init__()
        self.device = get_device()
        self.input_args = input_args
        self.conv_args = conv_args
        self.global_attn_engine = global_attn_engine
        self.global_attn_type = global_attn_type
        self.attn_only = bool(attn_only)
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.global_attn_heads = global_attn_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.global_attn_dropout = dropout
        self.equivariant_attn_lmax = equivariant_attn_lmax
        self.equivariant_attn_num_radial = equivariant_attn_num_radial
        self.equivariant_attn_num_hidden_layers = equivariant_attn_num_hidden_layers
        self.equivariant_attn_feedforward_multiplier = (
            equivariant_attn_feedforward_multiplier
        )
        self.equivariant_attn_allow_scalar_only = equivariant_attn_allow_scalar_only
        self.equivariant_attn_require_tensor_coupling = (
            equivariant_attn_require_tensor_coupling
        )
        self.equivariant_attn_chunk_size = equivariant_attn_chunk_size
        self.equivariant_attn_coupling_mode = equivariant_attn_coupling_mode
        self.equivariant_attn_periodic = equivariant_attn_periodic
        self.equivariant_attn_periodic_replication = (
            equivariant_attn_periodic_replication
        )
        self.num_conv_layers = num_conv_layers
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()
        self.num_nodes = num_nodes
        ##One head represent one variable
        ##Head can have different sizes, head_dims
        self.heads_NN = ModuleList()
        self.config_heads = config_heads
        self.head_type = output_type
        self.head_dims = output_dim
        self.num_heads = len(self.head_dims)
        ##convolutional layers for node level predictions
        self.convs_node_hidden = ModuleDict({})
        self.batch_norms_node_hidden = ModuleDict({})
        self.convs_node_output = ModuleDict({})
        self.batch_norms_node_output = ModuleDict({})
        self.equivariance = equivariance
        self.activation_function = activation_function_selection(
            activation_function_type
        )
        self.use_graph_attr_conditioning = use_graph_attr_conditioning
        self.graph_attr_conditioning_mode = validate_graph_attr_conditioning_mode(
            graph_attr_conditioning_mode
        )
        self._configure_loss(loss_function_type, loss_weights)

        # Condition to pass edge_attr through forward propagation.
        self.use_edge_attr = False
        if (
            hasattr(self, "edge_dim")
            and self.edge_dim is not None
            and self.edge_dim > 0
        ):
            self.use_edge_attr = True
            if "edge_attr" not in self.input_args:
                self.input_args += ", edge_attr"
            if "edge_attr" not in self.conv_args:
                self.conv_args += ", edge_attr"

        self._configure_graph_pooling(graph_pooling)

        self.input_feature_encoders = ModuleDict()
        self.input_feature_encoding_config = []
        self.atomistic_mode_enabled = False

        # Option to only train final property layers.
        self.freeze_conv = freeze_conv
        # Option to set initially large output bias (UQ).
        self.initial_bias = initial_bias

        # Specify global attention usage: specify input embedding dims and edge embedding dims;
        # if model can handle edge features, enforce use of relative edge encodings
        if self.global_attn_engine:
            self.use_global_attn = True
            self.embed_dim = hidden_dim
            self.edge_embed_dim = (
                hidden_dim
                if self.global_attn_engine == "GPS"
                else (
                    self.edge_dim
                    if hasattr(self, "edge_dim") and self.edge_dim is not None
                    else None
                )
            )
            if self.is_edge_model and self.global_attn_engine == "GPS":
                if "edge_attr" not in self.input_args:
                    self.input_args += ", edge_attr"
                if "edge_attr" not in self.conv_args:
                    self.conv_args += ", edge_attr"
        else:
            self.use_global_attn = False
            # ensure that all inputs maintain original dimensionality if gps is turned off
            self.embed_dim = input_dim
            self.edge_embed_dim = (
                self.edge_dim
                if (hasattr(self, "edge_dim") and (self.edge_dim is not None))
                else None
            )

        self.use_encodings = (
            False  # provision to decouple encodings from attention later
        )

        # Specify learnable embeddings
        if self.use_global_attn or self.use_encodings:
            if self.global_attn_engine == "GPS":
                self.pos_emb = Linear(self.pe_dim, self.hidden_dim, bias=False)
            if self.input_dim:
                self.node_emb = Linear(self.input_dim, self.hidden_dim, bias=False)
                if self.global_attn_engine == "GPS":
                    self.node_lin = Linear(
                        2 * self.hidden_dim, self.hidden_dim, bias=False
                    )
            if self.is_edge_model and self.global_attn_engine == "GPS":
                self.rel_pos_emb = Linear(self.pe_dim, self.hidden_dim, bias=False)
                if self.use_edge_attr:
                    self.edge_emb = Linear(self.edge_dim, self.hidden_dim, bias=False)
                    self.edge_lin = Linear(
                        2 * self.hidden_dim, self.hidden_dim, bias=False
                    )

        # Conditioning modules on data.graph_attr are created lazily on first use.
        self._init_graph_conditioning_modules()

        self._init_conv()
        if self.freeze_conv:
            self._freeze_conv()
        self._multihead()
        if self.initial_bias is not None:
            self._set_bias()

        self.conv_checkpointing = False

    def _apply_global_attn(self, mpnn, irreps=None):
        # choose to use global attention or mpnn
        if self.use_global_attn:
            # specify global attention engine; use this to support more engines in future
            if self.global_attn_engine == "GPS":
                return HydraGPSConv(
                    channels=self.hidden_dim,
                    conv=None if self.attn_only else mpnn,
                    heads=self.global_attn_heads,
                    dropout=self.global_attn_dropout,
                    attn_type=self.global_attn_type,
                )
            if self.global_attn_engine == "EquivariantTransformer":
                if not hasattr(self, "equivariant_attn_adapter_type"):
                    raise ValueError(
                        f"{self.__class__.__name__} does not yet support "
                        "EquivariantTransformer"
                    )
                return EquivariantLocalGlobalConv(
                    channels=self.hidden_dim,
                    conv=mpnn,
                    mpnn_type=self.equivariant_attn_adapter_type,
                    heads=self.global_attn_heads,
                    lmax=self.equivariant_attn_lmax,
                    num_radial=self.equivariant_attn_num_radial,
                    num_hidden_layers=self.equivariant_attn_num_hidden_layers,
                    feedforward_multiplier=(
                        self.equivariant_attn_feedforward_multiplier
                    ),
                    allow_scalar_only=self.equivariant_attn_allow_scalar_only,
                    require_tensor_coupling=(
                        self.equivariant_attn_require_tensor_coupling
                    ),
                    local_equivariance=self.equivariance,
                    chunk_size=self.equivariant_attn_chunk_size,
                    periodic=self.equivariant_attn_periodic,
                    periodic_replication=(self.equivariant_attn_periodic_replication),
                    coupling_mode=self.equivariant_attn_coupling_mode,
                    irreps=irreps,
                )
            raise ValueError(
                f"Unknown global attention engine: {self.global_attn_engine}"
            )
        else:
            return mpnn

    def _init_conv(self):
        self.graph_convs.append(
            self._apply_global_attn(
                self.get_conv(
                    self.embed_dim, self.hidden_dim, edge_dim=self.edge_embed_dim
                )
            )
        )
        self.feature_layers.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            self.graph_convs.append(
                self._apply_global_attn(
                    self.get_conv(
                        self.hidden_dim, self.hidden_dim, edge_dim=self.edge_embed_dim
                    )
                )
            )
            self.feature_layers.append(BatchNorm(self.hidden_dim))

    def configure_input_feature_encoders(self, encodings: list[dict] | None) -> None:
        """Build schema-configured encoders for arbitrary scalar node inputs."""
        self.input_feature_encoding_config = list(encodings or [])
        for index, encoding in enumerate(self.input_feature_encoding_config):
            if encoding["type"] == "embedding":
                self.input_feature_encoders[str(index)] = Embedding(
                    encoding["num_categories"], encoding["embedding_dim"]
                )

    @staticmethod
    def _atomic_numbers(data) -> torch.Tensor:
        """Return canonical atomic numbers for native atomistic encoders."""
        if not hasattr(data, "atomic_numbers") or data.atomic_numbers is None:
            raise ValueError("native atomistic encoding requires data.atomic_numbers")
        values = data.atomic_numbers
        if not torch.is_tensor(values):
            raise TypeError("data.atomic_numbers must be a torch.Tensor")
        return values.long().view(-1)

    def _input_node_features(self, data) -> torch.Tensor:
        """Apply configured categorical encoders to raw named node features."""
        if not self.input_feature_encoding_config:
            return data.x
        pieces = []
        cursor = 0
        for index, encoding in enumerate(self.input_feature_encoding_config):
            start = encoding["start"]
            if start > cursor:
                pieces.append(data.x[:, cursor:start])
            raw = data.x[:, start]
            rounded = raw.round()
            if not torch.equal(raw, rounded):
                raise TypeError(
                    f"encoded node feature '{encoding['name']}' must contain integers"
                )
            values = rounded.long()
            minimum = encoding["min_value"]
            maximum = minimum + encoding["num_categories"] - 1
            if values.numel() and not torch.all(
                (values >= minimum) & (values <= maximum)
            ):
                raise ValueError(
                    f"encoded node feature '{encoding['name']}' must satisfy "
                    f"{minimum} <= value <= {maximum}"
                )
            category_indices = values - minimum
            if encoding["type"] == "embedding":
                pieces.append(self.input_feature_encoders[str(index)](category_indices))
            else:
                pieces.append(
                    F.one_hot(
                        category_indices, num_classes=encoding["num_categories"]
                    ).float()
                )
            cursor = start + encoding["dim"]
        if cursor < data.x.shape[1]:
            pieces.append(data.x[:, cursor:])
        return torch.cat(pieces, dim=1)

    def _embedding(self, data):
        if not hasattr(data, "edge_shifts"):
            data.edge_shifts = torch.zeros(
                (data.edge_index.size(1), 3), device=data.edge_index.device
            )
        conv_args = {"edge_index": data.edge_index.to(torch.long)}
        if self.use_global_attn:
            conv_args["graph_batch"] = getattr(data, "batch", None)
        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_attr})

        node_features = self._input_node_features(data)
        if self.use_global_attn:
            if self.global_attn_engine == "EquivariantTransformer":
                if not self.input_dim:
                    raise ValueError(
                        "EquivariantTransformer requires invariant node features"
                    )
                node_features = node_features.to(self.node_emb.weight.dtype)
                return self.node_emb(node_features), data.pos, conv_args
            # encode node positional embeddings
            x = self.pos_emb(data.pe)
            # if node features are available, generate mebeddings, concatenate with positional embeddings and map to hidden dim
            if self.input_dim:
                node_features = node_features.to(self.node_emb.weight.dtype)
                x = torch.cat((self.node_emb(node_features), x), 1)
                x = self.node_lin(x)
            # repeat for edge features and relative edge encodings
            if self.is_edge_model:
                e = self.rel_pos_emb(data.rel_pe)
                if self.use_edge_attr:
                    e = torch.cat((self.edge_emb(conv_args["edge_attr"]), e), 1)
                    e = self.edge_lin(e)
                conv_args.update({"edge_attr": e})
            return x, data.pos, conv_args
        else:
            return node_features, data.pos, conv_args

    def _equivariant_attention_geometry(self, data):
        """Collect global-attention geometry under the configured PBC policy."""
        batch = (
            data.batch
            if data.batch is not None
            else torch.zeros(
                data.pos.shape[0], dtype=torch.long, device=data.pos.device
            )
        )
        has_periodic_edges = hasattr(data, "edge_shifts") and torch.any(
            data.edge_shifts != 0
        )
        if has_periodic_edges and not self.equivariant_attn_periodic:
            raise ValueError(
                "nonzero periodic edge shifts require " "equivariant_attn_periodic=true"
            )
        geometry = {"positions": data.pos, "graph_batch": batch}
        if self.equivariant_attn_periodic:
            if not hasattr(data, "cell") or not hasattr(data, "pbc"):
                raise ValueError(
                    "periodic equivariant attention requires data.cell and data.pbc"
                )
            geometry.update({"cell": data.cell, "pbc": data.pbc})
        return geometry

    def _init_node_conv(self):
        # *******convolutional layers for node level predictions*******#
        # two ways to implement node features from here:
        # 1. one graph for all node features
        # 2. one graph for one node features (currently implemented)
        nodeconfiglist = self.config_heads["node"]
        assert (
            self.num_branches == len(nodeconfiglist) or self.num_branches == 1
        ), "asumming node head has the same branches as graph head, if any"
        for branchdict in nodeconfiglist:
            # only support conv for all node branches
            if branchdict["architecture"]["type"] != "conv":
                return
        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return

        for branchdict in nodeconfiglist:
            branchtype = branchdict["type"]
            brancharct = branchdict["architecture"]
            num_conv_layers_node = brancharct["num_headlayers"]
            hidden_dim_node = brancharct["dim_headlayers"]

            convs_node_hidden = ModuleList()
            batch_norms_node_hidden = ModuleList()
            convs_node_output = ModuleList()
            batch_norms_node_output = ModuleList()

            # In this part, each head has same number of convolutional layers, but can have different output dimension
            if "last_layer" in inspect.signature(self.get_conv).parameters:
                convs_node_hidden.append(
                    self.get_conv(self.hidden_dim, hidden_dim_node[0], last_layer=False)
                )
            else:
                convs_node_hidden.append(
                    self.get_conv(self.hidden_dim, hidden_dim_node[0])
                )
            batch_norms_node_hidden.append(BatchNorm(hidden_dim_node[0]))
            for ilayer in range(num_conv_layers_node - 1):
                # This check is needed because the "get_conv" method of SCFStack takes one additional argument called last_layer
                if "last_layer" in inspect.signature(self.get_conv).parameters:
                    convs_node_hidden.append(
                        self.get_conv(
                            hidden_dim_node[ilayer],
                            hidden_dim_node[ilayer + 1],
                            last_layer=False,
                        )
                    )
                else:
                    convs_node_hidden.append(
                        self.get_conv(
                            hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1]
                        )
                    )
                batch_norms_node_hidden.append(BatchNorm(hidden_dim_node[ilayer + 1]))
            for ihead in node_feature_ind:
                # This check is needed because the "get_conv" method of SCFStack takes one additional argument called last_layer
                if "last_layer" in inspect.signature(self.get_conv).parameters:
                    convs_node_output.append(
                        self.get_conv(
                            hidden_dim_node[-1],
                            self.head_dims[ihead] * (1 + self.var_output),
                            last_layer=True,
                        )
                    )
                else:
                    convs_node_output.append(
                        self.get_conv(
                            hidden_dim_node[-1],
                            self.head_dims[ihead] * (1 + self.var_output),
                        )
                    )
                batch_norms_node_output.append(
                    BatchNorm(self.head_dims[ihead] * (1 + self.var_output))
                )
            self.convs_node_hidden[branchtype] = convs_node_hidden
            self.batch_norms_node_hidden[branchtype] = batch_norms_node_hidden
            self.convs_node_output[branchtype] = convs_node_output
            self.batch_norms_node_output[branchtype] = batch_norms_node_output

    def _multihead(self):
        # typename = config_heads_type["type"]
        # self.multiheads[typename]=Module()
        # self.multiheads[typename].heads_NN=ModuleList()

        self.graph_shared = ModuleDict({})
        ############multiple heads/taks################
        # shared dense layers for heads with graph level output
        dim_sharedlayers = 0
        self.num_branches = 1
        if "graph" in self.config_heads:
            self.num_branches = len(self.config_heads["graph"])
            for branchdict in self.config_heads["graph"]:
                denselayers = []
                dim_sharedlayers = branchdict["architecture"]["dim_sharedlayers"]
                denselayers.append(Linear(self.hidden_dim, dim_sharedlayers))
                denselayers.append(self.activation_function)
                for ishare in range(branchdict["architecture"]["num_sharedlayers"] - 1):
                    denselayers.append(Linear(dim_sharedlayers, dim_sharedlayers))
                    denselayers.append(self.activation_function)
                self.graph_shared[branchdict["type"]] = Sequential(*denselayers)

        if "node" in self.config_heads:
            self._init_node_conv()

        inode_feature = 0
        for ihead in range(self.num_heads):
            # mlp for each head output
            head_NN = ModuleDict({})
            if self.head_type[ihead] == "graph":
                for branchdict in self.config_heads["graph"]:
                    branchtype = branchdict["type"]
                    brancharct = branchdict["architecture"]
                    dim_sharedlayers = brancharct["dim_sharedlayers"]
                    num_head_hidden = brancharct["num_headlayers"]
                    dim_head_hidden = brancharct["dim_headlayers"]
                    denselayers = []
                    denselayers.append(Linear(dim_sharedlayers, dim_head_hidden[0]))
                    denselayers.append(self.activation_function)
                    for ilayer in range(num_head_hidden - 1):
                        denselayers.append(
                            Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                        )
                        denselayers.append(self.activation_function)
                    denselayers.append(
                        Linear(
                            dim_head_hidden[-1],
                            self.head_dims[ihead] * (1 + self.var_output),
                        )
                    )
                    head_NN[branchtype] = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                for branchdict in self.config_heads["node"]:
                    branchtype = branchdict["type"]
                    brancharct = branchdict["architecture"]
                    hidden_dim_node = brancharct["dim_headlayers"]
                    node_NN_type = brancharct["type"]
                    if node_NN_type == "mlp" or node_NN_type == "mlp_per_node":
                        self.num_mlp = 1 if node_NN_type == "mlp" else self.num_nodes
                        if node_NN_type == "mlp_per_node":
                            assert (
                                self.num_nodes is not None
                            ), "num_nodes must be provided for mlp_per_node; use 'mlp' for variable-size graphs"
                        head_NN[branchtype] = MLPNode(
                            self.hidden_dim,
                            self.head_dims[ihead] * (1 + self.var_output),
                            self.num_mlp,
                            hidden_dim_node,
                            node_NN_type,
                            self.activation_function,
                            num_nodes=(
                                self.num_nodes
                                if node_NN_type == "mlp_per_node"
                                else None
                            ),
                        )
                    elif node_NN_type == "conv":
                        head_NN[branchtype] = ModuleList()
                        for conv, batch_norm in zip(
                            self.convs_node_hidden[branchtype],
                            self.batch_norms_node_hidden[branchtype],
                        ):
                            head_NN[branchtype].append(conv)
                            head_NN[branchtype].append(batch_norm)
                        head_NN[branchtype].append(
                            self.convs_node_output[branchtype][inode_feature]
                        )
                        head_NN[branchtype].append(
                            self.batch_norms_node_output[branchtype][inode_feature]
                        )
                        inode_feature += 1
                    else:
                        raise ValueError(
                            "Unknown head NN structure for node features"
                            + node_NN_type
                            + "; currently only support 'mlp', 'mlp_per_node' or 'conv' (can be set with config['NeuralNetwork']['Architecture']['output_heads']['node']['type'], e.g., ./examples/ci_multihead.json)"
                        )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def _postprocess_conv_output(self, inv_node_feat, batch, data, feat_layer):
        """Apply HydraGNN processing after one convolution layer.

        Monolithic external backbones skip the generic normalization and
        activation because their encoder output is already final, but explicit
        graph conditioning must still affect that output.
        """
        if getattr(self, "use_graph_attr_conditioning", False):
            inv_node_feat = self._apply_graph_conditioning(inv_node_feat, batch, data)
        if getattr(self, "skip_post_conv_processing", False):
            return inv_node_feat
        return self.activation_function(feat_layer(inv_node_feat))

    def forward(self, data):
        ### encoder part ####
        tr.start("enc_forward")
        inv_node_feat, equiv_node_feat, conv_args = self._embedding(data)

        # Prepare batch indexing for graph conditioning (handles single-graph case gracefully).
        batch_for_cond = (
            data.batch if hasattr(data, "batch") and data.batch is not None else None
        )

        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                inv_node_feat, equiv_node_feat = conv(
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
            else:
                inv_node_feat, equiv_node_feat = checkpoint(
                    conv,
                    use_reentrant=False,
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )

            # External/monolithic backbones may already have completed their
            # encoder in ``_embedding``.  Their identity conv must be a genuine
            # no-op rather than an extra HydraGNN conditioning/activation step.
            inv_node_feat = self._postprocess_conv_output(
                inv_node_feat, batch_for_cond, data, feat_layer
            )

        x = inv_node_feat
        tr.stop("enc_forward")
        tr.start("branch_forward")
        #### multi-head decoder part####
        # shared dense layers for graph level output
        if data.batch is None:
            x_graph = self._pool_graph_features(x, None)
            # individual samplers
            data.batch = data.x * 0
        else:
            x_graph = self._pool_graph_features(x, data.batch)

        # A node-only model does not consume the pooled representation.  Do not
        # run a pool conditioner whose parameters cannot contribute to its loss.
        if "graph" in self.head_type:
            x_graph = self._apply_graph_pool_conditioning(x_graph, data)
        outputs = []
        outputs_var = []
        # if no dataset_name, set it to be 0
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
                        mask = data.dataset_name == ID
                        mask = mask[:, 0]
                        branchtype = f"branch-{ID.item()}"
                        # print("Pei debugging:", branchtype, data.dataset_name, mask, data.dataset_name[mask])
                        x_graph_head = self.graph_shared[branchtype](x_graph[mask, :])
                        output_head = headloc[branchtype](x_graph_head)
                        head[mask] = output_head[:, :head_dim].to(dtype=out_dtype)
                        headvar[mask] = (output_head[:, head_dim:] ** 2).to(
                            dtype=out_dtype
                        )
                outputs.append(head)
                outputs_var.append(headvar)
            else:
                # assuming all node types are the same
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
                        inv_node_feat = x
                        equiv_node_feat_ = equiv_node_feat
                        for conv, batch_norm in zip(
                            headloc[branchtype][0::2], headloc[branchtype][1::2]
                        ):
                            inv_node_feat, equiv_node_feat_ = conv(
                                inv_node_feat=inv_node_feat,
                                equiv_node_feat=equiv_node_feat_,
                                **conv_args,
                            )
                            inv_node_feat = batch_norm(inv_node_feat)
                            inv_node_feat = self.activation_function(inv_node_feat)
                        x_node = inv_node_feat
                    else:
                        x_node = headloc[branchtype](x=x, batch=data.batch)
                    head = x_node[:, :head_dim]
                    headvar = x_node[:, head_dim:] ** 2
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask_nodes = torch.repeat_interleave(mask, node_counts)
                        branchtype = f"branch-{ID.item()}"
                        # print("Pei debugging:", branchtype, data.dataset_name, mask, data.dataset_name[mask])
                        if node_NN_type == "conv":
                            inv_node_feat = x[mask_nodes, :]
                            equiv_node_feat_ = equiv_node_feat[mask_nodes, :]
                            for conv, batch_norm in zip(
                                headloc[branchtype][0::2], headloc[branchtype][1::2]
                            ):
                                inv_node_feat, equiv_node_feat_ = conv(
                                    inv_node_feat=inv_node_feat,
                                    equiv_node_feat=equiv_node_feat_,
                                    **conv_args,
                                )
                                inv_node_feat = batch_norm(inv_node_feat)
                                inv_node_feat = self.activation_function(inv_node_feat)
                            x_node = inv_node_feat
                        else:
                            x_node = headloc[branchtype](
                                x=x[mask_nodes, :], batch=data.batch[mask_nodes]
                            )
                        head[mask_nodes] = x_node[:, :head_dim].to(dtype=out_dtype)
                        headvar[mask_nodes] = (x_node[:, head_dim:] ** 2).to(
                            dtype=out_dtype
                        )
                outputs.append(head)
                outputs_var.append(headvar)
        tr.stop("branch_forward")
        if self.var_output:
            return outputs, outputs_var
        return outputs

    def __str__(self):
        return "Base"
