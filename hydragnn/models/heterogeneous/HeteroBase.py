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

import torch
from torch.nn import Module, ModuleList, ModuleDict, Linear, Sequential
from torch_geometric.nn import (
    BatchNorm,
    HeteroConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from hydragnn.utils.model import activation_function_selection, loss_function_selection
from hydragnn.utils.distributed import get_device
from hydragnn.models.Base import MLPNode


class HeteroBase(Module):
    """Base class for heterogeneous message passing models.

    This reuses HydraGNN's multi-head decoding logic while allowing hetero
    message passing via PyG HeteroConv.
    """

    def __init__(
        self,
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
        ilossweights_hyperp: int = 1,
        loss_weights: list = None,
        ilossweights_nll: int = 0,
        freeze_conv: bool = False,
        initial_bias=None,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_nodes: int = None,
        graph_pooling: str = "mean",
        use_graph_attr_conditioning: bool = False,
        graph_attr_conditioning_mode: str = "concat_node",
        hetero_pooling_mode: str = "sum",
        node_target_type: str = None,
        share_relation_weights: bool = False,
        node_input_dims: dict | None = None,
        metadata=None,
    ):
        super().__init__()

        if global_attn_engine:
            raise NotImplementedError(
                "HeteroBase does not yet support global attention. Set global_attn_engine=None."
            )

        self.device = get_device()
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.num_nodes = num_nodes
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()
        self.node_embedders = ModuleDict()
        self._node_input_dims = node_input_dims
        self.node_target_type = node_target_type
        self.share_relation_weights = share_relation_weights
        self._metadata = metadata
        self._initialized = False
        self._pending_node_conv_init = False
        self._node_conv_head_specs = []

        self.global_attn_engine = global_attn_engine
        self.global_attn_type = global_attn_type
        self.global_attn_heads = global_attn_heads

        self.heads_NN = ModuleList()
        self.config_heads = config_heads
        self.head_type = output_type
        self.head_dims = output_dim
        self.num_heads = len(self.head_dims)
        self.convs_node_hidden = ModuleDict({})
        self.batch_norms_node_hidden = ModuleDict({})
        self.convs_node_output = ModuleDict({})
        self.batch_norms_node_output = ModuleDict({})

        self.equivariance = equivariance
        self.activation_function = activation_function_selection(
            activation_function_type
        )

        self.use_graph_attr_conditioning = use_graph_attr_conditioning
        self.graph_attr_conditioning_mode = graph_attr_conditioning_mode.lower()
        if self.graph_attr_conditioning_mode not in (
            "film",
            "concat_node",
            "fuse_pool",
        ):
            raise ValueError(
                "graph_attr_conditioning_mode must be one of: 'film', 'concat_node', 'fuse_pool'."
            )

        # output variance for Gaussian negative log likelihood loss
        self.var_output = 0
        if loss_function_type == "GaussianNLLLoss":
            self.var_output = 1
        self.loss_function_type = loss_function_type
        self.loss_function = loss_function_selection(loss_function_type)
        self.ilossweights_nll = ilossweights_nll
        self.ilossweights_hyperp = ilossweights_hyperp

        if loss_weights is None:
            loss_weights = [1.0] * self.num_heads

        if self.ilossweights_hyperp * self.ilossweights_nll == 1:
            raise ValueError(
                "ilossweights_hyperp and ilossweights_nll cannot be both set to 1."
            )
        if self.ilossweights_hyperp == 1:
            if len(loss_weights) != self.num_heads:
                raise ValueError(
                    "Inconsistent number of loss weights and tasks: "
                    + str(len(loss_weights))
                    + " VS "
                    + str(self.num_heads)
                )
            else:
                self.loss_weights = loss_weights
            weightabssum = sum(abs(number) for number in self.loss_weights)
            self.loss_weights = [iw / weightabssum for iw in self.loss_weights]

        # Graph pooling policy
        pool_mode = graph_pooling.lower()
        if pool_mode == "sum":
            pool_mode = "add"
        pool_map = {
            "mean": (global_mean_pool, "mean"),
            "add": (global_add_pool, "sum"),
            "max": (global_max_pool, "max"),
        }
        if pool_mode not in pool_map:
            raise ValueError("Unsupported graph_pooling: " + graph_pooling)
        self.graph_pooling = pool_mode
        self.graph_pool_fn, self.graph_pool_reduction = pool_map[pool_mode]

        if hetero_pooling_mode not in ("sum", "mean"):
            raise ValueError("hetero_pooling_mode must be 'sum' or 'mean'.")
        self.hetero_pooling_mode = hetero_pooling_mode

        def _pool_graph_features(x_tensor, batch_tensor):
            if batch_tensor is None:
                if self.graph_pool_reduction == "mean":
                    return x_tensor.mean(dim=0, keepdim=True)
                if self.graph_pool_reduction == "max":
                    return x_tensor.max(dim=0, keepdim=True).values
                return x_tensor.sum(dim=0, keepdim=True)
            return self.graph_pool_fn(x_tensor, batch_tensor.to(x_tensor.device))

        self._pool_graph_features = _pool_graph_features

        self.freeze_conv = freeze_conv
        self.initial_bias = initial_bias

        # Graph conditioning modules (lazy)
        self.graph_conditioner = None
        self.graph_concat_projector = None
        self.graph_concat_projector_in_dim = None
        self.graph_pool_projector = None
        self.graph_pool_projector_in_dim = None

        self._multihead()
        if self.initial_bias is not None:
            self._set_bias()

        self.conv_checkpointing = False

        if self._metadata is not None:
            self._init_conv()

        if self._node_input_dims:
            self._init_node_embedders_from_dims(self._node_input_dims)

    def _init_node_embedders_from_dims(self, node_input_dims):
        for node_type, in_dim in node_input_dims.items():
            if node_type not in self.node_embedders:
                self.node_embedders[node_type] = Linear(int(in_dim), self.hidden_dim)
            if self.node_embedders[node_type].weight.device != self.device:
                self.node_embedders[node_type] = self.node_embedders[node_type].to(
                    self.device
                )

    def _ensure_node_embedders(self, x_dict):
        for node_type, x in x_dict.items():
            if node_type not in self.node_embedders:
                self.node_embedders[node_type] = Linear(x.size(-1), self.hidden_dim)
            if self.node_embedders[node_type].weight.device != x.device:
                self.node_embedders[node_type] = self.node_embedders[node_type].to(
                    x.device
                )

    def _maybe_init_metadata(self, data):
        if self._metadata is None:
            self._metadata = data.metadata()
        if not self._initialized:
            self._init_conv()
        if self._pending_node_conv_init:
            self._init_node_conv()
            self._finalize_node_conv_heads()
            self._pending_node_conv_init = False

    def _build_hetero_conv(self, input_dim: int, output_dim: int):
        conv_dict = {}
        shared_conv = None
        for edge_type in self._metadata[1]:
            if self.share_relation_weights:
                if shared_conv is None:
                    shared_conv = self.get_conv(input_dim, output_dim)
                conv_dict[edge_type] = shared_conv
            else:
                conv_dict[edge_type] = self.get_conv(input_dim, output_dim)
        return HeteroConv(conv_dict, aggr="sum")

    def _build_hetero_conv_node_head(self, input_dim: int, output_dim: int):
        conv_dict = {}
        shared_conv = None
        for edge_type in self._metadata[1]:
            if self.share_relation_weights:
                if shared_conv is None:
                    shared_conv = self.get_conv(input_dim, output_dim)
                conv_dict[edge_type] = shared_conv
            else:
                conv_dict[edge_type] = self.get_conv(input_dim, output_dim)
        return HeteroConv(conv_dict, aggr="sum")

    def _init_conv(self):
        self.graph_convs = ModuleList()
        self.feature_layers = ModuleList()
        for layer_idx in range(self.num_conv_layers):
            in_dim = self.hidden_dim if layer_idx > 0 else self.hidden_dim
            out_dim = self.hidden_dim
            self.graph_convs.append(self._build_hetero_conv(in_dim, out_dim))
            node_norms = ModuleDict({})
            for node_type in self._metadata[0]:
                node_norms[node_type] = BatchNorm(out_dim)
            self.feature_layers.append(node_norms)
        self._initialized = True

    def _init_node_conv(self):
        nodeconfiglist = self.config_heads["node"]
        assert (
            self.num_branches == len(nodeconfiglist) or self.num_branches == 1
        ), "assuming node head has the same branches as graph head, if any"
        for branchdict in nodeconfiglist:
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

            convs_node_hidden.append(
                self._build_hetero_conv_node_head(self.hidden_dim, hidden_dim_node[0])
            )
            bn_dict = ModuleDict({})
            for node_type in self._metadata[0]:
                bn_dict[node_type] = BatchNorm(hidden_dim_node[0])
            batch_norms_node_hidden.append(bn_dict)

            for ilayer in range(num_conv_layers_node - 1):
                convs_node_hidden.append(
                    self._build_hetero_conv_node_head(
                        hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1]
                    )
                )
                bn_dict = ModuleDict({})
                for node_type in self._metadata[0]:
                    bn_dict[node_type] = BatchNorm(hidden_dim_node[ilayer + 1])
                batch_norms_node_hidden.append(bn_dict)

            for ihead in node_feature_ind:
                convs_node_output.append(
                    self._build_hetero_conv_node_head(
                        hidden_dim_node[-1],
                        self.head_dims[ihead] * (1 + self.var_output),
                    )
                )
                bn_dict = ModuleDict({})
                for node_type in self._metadata[0]:
                    bn_dict[node_type] = BatchNorm(
                        self.head_dims[ihead] * (1 + self.var_output)
                    )
                batch_norms_node_output.append(bn_dict)

            self.convs_node_hidden[branchtype] = convs_node_hidden
            self.batch_norms_node_hidden[branchtype] = batch_norms_node_hidden
            self.convs_node_output[branchtype] = convs_node_output
            self.batch_norms_node_output[branchtype] = batch_norms_node_output

    def _freeze_conv(self):
        for module in [self.graph_convs, self.feature_layers]:
            for layer in module:
                for param in layer.parameters():
                    param.requires_grad = False

    def _set_bias(self):
        for head, type in zip(self.heads_NN, self.head_type):
            if type == "graph":
                head[-1].bias.data.fill_(self.initial_bias)

    def _multihead(self):
        self.graph_shared = ModuleDict({})
        dim_sharedlayers = 0
        self.num_branches = 1
        if "graph" in self.config_heads:
            self.num_branches = len(self.config_heads["graph"])
            for branchdict in self.config_heads["graph"]:
                denselayers = []
                dim_sharedlayers = branchdict["architecture"]["dim_sharedlayers"]
                denselayers.append(Linear(self.hidden_dim, dim_sharedlayers))
                denselayers.append(self.activation_function)
                for _ in range(branchdict["architecture"]["num_sharedlayers"] - 1):
                    denselayers.append(Linear(dim_sharedlayers, dim_sharedlayers))
                    denselayers.append(self.activation_function)
                self.graph_shared[branchdict["type"]] = Sequential(*denselayers)

        if "node" in self.config_heads:
            if self._metadata is None:
                self._pending_node_conv_init = True
            else:
                self._init_node_conv()

        inode_feature = 0
        for ihead in range(self.num_heads):
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
                            num_nodes=self.num_nodes
                            if node_NN_type == "mlp_per_node"
                            else None,
                        )
                    elif node_NN_type == "conv":
                        head_NN[branchtype] = ModuleList()
                        if self._metadata is None:
                            self._node_conv_head_specs.append(
                                (ihead, branchtype, inode_feature)
                            )
                            inode_feature += 1
                        else:
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
                            "HeteroBase only supports node heads with 'mlp', 'mlp_per_node', or 'conv'."
                        )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def _get_batch_dict(self, data, x_dict):
        batch_dict = None
        try:
            batch_dict = data.batch_dict
        except (AttributeError, KeyError):
            batch_dict = None
        if batch_dict is not None:
            return batch_dict
        batch_dict = {}
        for node_type, x in x_dict.items():
            batch_dict[node_type] = torch.zeros(
                x.size(0), device=x.device, dtype=torch.long
            )
        return batch_dict

    def _get_edge_attr_dict(self, data):
        if not getattr(self, "is_edge_model", False):
            return None
        edge_attr_dict = None
        try:
            edge_attr_dict = data.edge_attr_dict
        except (AttributeError, KeyError):
            edge_attr_dict = None
        return edge_attr_dict

    def _pool_hetero_graph_features(self, x_dict, batch_dict):
        pooled = []
        for node_type, x in x_dict.items():
            pooled.append(self._pool_graph_features(x, batch_dict[node_type]))
        if len(pooled) == 1:
            return pooled[0]
        if self.hetero_pooling_mode == "sum":
            return torch.stack(pooled, dim=0).sum(dim=0)
        return torch.stack(pooled, dim=0).mean(dim=0)

    def _finalize_node_conv_heads(self):
        if not self._node_conv_head_specs:
            return
        for head_index, branchtype, output_index in self._node_conv_head_specs:
            headloc = self.heads_NN[head_index]
            if branchtype not in headloc:
                headloc[branchtype] = ModuleList()
            for conv, batch_norm in zip(
                self.convs_node_hidden[branchtype],
                self.batch_norms_node_hidden[branchtype],
            ):
                headloc[branchtype].append(conv)
                headloc[branchtype].append(batch_norm)
            headloc[branchtype].append(self.convs_node_output[branchtype][output_index])
            headloc[branchtype].append(
                self.batch_norms_node_output[branchtype][output_index]
            )

    def _ensure_graph_conditioner(self, graph_attr_dim: int, device):
        if self.graph_conditioner is None:
            hidden = max(self.hidden_dim, graph_attr_dim)
            self.graph_conditioner = Sequential(
                Linear(graph_attr_dim, hidden),
                self.activation_function,
                Linear(hidden, 2 * self.hidden_dim),
            )
        if self.graph_conditioner[0].weight.device != device:
            self.graph_conditioner = self.graph_conditioner.to(device)

    def _ensure_graph_concat_projector(
        self, graph_attr_dim: int, channel_dim: int, device
    ):
        in_dim = channel_dim + graph_attr_dim
        if (self.graph_concat_projector is None) or (
            self.graph_concat_projector_in_dim != in_dim
        ):
            self.graph_concat_projector = Linear(in_dim, channel_dim)
            self.graph_concat_projector_in_dim = in_dim
        if self.graph_concat_projector.weight.device != device:
            self.graph_concat_projector = self.graph_concat_projector.to(device)

    def _ensure_graph_pool_projector(
        self, graph_attr_dim: int, channel_dim: int, device
    ):
        in_dim = channel_dim + graph_attr_dim
        if (self.graph_pool_projector is None) or (
            self.graph_pool_projector_in_dim != in_dim
        ):
            self.graph_pool_projector = Sequential(
                Linear(in_dim, channel_dim),
                self.activation_function,
                Linear(channel_dim, channel_dim),
            )
            self.graph_pool_projector_in_dim = in_dim
        if self.graph_pool_projector[0].weight.device != device:
            self.graph_pool_projector = self.graph_pool_projector.to(device)

    def _apply_graph_conditioning(self, inv_node_feat, batch, data):
        if not self.use_graph_attr_conditioning:
            return inv_node_feat

        if not hasattr(data, "graph_attr") or data.graph_attr is None:
            raise ValueError(
                "use_graph_attr_conditioning=True but data.graph_attr is missing."
            )

        graph_attr = data.graph_attr
        graph_attr = graph_attr.to(inv_node_feat.device).float()

        if batch is None:
            batch = torch.zeros(
                inv_node_feat.size(0), device=inv_node_feat.device, dtype=torch.long
            )

        num_graphs = int(batch.max().item() + 1)

        if graph_attr.dim() == 1:
            if graph_attr.numel() % num_graphs == 0:
                feat_dim = graph_attr.numel() // num_graphs
                graph_attr = graph_attr.view(num_graphs, feat_dim)
            else:
                raise ValueError(
                    f"One-dimensional graph_attr with numel={graph_attr.numel()} is not divisible by num_graphs={num_graphs}."
                )
        elif graph_attr.dim() == 2:
            if graph_attr.size(0) != num_graphs:
                raise ValueError(
                    f"graph_attr first dim {graph_attr.size(0)} does not match num_graphs={num_graphs}."
                )
        else:
            raise ValueError(
                f"Unsupported graph_attr ndim={graph_attr.dim()}; expected 1/2."
            )

        if self.graph_attr_conditioning_mode == "film":
            self._ensure_graph_conditioner(graph_attr.size(-1), inv_node_feat.device)

            scale_shift = self.graph_conditioner(graph_attr)
            scale, shift = scale_shift.split(self.hidden_dim, dim=-1)
            scale = torch.tanh(scale)

            channel_dim = inv_node_feat.size(-1)
            scale_b = scale[batch]
            shift_b = shift[batch]
            if channel_dim != self.hidden_dim:
                if channel_dim % self.hidden_dim != 0:
                    raise ValueError(
                        f"Graph conditioning expects channels divisible by hidden_dim (got {channel_dim} vs {self.hidden_dim})."
                    )
                factor = channel_dim // self.hidden_dim
                scale_b = scale_b.repeat_interleave(factor, dim=-1)
                shift_b = shift_b.repeat_interleave(factor, dim=-1)

            return inv_node_feat * (1 + scale_b) + shift_b

        if self.graph_attr_conditioning_mode == "concat_node":
            channel_dim = inv_node_feat.size(-1)
            self._ensure_graph_concat_projector(
                graph_attr_dim=graph_attr.size(-1),
                channel_dim=channel_dim,
                device=inv_node_feat.device,
            )
            graph_attr_b = graph_attr[batch]
            fused = torch.cat([inv_node_feat, graph_attr_b], dim=-1)
            return self.graph_concat_projector(fused)

        if self.graph_attr_conditioning_mode == "fuse_pool":
            return inv_node_feat

        raise ValueError(
            f"Unsupported graph_attr_conditioning_mode: {self.graph_attr_conditioning_mode}"
        )

    def _apply_graph_pool_conditioning(self, x_graph, data):
        if not self.use_graph_attr_conditioning:
            return x_graph
        if self.graph_attr_conditioning_mode != "fuse_pool":
            return x_graph
        if not hasattr(data, "graph_attr") or data.graph_attr is None:
            raise ValueError(
                "use_graph_attr_conditioning=True but data.graph_attr is missing."
            )

        graph_attr = data.graph_attr
        num_graphs = x_graph.size(0)

        if graph_attr.dim() == 1:
            if graph_attr.numel() % num_graphs == 0:
                feat_dim = graph_attr.numel() // num_graphs
                graph_attr = graph_attr.view(num_graphs, feat_dim)
            else:
                raise ValueError(
                    f"One-dimensional graph attribute with graph_attr.numel()={graph_attr.numel()} is not divisible by num_graphs={num_graphs}."
                )
        elif graph_attr.dim() == 2:
            if graph_attr.size(0) != num_graphs:
                raise ValueError(
                    f"graph_attr batch size does not match pooled graph embeddings: graph_attr={tuple(graph_attr.size())}, num_graphs={num_graphs}"
                )
        else:
            raise ValueError(
                f"Unsupported graph_attr ndim={graph_attr.dim()}; expected 1/2."
            )

        graph_attr = graph_attr.to(x_graph.device).float()

        self._ensure_graph_pool_projector(
            graph_attr_dim=graph_attr.size(-1),
            channel_dim=x_graph.size(-1),
            device=x_graph.device,
        )

        if graph_attr.size(0) != num_graphs:
            raise ValueError(
                f"graph_attr batch size does not match pooled graph embeddings: "
                f"graph_attr={tuple(graph_attr.size())}, x_graph={tuple(x_graph.size())}, num_graphs={num_graphs}"
            )

        fused = torch.cat([x_graph, graph_attr], dim=-1)
        return self.graph_pool_projector(fused)

    def forward(self, data):
        self._maybe_init_metadata(data)

        device = next(self.parameters()).device
        if hasattr(data, "to"):
            data = data.to(device)

        if hasattr(data, "node_types"):
            for node_type in data.node_types:
                store = data[node_type]
                if hasattr(store, "x") and store.x is not None:
                    store.x = store.x.to(device)
        if hasattr(data, "edge_types"):
            for edge_type in data.edge_types:
                store = data[edge_type]
                if hasattr(store, "edge_index") and store.edge_index is not None:
                    store.edge_index = store.edge_index.to(device)
                if hasattr(store, "edge_attr") and store.edge_attr is not None:
                    store.edge_attr = store.edge_attr.to(device)

        x_dict = {node_type: x.to(device) for node_type, x in data.x_dict.items()}
        self._ensure_node_embedders(x_dict)
        x_dict = {
            node_type: self.node_embedders[node_type](x.float())
            for node_type, x in x_dict.items()
        }

        batch_dict = self._get_batch_dict(data, x_dict)
        edge_attr_dict = self._get_edge_attr_dict(data)

        for conv, node_norms in zip(self.graph_convs, self.feature_layers):
            if edge_attr_dict is None:
                x_dict = conv(x_dict, data.edge_index_dict)
            else:
                x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            for node_type, x in x_dict.items():
                x = self._apply_graph_conditioning(x, batch_dict[node_type], data)
                x = node_norms[node_type](x)
                x = self.activation_function(x)
                x_dict[node_type] = x

        x_graph = self._pool_hetero_graph_features(x_dict, batch_dict)
        x_graph = x_graph.to(device)
        x_graph = self._apply_graph_pool_conditioning(x_graph, data)

        # Prepare dataset_name for multi-branch heads
        if not hasattr(data, "dataset_name"):
            num_graphs = x_graph.size(0)
            data.dataset_name = torch.zeros(
                (num_graphs, 1), device=x_graph.device, dtype=torch.long
            )
        else:
            data.dataset_name = data.dataset_name.to(x_graph.device)

        outputs = []
        outputs_var = []

        datasetIDs = data.dataset_name.unique()

        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                head = torch.zeros(
                    (len(data.dataset_name), head_dim), device=x_graph.device
                )
                headvar = torch.zeros(
                    (len(data.dataset_name), head_dim * self.var_output),
                    device=x_graph.device,
                )
                if self.num_branches == 1:
                    head_device = next(
                        self.graph_shared["branch-0"].parameters()
                    ).device
                    x_graph = x_graph.to(head_device)
                    x_graph_head = self.graph_shared["branch-0"](x_graph)
                    output_head = headloc["branch-0"](x_graph_head)
                    head = output_head[:, :head_dim]
                    headvar = output_head[:, head_dim:] ** 2
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask = mask[:, 0]
                        branchtype = f"branch-{ID.item()}"
                        head_device = next(
                            self.graph_shared[branchtype].parameters()
                        ).device
                        x_graph = x_graph.to(head_device)
                        x_graph_head = self.graph_shared[branchtype](x_graph[mask, :])
                        output_head = headloc[branchtype](x_graph_head)
                        head[mask] = output_head[:, :head_dim]
                        headvar[mask] = output_head[:, head_dim:] ** 2
                outputs.append(head)
                outputs_var.append(headvar)
            else:
                if self.node_target_type is None:
                    self.node_target_type = self._metadata[0][0]
                x_node = x_dict[self.node_target_type]
                batch_node = batch_dict[self.node_target_type]

                try:
                    head_device = next(headloc.parameters()).device
                except StopIteration:
                    head_device = x_node.device
                if x_node.device != head_device:
                    x_node = x_node.to(head_device)
                if batch_node.device != head_device:
                    batch_node = batch_node.to(head_device)

                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                if node_NN_type not in ("mlp", "mlp_per_node", "conv"):
                    raise ValueError(
                        "HeteroBase only supports node heads with 'mlp', 'mlp_per_node', or 'conv'."
                    )

                head = torch.zeros((x_node.shape[0], head_dim), device=x_node.device)
                headvar = torch.zeros(
                    (x_node.shape[0], head_dim * self.var_output), device=x_node.device
                )

                if node_NN_type == "conv":
                    if self.num_branches != 1:
                        raise NotImplementedError(
                            "conv-based node heads with multiple branches are not supported yet for hetero models."
                        )
                    branchtype = "branch-0"
                    x_dict_node = x_dict
                    for conv, batch_norm in zip(
                        headloc[branchtype][0::2], headloc[branchtype][1::2]
                    ):
                        if edge_attr_dict is None:
                            x_dict_node = conv(x_dict_node, data.edge_index_dict)
                        else:
                            x_dict_node = conv(
                                x_dict_node, data.edge_index_dict, edge_attr_dict
                            )
                        for node_type, x in x_dict_node.items():
                            x = batch_norm[node_type](x)
                            x = self.activation_function(x)
                            x_dict_node[node_type] = x
                    x_node_out = x_dict_node[self.node_target_type]
                    head = x_node_out[:, :head_dim]
                    headvar = x_node_out[:, head_dim:] ** 2
                else:
                    if self.num_branches == 1:
                        branchtype = "branch-0"
                        x_node_out = headloc[branchtype](x=x_node, batch=batch_node)
                        head = x_node_out[:, :head_dim]
                        headvar = x_node_out[:, head_dim:] ** 2
                    else:
                        unique, node_counts = torch.unique_consecutive(
                            batch_node, return_counts=True
                        )
                        for ID in datasetIDs:
                            mask = data.dataset_name == ID
                            mask_nodes = torch.repeat_interleave(mask, node_counts)
                            branchtype = f"branch-{ID.item()}"
                            x_node_out = headloc[branchtype](
                                x=x_node[mask_nodes, :], batch=batch_node[mask_nodes]
                            )
                            head[mask_nodes] = x_node_out[:, :head_dim]
                            headvar[mask_nodes] = x_node_out[:, head_dim:] ** 2

                outputs.append(head)
                outputs_var.append(headvar)

        if self.var_output:
            return outputs, outputs_var
        return outputs

    def loss(self, pred, value, head_index):
        var = None
        if self.var_output:
            var = pred[1]
            pred = pred[0]
        if self.ilossweights_nll == 1:
            raise ValueError("loss_nll() not ready yet")
        if self.ilossweights_hyperp == 1:
            return self.loss_hpweighted(pred, value, head_index, var=var)
        raise ValueError("Unsupported loss weighting configuration")

    def loss_hpweighted(self, pred, value, head_index, var=None):
        tot_loss = 0
        tasks_loss = []
        for ihead in range(self.num_heads):
            head_pre = pred[ihead]
            pred_shape = head_pre.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)
            head_val = head_val.to(head_pre.device)
            if var is None:
                tot_loss += (
                    self.loss_function(head_pre, head_val) * self.loss_weights[ihead]
                )
                tasks_loss.append(self.loss_function(head_pre, head_val))
            else:
                head_var = var[ihead]
                tot_loss += (
                    self.loss_function(head_pre, head_val, head_var)
                    * self.loss_weights[ihead]
                )
                tasks_loss.append(self.loss_function(head_pre, head_val, head_var))

        return tot_loss, tasks_loss
