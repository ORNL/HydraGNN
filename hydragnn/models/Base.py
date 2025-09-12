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
from torch.nn import ModuleList, Sequential, ReLU, Linear, Module, ModuleDict
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch_geometric.nn import Sequential as PyGSequential
from torch.utils.checkpoint import checkpoint
import torch_scatter
from hydragnn.utils.model import activation_function_selection, loss_function_selection
import sys
import pdb
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print.print_utils import print_master
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths
from hydragnn.globalAtt.gps import GPSConv
import hydragnn.utils.profiling_and_tracing.tracer as tr

import inspect


class Base(Module):
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
        ilossweights_hyperp: int = 1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        loss_weights: list = [1.0, 1.0, 1.0],  # weights for losses of different tasks
        ilossweights_nll: int = 0,  # if =1, using the scalar uncertainty as weights, as in paper# https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        freeze_conv=False,
        initial_bias=None,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_nodes: int = None,
    ):
        super().__init__()
        self.device = get_device()
        self.input_args = input_args
        self.conv_args = conv_args
        self.global_attn_engine = global_attn_engine
        self.global_attn_type = global_attn_type
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.global_attn_heads = global_attn_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.global_attn_dropout = dropout
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

        # output variance for Gaussian negative log likelihood loss
        self.var_output = 0
        if loss_function_type == "GaussianNLLLoss":
            self.var_output = 1
        self.loss_function_type = loss_function_type

        self.loss_function = loss_function_selection(loss_function_type)
        self.ilossweights_nll = ilossweights_nll
        self.ilossweights_hyperp = ilossweights_hyperp
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

        # Option to only train final property layers.
        self.freeze_conv = freeze_conv
        # Option to set initially large output bias (UQ).
        self.initial_bias = initial_bias

        # Specify global attention usage: specify input embedding dims and edge embedding dims;
        # if model can handle edge features, enforce use of relative edge encodings
        if self.global_attn_engine:
            self.use_global_attn = True
            self.embed_dim = (
                self.edge_embed_dim
            ) = hidden_dim  # ensure that all input to gps have the same dimensionality
            if self.is_edge_model:
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
            False  # provision to decouple encodings from globalAtt later
        )

        # Specify learnable embeddings
        if self.use_global_attn or self.use_encodings:
            self.pos_emb = Linear(self.pe_dim, self.hidden_dim, bias=False)
            if self.input_dim:
                self.node_emb = Linear(self.input_dim, self.hidden_dim, bias=False)
                self.node_lin = Linear(2 * self.hidden_dim, self.hidden_dim, bias=False)
            if self.is_edge_model:
                self.rel_pos_emb = Linear(self.pe_dim, self.hidden_dim, bias=False)
                if self.use_edge_attr:
                    self.edge_emb = Linear(self.edge_dim, self.hidden_dim, bias=False)
                    self.edge_lin = Linear(
                        2 * self.hidden_dim, self.hidden_dim, bias=False
                    )

        self._init_conv()
        if self.freeze_conv:
            self._freeze_conv()
        self._multihead()
        if self.initial_bias is not None:
            self._set_bias()

        self.conv_checkpointing = False

    def _apply_global_attn(self, mpnn):
        # choose to use global attention or mpnn
        if self.use_global_attn:
            # specify global attention engine; use this to support more engines in future
            if self.global_attn_engine == "GPS":
                return GPSConv(
                    channels=self.hidden_dim,
                    conv=mpnn,
                    heads=self.global_attn_heads,
                    dropout=self.global_attn_dropout,
                    attn_type=self.global_attn_type,
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

    def _embedding(self, data):
        if not hasattr(data, "edge_shifts"):
            data.edge_shifts = torch.zeros(
                (data.edge_index.size(1), 3), device=data.edge_index.device
            )
        conv_args = {"edge_index": data.edge_index.to(torch.long)}
        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_attr})

        if self.use_global_attn:
            # encode node positional embeddings
            x = self.pos_emb(data.pe)
            # if node features are available, generate mebeddings, concatenate with positional embeddings and map to hidden dim
            if self.input_dim:
                x = torch.cat((self.node_emb(data.x.float()), x), 1)
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
            return data.x, data.pos, conv_args

    def _freeze_conv(self):
        for module in [self.graph_convs, self.feature_layers]:
            for layer in module:
                for param in layer.parameters():
                    param.requires_grad = False

    def _set_bias(self):
        for head, type in zip(self.heads_NN, self.head_type):
            # FIXME: we only currently enable this for graph outputs.
            if type == "graph":
                # Set the bias of the last linear layer to a large value (UQ)
                head[-1].bias.data.fill_(self.initial_bias)

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
                        assert (
                            self.num_nodes is not None
                        ), "num_nodes must be positive integer for MLP"
                        # """if different graphs in the datasets have different size, one MLP is shared across all nodes """
                        head_NN[branchtype] = MLPNode(
                            self.hidden_dim,
                            self.head_dims[ihead] * (1 + self.var_output),
                            self.num_mlp,
                            hidden_dim_node,
                            node_NN_type,
                            self.activation_function,
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

    def enable_conv_checkpointing(self):
        print_master("Enabling checkpointing")
        self.conv_checkpointing = True

    def forward(self, data):
        ### encoder part ####
        tr.start("enc_forward")
        inv_node_feat, equiv_node_feat, conv_args = self._embedding(data)

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
            inv_node_feat = self.activation_function(feat_layer(inv_node_feat))

        x = inv_node_feat
        tr.stop("enc_forward")
        tr.start("branch_forward")
        #### multi-head decoder part####
        # shared dense layers for graph level output
        if data.batch is None:
            x_graph = x.mean(dim=0, keepdim=True)
            # individual samplers
            data.batch = data.x * 0
        else:
            x_graph = global_mean_pool(x, data.batch.to(x.device))
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
                head = torch.zeros((len(data.dataset_name), head_dim), device=x.device)
                headvar = torch.zeros(
                    (len(data.dataset_name), head_dim * self.var_output),
                    device=x.device,
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
                        head[mask] = output_head[:, :head_dim]
                        headvar[mask] = output_head[:, head_dim:] ** 2
                outputs.append(head)
                outputs_var.append(headvar)
            else:
                # assuming all node types are the same
                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                head = torch.zeros((x.shape[0], head_dim), device=x.device)
                headvar = torch.zeros(
                    (x.shape[0], head_dim * self.var_output), device=x.device
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
                        head[mask_nodes] = x_node[:, :head_dim]
                        headvar[mask_nodes] = x_node[:, head_dim:] ** 2
                outputs.append(head)
                outputs_var.append(headvar)
        tr.stop("branch_forward")
        if self.var_output:
            return outputs, outputs_var
        return outputs

    def loss(self, pred, value, head_index):
        var = None
        if self.var_output:
            var = pred[1]
            pred = pred[0]
        if self.ilossweights_nll == 1:
            return self.loss_nll(pred, value, head_index, var=var)
        elif self.ilossweights_hyperp == 1:
            return self.loss_hpweighted(pred, value, head_index, var=var)

    def loss_nll(self, pred, value, head_index, var=None):
        # negative log likelihood loss
        # uncertainty to weigh losses in https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        # fixme: Pei said that right now this is never used
        raise ValueError("loss_nll() not ready yet")
        nll_loss = 0
        tasks_mseloss = []
        loss = GaussianNLLLoss()
        for ihead in range(self.num_heads):
            head_pre = pred[ihead][:, :-1]
            pred_shape = head_pre.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)
            head_var = torch.exp(pred[ihead][:, -1])
            nll_loss += loss(head_pre, head_val, head_var)
            tasks_mseloss.append(F.mse_loss(head_pre, head_val))

        return nll_loss, tasks_mseloss, []

    def loss_hpweighted(self, pred, value, head_index, var=None):
        # weights for different tasks as hyper-parameters
        tot_loss = 0
        tasks_loss = []
        for ihead in range(self.num_heads):
            head_pre = pred[ihead]
            pred_shape = head_pre.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)
            if var is None:
                assert (
                    self.loss_function_type != "GaussianNLLLoss"
                ), "Expecting var for GaussianNLLLoss, but got None"
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

    def __str__(self):
        return "Base"


class MLPNode(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_mlp,
        hidden_dim_node,
        node_type,
        activation_function,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_type = node_type
        self.num_mlp = num_mlp
        self.activation_function = activation_function

        self.mlp = ModuleList()
        for _ in range(self.num_mlp):
            denselayers = []
            denselayers.append(Linear(self.input_dim, hidden_dim_node[0]))
            denselayers.append(self.activation_function)
            for ilayer in range(len(hidden_dim_node) - 1):
                denselayers.append(
                    Linear(hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1])
                )
                denselayers.append(self.activation_function)
            denselayers.append(Linear(hidden_dim_node[-1], output_dim))
            self.mlp.append(Sequential(*denselayers))

    def node_features_reshape(self, x, batch):
        """reshape x from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = x.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=x.dtype,
            device=x.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = x[inode_index, :]
        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        if self.node_type == "mlp":
            outs = self.mlp[0](x)
        else:
            outs = torch.zeros(
                (x.shape[0], self.output_dim),
                dtype=x.dtype,
                device=x.device,
            )
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"
