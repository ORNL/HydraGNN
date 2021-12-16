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

import torch
from torch.nn import ModuleList, Sequential, ReLU, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch.nn import GaussianNLLLoss
import sys


class Base(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: list,
        output_type: list,
        config_heads: {},
        ilossweights_hyperp: int,
        loss_weights: list,
        ilossweights_nll: int,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_nodes: int = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.num_nodes = num_nodes
        ##One head represent one variable
        ##Head can have different sizes, head_dims
        self.heads_NN = ModuleList()
        self.config_heads = config_heads
        self.head_type = output_type
        self.head_dims = output_dim
        self.num_heads = len(self.head_dims)
        ##convolutional layers for node level predictions
        self.convs_node_hidden = ModuleList()
        self.batch_norms_node_hidden = ModuleList()
        self.convs_node_output = ModuleList()
        self.batch_norms_node_output = ModuleList()

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

        self._init_conv()
        self._init_node_conv()
        self._multihead()

    def _init_conv(self):
        self.convs.append(self.get_conv(self.input_dim, self.hidden_dim))
        self.batch_norms.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            conv = self.get_conv(self.hidden_dim, self.hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

    def _init_node_conv(self):
        # *******convolutional layers for node level predictions*******#
        # two ways to implement node features from here:
        # 1. one graph for all node features
        # 2. one graph for one node features (currently implemented)
        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return
        self.num_conv_layers_node = self.config_heads["node"]["num_headlayers"]
        self.hidden_dim_node = self.config_heads["node"]["dim_headlayers"]
        # In this part, each head has same number of convolutional layers, but can have different output dimension
        self.convs_node_hidden.append(
            self.get_conv(self.hidden_dim, self.hidden_dim_node[0])
        )
        self.batch_norms_node_hidden.append(BatchNorm(self.hidden_dim_node[0]))
        for ilayer in range(self.num_conv_layers_node - 1):
            self.convs_node_hidden.append(
                self.get_conv(
                    self.hidden_dim_node[ilayer], self.hidden_dim_node[ilayer + 1]
                )
            )
            self.batch_norms_node_hidden.append(
                BatchNorm(self.hidden_dim_node[ilayer + 1])
            )
        for ihead in node_feature_ind:
            self.convs_node_output.append(
                self.get_conv(self.hidden_dim_node[-1], self.head_dims[ihead])
            )
            self.batch_norms_node_output.append(BatchNorm(self.head_dims[ihead]))

    def _multihead(self):
        ############multiple heads/taks################
        # shared dense layers for heads with graph level output
        dim_sharedlayers = 0
        if "graph" in self.config_heads:
            denselayers = []
            dim_sharedlayers = self.config_heads["graph"]["dim_sharedlayers"]
            denselayers.append(ReLU())
            denselayers.append(Linear(self.hidden_dim, dim_sharedlayers))
            for ishare in range(self.config_heads["graph"]["num_sharedlayers"] - 1):
                denselayers.append(Linear(dim_sharedlayers, dim_sharedlayers))
                denselayers.append(ReLU())
            self.graph_shared = Sequential(*denselayers)

        inode_feature = 0
        for ihead in range(self.num_heads):
            # mlp for each head output
            if self.head_type[ihead] == "graph":
                num_head_hidden = self.config_heads["graph"]["num_headlayers"]
                dim_head_hidden = self.config_heads["graph"]["dim_headlayers"]
                denselayers = []
                denselayers.append(Linear(dim_sharedlayers, dim_head_hidden[0]))
                denselayers.append(ReLU())
                for ilayer in range(num_head_hidden - 1):
                    denselayers.append(
                        Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                    )
                    denselayers.append(ReLU())
                denselayers.append(
                    Linear(
                        dim_head_hidden[-1],
                        self.head_dims[ihead] + self.ilossweights_nll * 1,
                    )
                )
                head_NN = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                self.node_NN_type = self.config_heads["node"]["type"]
                head_NN = ModuleList()
                if self.node_NN_type == "mlp" or self.node_NN_type == "mlp_per_node":
                    assert (
                        self.num_nodes is not None
                    ), "num_nodes must be positive integer for MLP"
                    # """if different graphs in the dataset have different size, one MLP is shared across all nodes """
                    head_NN = MLPNode(
                        self.hidden_dim,
                        self.head_dims[ihead],
                        self.num_nodes,
                        self.hidden_dim_node,
                        self.config_heads["node"]["type"],
                    )
                elif self.node_NN_type == "conv":
                    for conv, batch_norm in zip(
                        self.convs_node_hidden, self.batch_norms_node_hidden
                    ):
                        head_NN.append(conv)
                        head_NN.append(batch_norm)
                    head_NN.append(self.convs_node_output[inode_feature])
                    head_NN.append(self.batch_norms_node_output[inode_feature])
                    inode_feature += 1
                else:
                    raise ValueError(
                        "Unknown head NN structure for node features"
                        + self.node_NN_type
                        + "; currently only support 'mlp', 'mlp_per_node' or 'conv' (can be set with config['NeuralNetwork']['Architecture']['output_heads']['node']['type'], e.g., ./examples/ci_multihead.json)"
                    )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        use_edge_attr = False
        if (data.edge_attr is not None) and (self.use_edge_attr):
            use_edge_attr = True

        ### encoder part ####
        if use_edge_attr:
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                c = conv(x=x, edge_index=edge_index, edge_attr=data.edge_attr)
                x = F.relu(batch_norm(c))
        else:
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                c = conv(x=x, edge_index=edge_index)
                x = F.relu(batch_norm(c))

        #### multi-head decoder part####
        # shared dense layers for graph level output
        x_graph = global_mean_pool(x, batch.to(x.device))
        outputs = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                x_graph_head = x_graph.clone()
                x_graph_head = self.graph_shared(x_graph_head)
                outputs.append(headloc(x_graph_head))
            else:
                x_node = x.clone()
                if self.node_NN_type == "conv":
                    for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                        x_node = F.relu(
                            batch_norm(conv(x=x_node, edge_index=edge_index))
                        )
                else:
                    x_node = headloc(x=x_node, batch=batch)
                outputs.append(x_node)
        return outputs

    def loss_rmse(self, pred, value, head_index):
        if self.ilossweights_nll == 1:
            return self.loss_nll(pred, value, head_index)
        elif self.ilossweights_hyperp == 1:
            return self.loss_hpweighted(pred, value, head_index)

    def loss_nll(self, pred, value, head_index):
        # negative log likelihood loss
        # uncertainty to weigh losses in https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        # fixme
        raise ValueError("loss_nll() not ready yet")
        nll_loss = 0
        tasks_rmseloss = []
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
            tasks_rmseloss.append(torch.sqrt(F.mse_loss(head_pre, head_val)))

        return nll_loss, tasks_rmseloss, []

    def loss_hpweighted(self, pred, value, head_index):
        # weights for different tasks as hyper-parameters
        tot_loss = 0
        tasks_rmse = []
        for ihead in range(self.num_heads):
            head_pre = pred[ihead]
            pred_shape = head_pre.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)

            tot_loss += (
                torch.sqrt(F.mse_loss(head_pre, head_val)) * self.loss_weights[ihead]
            )
            tasks_rmse.append(torch.sqrt(F.mse_loss(head_pre, head_val)))

        return tot_loss, tasks_rmse

    def __str__(self):
        return "Base"


class MLPNode(Module):
    def __init__(self, input_dim, output_dim, num_nodes, hidden_dim_node, node_type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.node_type = node_type

        self.mlp = ModuleList()
        for inode in range(self.num_nodes):
            denselayers = []
            denselayers.append(Linear(self.input_dim, hidden_dim_node[0]))
            denselayers.append(ReLU())
            for ilayer in range(len(hidden_dim_node) - 1):
                denselayers.append(
                    Linear(hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1])
                )
                denselayers.append(ReLU())
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
        outs = torch.zeros(
            (x.shape[0], self.output_dim),
            dtype=x.dtype,
            device=x.device,
        )
        if self.node_type == "mlp":
            outs = self.mlp[0](x)
        else:
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"
