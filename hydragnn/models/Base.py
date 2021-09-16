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
from torch.nn import ModuleList, Sequential, ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import GaussianNLLLoss
import sys
from .mlp_node_feature import mlp_node_feature


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.25

    def _multihead(
        self,
        num_nodes: int,
        ilossweights_hyperp: int,
        loss_weights: list,
        ilossweights_nll: int,
    ):

        ############multiple heads/taks################
        # get number of heads from input
        ##One head represent one variable
        ##Head can have different sizes, head_dims;
        ###e.g., 1 for energy, 32 for charge density, 32 or 32*3 for magnetic moments
        self.num_heads = len(self.head_dims)
        self.num_nodes = num_nodes

        # self.head_dim_sum = sum(self.head_dims) #need to be improved due to num_nodes

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

        self.heads = ModuleList()
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
                        self.head_dims[ihead] + ilossweights_nll * 1,
                    )
                )
                head_NN = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                self.node_NN_type = self.config_heads["node"]["type"]
                head_NN = ModuleList()
                if self.node_NN_type == "mlp":
                    head_NN = mlp_node_feature(
                        self.hidden_dim,
                        self.head_dims[ihead],
                        self.num_nodes,
                        self.hidden_dim_node,
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
                        + "; currently only support 'mlp' (for constant num_nodes) or 'conv'"
                    )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads.append(head_NN)

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        ### encoder part ####
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x=x, edge_index=edge_index)))
        #### multi-head decoder part####
        # shared dense layers for graph level output
        x_graph = global_mean_pool(x, batch)
        outputs = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads, self.head_type
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
        # weights for difficult tasks as hyper-parameters
        tot_loss = 0
        tasks_rmseloss = []
        tasks_nodes = []
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
            tasks_nodes.append(torch.sqrt(F.mse_loss(head_pre, head_val)))
            # loss of summation across nodes/atoms
            tasks_rmseloss.append(
                torch.sqrt(F.mse_loss(torch.sum(head_pre, 1), torch.sum(head_val, 1)))
            )

        return tot_loss, tasks_rmseloss, tasks_nodes

    def __str__(self):
        return "Base"
