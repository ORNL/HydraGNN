import torch
from torch.nn import ModuleList, Sequential, ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import GaussianNLLLoss
import sys


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.25

    def _multihead(
        self,
        output_dim: list,
        num_nodes: int,
        output_type: list,
        config_heads: {},
        ilossweights_hyperp: int,
        loss_weights: list,
        ilossweights_nll: int,
    ):

        ############multiple heads/taks################
        # get number of heads from input
        ##One head represent one variable
        ##Head can have different sizes, head_dims;
        ###e.g., 1 for energy, 32 for charge density, 32 or 32*3 for magnetic moments
        self.num_heads = len(output_dim)
        self.head_dims = output_dim
        self.head_type = output_type
        self.num_nodes = num_nodes
        self.head_dim_sum = sum(self.head_dims)

        # shared dense layers for heads with graph level output
        dim_sharedlayers = 0
        if "graph" in config_heads:
            denselayers = []
            dim_sharedlayers = config_heads["graph"]["dim_sharedlayers"]
            denselayers.append(ReLU())
            denselayers.append(Linear(self.hidden_dim, dim_sharedlayers))
            for ishare in range(config_heads["graph"]["num_sharedlayers"] - 1):
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
        for ihead in range(self.num_heads):
            # mlp for each head output
            if self.head_type[ihead] == "graph":
                num_head_hidden = config_heads["graph"]["num_headlayers"]
                dim_head_hidden = config_heads["graph"]["dim_headlayers"]
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
                mlp = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                mlp = ModuleList()
                for inode in range(self.num_nodes):
                    num_head_hidden = config_heads["node"]["num_headlayers"]
                    dim_head_hidden = config_heads["node"]["dim_headlayers"]
                    denselayers = []
                    denselayers.append(Linear(self.hidden_dim, dim_head_hidden[0]))
                    denselayers.append(ReLU())
                    for ilayer in range(num_head_hidden - 1):
                        denselayers.append(
                            Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                        )
                        denselayers.append(ReLU())
                    denselayers.append(
                        Linear(dim_head_hidden[-1], 1 + ilossweights_nll * 1)
                    )
                    mlp.append(Sequential(*denselayers))
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads.append(mlp)

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
        # node features for node level output
        x_nodes = self.node_features_reshape(x, batch)
        outputs = torch.zeros(
            (x_nodes.shape[0], self.head_dim_sum), dtype=x.dtype, device=x.device
        )
        istart = 0
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads, self.head_type
        ):
            if type_head == "graph":
                x_graph = self.graph_shared(x_graph)
                outputs[:, istart : (istart + head_dim)] = headloc(x_graph)
            else:
                for inode in range(self.num_nodes):
                    outputs[:, istart + inode] = headloc[inode](
                        x_nodes[:, :, inode]
                    ).squeeze()
            istart += head_dim
        return outputs

    def loss(self, pred, value):
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return F.l1_loss(pred, value)

    def loss_rmse(self, pred, value):

        if self.ilossweights_nll == 1:
            return self.loss_nll(pred, value)
        elif self.ilossweights_hyperp == 1:
            return self.loss_hpweighted(pred, value)

        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return torch.sqrt(F.mse_loss(pred, value)), [], []

    def loss_nll(self, pred, value):
        # negative log likelihood loss
        # uncertainty to weigh losses in https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape[0] != value_shape[0]:
            value = torch.reshape(value, (pred_shape[0], -1))
            if value.shape[1] != pred.shape[1] - self.num_heads:
                raise ValueError(
                    "expected feature dims="
                    + str(pred.shape[1] - self.num_heads)
                    + "; actual="
                    + str(value.shape[1])
                )

        nll_loss = 0
        tasks_rmseloss = []
        loss = GaussianNLLLoss()
        for ihead in range(self.num_heads):
            isum = sum(self.head_dims[: ihead + 1])
            ivar = isum + (ihead + 1) * 1 - 1

            head_var = torch.exp(pred[:, ivar])
            head_pre = pred[:, ivar - self.head_dims[ihead] : ivar]
            head_val = value[:, isum - self.head_dims[ihead] : isum]
            nll_loss += loss(head_pre, head_val, head_var)
            tasks_rmseloss.append(torch.sqrt(F.mse_loss(head_pre, head_val)))

        return nll_loss, tasks_rmseloss, []

    def loss_hpweighted(self, pred, value):
        # weights for difficult tasks as hyper-parameters
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)

        tot_loss = 0
        tasks_rmseloss = []
        tasks_nodes = []
        for ihead in range(self.num_heads):
            isum = sum(self.head_dims[: ihead + 1])

            head_pre = pred[:, isum - self.head_dims[ihead] : isum]
            head_val = value[:, isum - self.head_dims[ihead] : isum]

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
