import torch
from torch.nn import ModuleList, Sequential, ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.25

    def _multihead(
        self, output_dim: list, num_nodes: int, output_type: list, config_heads: {}
    ):
        ############multiple heads/taks################
        # get number of heads from input
        ##One head represent one variable
        ##Head can have different sizes, head_dims;
        ###e.g., 1 for energy, 32 for charge density, 32*3 for magnetic moments
        self.num_heads = len(output_dim)
        self.head_dims = output_dim
        self.head_type = output_type
        self.num_nodes = num_nodes

        # shared dense layers for heads with graph level output
        dim_sharedlayers = 0
        if "output_graph" in config_heads:
            denselayers = []
            dim_sharedlayers = config_heads["output_graph"]["dim_sharedlayers"]
            denselayers.append(ReLU())
            denselayers.append(Linear(self.hidden_dim, dim_sharedlayers))
            for ishare in range(config_heads["output_graph"]["num_sharedlayers"] - 1):
                denselayers.append(Linear(dim_sharedlayers, dim_sharedlayers))
                denselayers.append(ReLU())
            self.graph_shared = Sequential(*denselayers)

        self.heads = ModuleList()
        for ihead in range(self.num_heads):
            # mlp for each head output
            if self.head_type[ihead] == "graph":
                num_head_hidden = config_heads["output_graph"]["num_headlayers"]
                dim_head_hidden = config_heads["output_graph"]["dim_headlayers"]
                denselayers = []
                denselayers.append(Linear(dim_sharedlayers, dim_head_hidden[0]))
                denselayers.append(ReLU())
                for ilayer in range(num_head_hidden - 1):
                    denselayers.append(
                        Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                    )
                    denselayers.append(ReLU())
                denselayers.append(Linear(dim_head_hidden[-1], self.head_dims[ihead]))
                mlp = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                mlp = ModuleList()
                for inode in range(self.num_nodes):
                    num_head_hidden = config_heads["output_node"]["num_headlayers"]
                    dim_head_hidden = config_heads["output_node"]["dim_headlayers"]
                    denselayers = []
                    denselayers.append(Linear(self.hidden_dim, dim_head_hidden[0]))
                    denselayers.append(ReLU())
                    for ilayer in range(num_head_hidden - 1):
                        denselayers.append(
                            Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                        )
                        denselayers.append(ReLU())
                    denselayers.append(Linear(dim_head_hidden[-1], 1))
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
        batch_size = batch.max() + 1
        num_features = x.shape[1]
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes), dtype=x.dtype, device=x.device
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
        outputs = []
        for headloc, type_head in zip(self.heads, self.head_type):
            if type_head == "graph":
                x_graph = self.graph_shared(x_graph)
                outputs.append(headloc(x_graph))
            else:
                node_output = []
                for inode in range(self.num_nodes):
                    node_output.append(headloc[inode](x_nodes[:, :, inode]))
                outputs.append(torch.cat(node_output, dim=1))
        return torch.cat(outputs, dim=1)

    def loss(self, pred, value):
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return F.l1_loss(pred, value)

    def loss_rmse(self, pred, value):
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return torch.sqrt(F.mse_loss(pred, value))

    def __str__(self):
        return "Base"
