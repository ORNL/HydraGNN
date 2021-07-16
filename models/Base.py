import torch
from torch.nn import ModuleList, Sequential, ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.25

    def _multihead(
        self, output_dim: int, num_nodes: int, num_shared: int
    ):
        denselayers = []  # shared dense layers, before mutli-heads
        for ishare in range(num_shared):
            denselayers.append(Linear(self.hidden_dim, self.hidden_dim))
            denselayers.append(ReLU())
        self.shared = Sequential(*denselayers)

        # currently, only two types of outputs are considered, graph-level scalars and nodes-level vectors with num_nodes dimension, or mixed or the two
        if output_dim < num_nodes:  # all graph-level outputs
            self.num_heads = output_dim
            outputs_dims = [1 for _ in range(self.num_heads)]
        elif output_dim % num_nodes == 0:  # all node-level outputs
            self.num_heads = output_dim // num_nodes
            outputs_dims = [num_nodes for _ in range(self.num_heads)]
        else:  # mixed graph-level and node-level
            self.num_heads = output_dim % num_nodes + output_dim // num_nodes
            outputs_dims = [
                1 if ih < output_dim % num_nodes else num_nodes
                for ih in range(self.num_heads)
            ]

        self.num_heads = len(outputs_dims)  # number of heads/tasks
        self.heads = ModuleList()
        for ihead in range(self.num_heads):
            mlp = Sequential(
                Linear(self.hidden_dim, 50),
                ReLU(),
                Linear(50, 25),
                ReLU(),
                Linear(25, outputs_dims[ihead]),
            )
            self.heads.append(mlp)

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        ### encoder part ####
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x=x, edge_index=edge_index)))
        x = global_mean_pool(x, batch)
        x = self.shared(x)  # shared dense layers
        #### multi-head decoder part####
        outputs = []
        for headloc in self.heads:
            outputs.append(headloc(x))
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
