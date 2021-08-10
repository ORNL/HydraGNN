import torch
from torch.nn import ModuleList, Sequential, ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.25

    def _multihead(self, output_dim: list, num_nodes: int, num_shared: int):
        denselayers = []  # shared dense layers, before mutli-heads
        for ishare in range(num_shared):
            denselayers.append(Linear(self.hidden_dim, self.hidden_dim))
            denselayers.append(ReLU())
        self.shared = Sequential(*denselayers)

        ############multiple heads/taks################
        # get number of heads from input
        ##One head represent one variable
        ##Head can have different sizes, head_dims;
        ###e.g., 1 for energy, 32 for charge density, 32*3 for magnetic moments
        self.num_heads = len(output_dim)
        self.head_dims = output_dim

        self.heads = ModuleList()
        for ihead in range(self.num_heads):
            mlp = Sequential(
                Linear(self.hidden_dim, 50),
                ReLU(),
                Linear(50, 25),
                ReLU(),
                Linear(25, self.head_dims[ihead]),
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
