import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import MFConv, BatchNorm, global_mean_pool


class MFCStack(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        max_degree: int,
        hidden_dim: int = 16,
        num_conv_layers: int = 16,
    ):
        super(MFCStack, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            MFConv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                max_degree=self.max_degree,
            )
        )
        for _ in range(self.num_conv_layers):
            conv = MFConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                max_degree=self.max_degree,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        self.mlp = Sequential(
            Linear(self.hidden_dim, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.mlp(x)

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
        return "MFCStack"
