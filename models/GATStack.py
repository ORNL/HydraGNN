import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool


class GATStack(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim: int = 16,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
    ):
        super(GATStack, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.heads = heads
        self.negative_slope = negative_slope
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            GATConv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                heads=self.heads,
                negative_slope=self.negative_slope,
                dropout=self.dropout,
                add_self_loops=True,
            )
        )
        for _ in range(self.num_conv_layers):
            conv = GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=self.heads,
                negative_slope=self.negative_slope,
                dropout=self.dropout,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        self.mlp = Sequential(
            Linear(self.hidden_dim, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, output_dim)
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
        return "GATStack"
