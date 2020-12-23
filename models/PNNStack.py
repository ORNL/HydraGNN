import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool


class PNNStack(torch.nn.Module):
    def __init__(self, deg, input_dim, hidden_dim, num_conv_layers):
        super(PNNStack, self).__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]

        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            PNAConv(
                in_channels=input_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                edge_dim=1,
                deg=deg,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )
        for _ in range(self.num_conv_layers):
            conv = PNAConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                edge_dim=1,
                deg=deg,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        self.mlp = Sequential(
            Linear(self.hidden_dim, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        for conv, batch_norm in zip(
            self.convs, self.batch_norms
        ):
            x = F.relu(
                    batch_norm(conv(x=x, edge_index=edge_index, edge_attr=edge_attr))
                )
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
        return "PNNStack"
