import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class Base(torch.nn.Module):
    def __init__(self):
        super(Base, self).__init__()

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
        return "Base"
