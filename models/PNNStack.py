import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, global_mean_pool

class PNNStack(torch.nn.Module):
    def __init__(self, deg, input_dim, hidden_dim, num_conv_layer):
        super(PNNStack, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.dropout = 0.25
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(PNAConv(in_channels=input_dim, out_channels=hidden_dim,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False))
        for _ in range(num_conv_layer):
            conv = PNAConv(in_channels=hidden_dim, out_channels=hidden_dim,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(Linear(hidden_dim, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

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
        return F.mse_loss(pred, value)

    def __str__(self):
        return "PNNStack"