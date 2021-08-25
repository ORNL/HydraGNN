import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
import torch_geometric.nn as pyg_nn

from .Base import Base


class GINStack(Base):
    def __init__(
        self,
        input_dim: int,
        output_dim: list,
        num_nodes: int,
        hidden_dim: int,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_shared: int = 1,
    ):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.convs = ModuleList()
        self.convs.append(self.build_conv_model(input_dim, self.hidden_dim))
        self.batch_norms = ModuleList()
        self.lns = ModuleList()
        for l in range(self.num_conv_layers):
            self.convs.append(self.build_conv_model(self.hidden_dim, self.hidden_dim))
            self.lns.append(pyg_nn.BatchNorm(self.hidden_dim))

        super()._multihead(output_dim, num_nodes, num_shared)

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return pyg_nn.GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            if not i == self.num_conv_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_mean_pool(x, batch)

        x = self.shared(x)  # shared dense layers
        #### multi-head decoder part####
        outputs = []
        for headloc in self.heads:
            outputs.append(headloc(x))
        return torch.cat(outputs, dim=1)

        return x

    def __str__(self):
        return "GINStack"
