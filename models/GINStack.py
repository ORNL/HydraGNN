import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GINConv, BatchNorm

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
        self.batch_norms = ModuleList()
        self.convs.append(self.build_conv_model(input_dim, self.hidden_dim))
        for _ in range(self.num_conv_layers):
            self.convs.append(self.build_conv_model(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        super()._multihead(output_dim, num_nodes, num_shared)

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )

    def __str__(self):
        return "GINStack"
