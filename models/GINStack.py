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
        output_type: list,
        num_nodes: int,
        hidden_dim: int,
        config_heads: {},
        dropout: float = 0.25,
        num_conv_layers: int = 16,
    ):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ),
                eps=100.0,
                train_eps=True,
            )
        )

        self.batch_norms.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            conv = GINConv(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ),
                eps=100.0,
                train_eps=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        super()._multihead(output_dim, num_nodes, output_type, config_heads)

    def __str__(self):
        return "GINStack"
