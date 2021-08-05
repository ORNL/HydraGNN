import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import GATConv, BatchNorm

from .Base import Base


class GATStack(Base):
    def __init__(
        self,
        input_dim: int,
        output_dim: list,
        num_nodes: int,
        hidden_dim: int = 16,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_shared: int = 1,
    ):
        super().__init__()

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

        super()._multihead(output_dim, num_nodes, num_shared)

    def __str__(self):
        return "GATStack"
