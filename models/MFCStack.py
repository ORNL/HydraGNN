import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import MFConv, BatchNorm, global_mean_pool

from .Base import Base


class MFCStack(Base):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        max_degree: int,
        hidden_dim: int = 16,
        dropout: float = 0.25,
        num_conv_layers: int = 16,
        num_shared: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
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

        super()._multihead(output_dim, num_nodes, num_shared)

    def __str__(self):
        return "MFCStack"
