import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool

from .Base import Base


class PNNStack(Base):
    def __init__(
        self,
        deg,
        input_dim,
        output_dim,
        num_nodes,
        hidden_dim,
        num_conv_layers,
        num_shared=1,
    ):
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
                deg=deg,
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
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))
        ############multiple heads/taks################
        denselayers = []  # shared dense layers, before mutli-heads
        for ishare in range(num_shared):
            denselayers.append(Linear(self.hidden_dim, self.hidden_dim))
            denselayers.append(ReLU())
        self.shared = Sequential(*denselayers)

        # currently, only two types of outputs are considered, graph-level scalars and nodes-level vectors with num_nodes dimension, or mixed or the two
        if output_dim < num_nodes:  # all graph-level outputs
            self.num_heads = output_dim
            outputs_dims = [1 for _ in range(self.num_heads)]
        elif output_dim % num_nodes == 0:  # all node-level outputs
            self.num_heads = output_dim // num_nodes
            outputs_dims = [num_nodes for _ in range(self.num_heads)]
        else:  # mixed graph-level and node-level
            self.num_heads = output_dim % num_nodes + output_dim // num_nodes
            outputs_dims = [
                1 if ih < output_dim % num_nodes else num_nodes
                for ih in range(self.num_heads)
            ]

        self.num_heads = len(outputs_dims)  # number of heads/tasks
        self.heads = ModuleList()
        for ihead in range(self.num_heads):
            mlp = Sequential(
                Linear(self.hidden_dim, 50),
                ReLU(),
                Linear(50, 25),
                ReLU(),
                Linear(25, outputs_dims[ihead]),
            )
            self.heads.append(mlp)

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x=x, edge_index=edge_index)))
        x = global_mean_pool(x, batch)
        ####
        x = self.shared(x)  # shared dense layers
        outputs = []
        for headloc in self.heads:
            outputs.append(headloc(x))
        return torch.cat(outputs, dim=1)

    def __str__(self):
        return "PNNStack"
