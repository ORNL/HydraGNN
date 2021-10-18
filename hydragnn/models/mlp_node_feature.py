import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential, ReLU, Linear, Module


class mlp_node_feature(Module):
    def __init__(self, input_dim, output_dim, num_nodes, hidden_dim_node):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes

        self.mlp = ModuleList()
        for inode in range(self.num_nodes):
            denselayers = []
            denselayers.append(Linear(self.input_dim, hidden_dim_node[0]))
            denselayers.append(ReLU())
            for ilayer in range(len(hidden_dim_node) - 1):
                denselayers.append(
                    Linear(hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1])
                )
                denselayers.append(ReLU())
            denselayers.append(Linear(hidden_dim_node[-1], output_dim))
            self.mlp.append(Sequential(*denselayers))

    def node_features_reshape(self, x, batch):
        """reshape x from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = x.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=x.dtype,
            device=x.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = x[inode_index, :]
        return out

    def forward(self, x: Tensor, batch: Tensor, share_mlp: bool = False):
        outs = torch.zeros(
            (x.shape[0], self.output_dim),
            dtype=x.dtype,
            device=x.device,
        )
        if share_mlp:
            outs = self.mlp[0](x)
        else:
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs
