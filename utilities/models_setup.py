from models.GINStack import GINStack
from models.PNNStack import PNNStack
from models.GATStack import GATStack
from models.MFCStack import MFCStack
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def generate_model(model_type: str, input_dim: int, dataset: [Data], config: dict):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "GIN":
        model = GINStack(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
        ).to(device)

    elif model_type == "PNN":
        deg = torch.zeros(config["max_num_node_neighbours"] + 1, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        dropout = float(input("Enter dropout probability: "))
        model = PNNStack(
            deg=deg,
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
            dropout=dropout,
        ).to(device)

    elif model_type == "GAT":
        # heads = int(input("Enter the number of multi-head-attentions(default 1): "))
        # negative_slope = float(
        #     input("Enter LeakyReLU angle of the negative slope(default 0.2): ")
        # )
        # dropout = float(
        #     input(
        #         "Enter dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training(default 0): "
        #     )
        # )

        model = GATStack(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
        ).to(device)

    elif model_type == "MFC":
        model = MFCStack(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            max_degree=config["max_num_node_neighbours"],
            num_conv_layers=config["num_conv_layers"],
        ).to(device)

    return model
