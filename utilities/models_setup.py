from models.GINStack import GINStack
from models.PNNStack import PNNStack
from models.GATStack import GATStack
from models.MFCStack import MFCStack
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def generate_model(
    model_type: str, input_dim: int, dataset: [Data], max_num_node_neighbours: int
):
    torch.manual_seed(0)
    num_conv_layers = int(
        input("Number of convolutional layers(depends on the model used): ")
    )
    hidden_dim = int(input("Size of the convolutional layer(hidden dimension): "))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "GIN":
        model = GINStack(
            input_dim=input_dim, hidden_dim=hidden_dim, num_conv_layers=num_conv_layers
        ).to(device)

    elif model_type == "PNN":
        data_size = len(dataset)
        deg = torch.zeros(max_num_node_neighbours + 1, dtype=torch.long)
        for data in dataset[: int(data_size * 0.7)]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNNStack(
            deg=deg,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
        ).to(device)

    elif model_type == "GAT":
        heads = int(input("Enter the number of multi-head-attentions(default 1): "))
        negative_slope = float(
            input("Enter LeakyReLU angle of the negative slope(default 0.2): ")
        )
        dropout = float(
            input(
                "Enter dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training(default 0): "
            )
        )

        model = GATStack(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            num_conv_layers=num_conv_layers,
        ).to(device)

    elif model_type == "MFC":
        model = MFCStack(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_degree=max_num_node_neighbours,
            num_conv_layers=num_conv_layers,
        ).to(device)

    return model
