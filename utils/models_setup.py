import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from models.GINStack import GINStack
from models.PNNStack import PNNStack
from models.GATStack import GATStack
from models.MFCStack import MFCStack


def get_gpus_list():

    available_gpus = [i for i in range(torch.cuda.device_count())]

    return available_gpus


def get_gpu(number):
    gpus_list = get_gpus_list()
    if number not in gpus_list:
        raise ValueError(
            "The GPU ID:" + str(number) + " is not inside the list of GPUs available"
        )
    else:
        device = torch.device(
            "cuda:" + str(number)
        )  # you can continue going on here, like cuda:1 cuda:2....etc.

    return device


def generate_model(
    model_type: str,
    input_dim: int,
    dataset: [Data],
    config: dict,
    distributed_data_parallelism: bool = False,
):

    if distributed_data_parallelism:
        world_size = os.environ["OMPI_COMM_WORLD_SIZE"]
        world_rank = os.environ["OMPI_COMM_WORLD_RANK"]

    torch.manual_seed(0)

    available_gpus = get_gpus_list()
    if len(available_gpus) > 0:
        device = get_gpu(int(world_rank) % len(available_gpus))
    else:
        device = torch.device("cpu")

    if model_type == "GIN":
        model = GINStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
        ).to(device)

    elif model_type == "PNN":
        deg = torch.zeros(config["max_num_node_neighbours"] + 1, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNNStack(
            deg=deg,
            input_dim=input_dim,
            output_dim=config["output_dim"],
            num_nodes=dataset[0].num_nodes,
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
            num_shared=1,
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
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
        ).to(device)

    elif model_type == "MFC":
        model = MFCStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            max_degree=config["max_num_node_neighbours"],
            num_conv_layers=config["num_conv_layers"],
        ).to(device)

    return model
