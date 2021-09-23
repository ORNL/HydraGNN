import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from .GINStack import GINStack
from .PNAStack import PNAStack
from .GATStack import GATStack
from .MFCStack import MFCStack

from utils.utils import get_comm_size_and_rank


def get_gpu_list():

    available_gpus = [i for i in range(torch.cuda.device_count())]

    return available_gpus


def get_device(use_gpu=True, rank_per_model=1):

    available_gpus = get_gpu_list()
    if not use_gpu or not available_gpus:
        print("Using CPU")
        return "cpu", torch.device("cpu")

    world_size, world_rank = get_comm_size_and_rank()
    if rank_per_model != 1:
        raise ValueError("Exactly 1 rank per device currently supported")

    print("Using GPU")
    ## We need to ge a local rank if there are multiple GPUs available.
    localrank = 0
    if torch.cuda.device_count() > 1:
        if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
            ## Summit
            localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        elif os.getenv("SLURM_LOCALID"):
            ## CADES
            localrank = int(os.environ["SLURM_LOCALID"])

        if localrank >= torch.cuda.device_count():
            print(
                "WARN: localrank is greater than the available device count - %d %d"
                % (localrank, torch.cuda.device_count())
            )

    device_name = "cuda:" + str(localrank)

    return device_name, torch.device(device_name)


def generate_model(
    model_type: str,
    input_dim: int,
    dataset: [Data],
    config: dict,
    use_gpu: bool = True,
    use_distributed: bool = False,
):
    torch.manual_seed(0)

    _, device = get_device(use_gpu)

    num_atoms = dataset[0].num_nodes  # FIXME: assumes constant number of atoms

    if model_type == "GIN":
        model = GINStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            num_nodes=num_atoms,
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)

    elif model_type == "PNA":
        deg = torch.zeros(config["max_neighbours"] + 1, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNAStack(
            deg=deg,
            input_dim=input_dim,
            output_dim=config["output_dim"],
            num_nodes=num_atoms,
            hidden_dim=config["hidden_dim"],
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
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
            num_nodes=num_atoms,
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)

    elif model_type == "MFC":
        model = MFCStack(
            input_dim=input_dim,
            output_dim=config["output_dim"],
            num_nodes=num_atoms,
            hidden_dim=config["hidden_dim"],
            max_degree=config["max_neighbours"],
            num_conv_layers=config["num_conv_layers"],
            output_type=config["output_type"],
            config_heads=config["output_heads"],
            loss_weights=config["task_weights"],
        ).to(device)

    return model
