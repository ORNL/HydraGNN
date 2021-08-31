import sys, os, json

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.utils import (
    dataset_loading_and_splitting,
    train_validate_test_normal,
    setup_ddp,
    get_comm_size_and_rank,
)
from utils.models_setup import generate_model, get_device
from data_utils.dataset_descriptors import (
    AtomFeatures,
    Dataset,
)
import pickle


def run_normal_terminal_input():
    config = {}
    load_from_json = int(
        input(
            "Do you want to upload the configuration from the file or input it from terminal(use configuration.json in the utils folder)? 1)JSON 2)Terminal"
        )
    )
    if load_from_json == 1:
        with open("./examples/configuration.json", "r") as f:
            config = json.load(f)

    atom_features_options = {
        1: [AtomFeatures.NUM_OF_PROTONS],
        2: [AtomFeatures.NUM_OF_PROTONS, AtomFeatures.CHARGE_DENSITY],
        3: [AtomFeatures.NUM_OF_PROTONS, AtomFeatures.MAGNETIC_MOMENT],
        4: [
            AtomFeatures.NUM_OF_PROTONS,
            AtomFeatures.CHARGE_DENSITY,
            AtomFeatures.MAGNETIC_MOMENT,
        ],
    }
    print(
        "Select the atom features you want in the dataset: 1)proton number 2)proton number+charge density 3)proton number+magnetic moment 4)all"
    )
    config["Dataset"] = {}
    config["Dataset"]["node_features"] = {}
    config["Dataset"]["graph_features"] = {}

    config["NeuralNetwork"] = {}
    config["NeuralNetwork"]["Architecture"] = {}
    config["NeuralNetwork"]["Variables_of_interest"] = {}
    config["NeuralNetwork"]["Training"] = {}

    config["Dataset"]["node_features"]["name"] = [
        "num_of_protons",
        "charge_density",
        "magnetic_moment",
    ]
    config["Dataset"]["node_features"]["dim"] = [1, 1, 1]
    config["Dataset"]["node_features"]["column_index"] = [0, 5, 6]
    config["Dataset"]["graph_features"]["name"] = ["free_energy"]
    config["Dataset"]["graph_features"]["dim"] = [1]
    config["Dataset"]["graph_features"]["column_index"] = [0]

    config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"] = []
    chosen_atom_features = int(input("Selected value: "))
    config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"] = [
        x.value for x in atom_features_options[chosen_atom_features]
    ]

    config["NeuralNetwork"]["Training"]["batch_size"] = int(
        input("Select batch size(8,16,32,64): ")
    )
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = int(
        input("Select hidden dimension: ")
    )
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = int(
        input("Select number of convolutional layers: ")
    )
    config["NeuralNetwork"]["Training"]["learning_rate"] = float(
        input("Select learning rate: ")
    )
    config["NeuralNetwork"]["Architecture"]["radius"] = int(
        input("Select the radius within which neighbours of a node are chosen: ")
    )
    config["NeuralNetwork"]["Architecture"]["max_neighbours"] = int(
        input("Select the maximum number of node neighbours: ")
    )
    config["NeuralNetwork"]["Training"]["num_epoch"] = int(
        input("Select the number of epochs: ")
    )
    config["NeuralNetwork"]["Training"]["perc_train"] = float(
        input("Select train percentage: ")
    )

    print(
        "Select the values (one or multiple) you want to predict: 0)free energy 1)charge density 2)magnetic moment"
    )
    chosen_prediction_value = [int(item) for item in input("Selected value: ").split()]
    config["NeuralNetwork"]["Variables_of_interest"][
        "output_index"
    ] = chosen_prediction_value
    config["NeuralNetwork"]["Variables_of_interest"]["type"] = [
        "graph" if item == 0 else "node" for item in chosen_prediction_value
    ]

    dataset_options = {
        1: Dataset.CuAu,
        2: Dataset.FePt,
        3: Dataset.FeSi,
        4: Dataset.unit_test,
    }
    print("Select the dataset you want to use: 1) CuAu 2) FePt 3)FeSi , 4) unit_test")
    chosen_dataset_option = int(input("Selected value: "))
    config["Dataset"]["name"] = dataset_options[chosen_dataset_option].value
    if chosen_dataset_option < 4:
        config["Dataset"]["format"] = "LSMS"
    else:
        config["Dataset"]["format"] = "unit_test"

    config["Dataset"]["num_nodes"] = int(input("Enter the number of nodes: "))

    print(
        "Do you want to use subsample of the dataset? If yes input the percentage of the original dataset if no enter 0."
    )
    subsample_percentage = float(input("Selected value: "))
    if subsample_percentage > 0:
        config["NeuralNetwork"]["Variables_of_interest"][
            "subsample_percentage"
        ] = subsample_percentage

    config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = "False"
    print("Select if you want to denormalize the output: 0) No (by default), 1) Yes")
    if int(input("Selected value: ")) == 1:
        config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = "True"

    input_dim = len(
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
    )
    model_choices = {"1": "GIN", "2": "PNN", "3": "GAT", "4": "MFC"}
    print("Select which model you want to use: 1) GIN 2) PNN 3) GAT 4) MFC")
    chosen_model = model_choices[input("Selected value: ")]

    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]
    config["NeuralNetwork"]["Architecture"]["output_dim"] = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["graph_features"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            dim_item = (
                config["Dataset"]["node_features"]["dim"][output_index[item]]
                * config["Dataset"]["num_nodes"]
            )
        else:
            raise ValueError("Unknown output type", output_type[item])
        config["NeuralNetwork"]["Architecture"]["output_dim"].append(dim_item)

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=config["Dataset"]["name"],
    )

    if (
        "denormalize_output" in config["NeuralNetwork"]["Variables_of_interest"]
        and config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"]
        == "True"
    ):
        dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
        with open(dataset_path, "rb") as f:
            node_minmax = pickle.load(f)
            graph_minmax = pickle.load(f)
        config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"] = []
        config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"] = []
        feature_indices = [
            i
            for i in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        ]
        for item in feature_indices:
            config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"].append(
                node_minmax[:, :, item].tolist()
            )
        for item in range(len(output_type)):
            if output_type[item] == "graph":
                config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"].append(
                    graph_minmax[:, output_index[item], None].tolist()
                )
            elif output_type[item] == "node":
                config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"].append(
                    node_minmax[:, :, output_index[item]].tolist()
                )
            else:
                raise ValueError("Unknown output type", output_type[item])

    else:
        config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = "False"
    config["NeuralNetwork"]["Architecture"]["output_type"] = config["NeuralNetwork"][
        "Variables_of_interest"
    ]["type"]

    model = generate_model(
        model_type=chosen_model,
        input_dim=input_dim,
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["NeuralNetwork"]["Training"]["learning_rate"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model_with_config_name = (
        model.__str__()
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-mnnn-"
        + str(config["NeuralNetwork"]["Architecture"]["max_neighbours"])
        + "-ncl-"
        + str(model.num_conv_layers)
        + "-hd-"
        + str(model.hidden_dim)
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-data-"
        + config["Dataset"]["name"]
        + "-node_ft-"
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
    )
    writer = SummaryWriter("./logs/" + model_with_config_name)

    with open("./logs/" + model_with_config_name + "/config.json", "w") as f:
        json.dump(config, f)

    print(
        f"Starting training with the configuration: \n{json.dumps(config, indent=4, sort_keys=True)}"
    )
    train_validate_test_normal(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        model_with_config_name,
    )
    torch.save(
        model.state_dict(),
        "./logs/" + model_with_config_name + "/" + model_with_config_name + ".pk",
    )


def run_normal_config_file(config_file="./examples/configuration.json"):

    run_in_parallel, world_size, world_rank = setup_ddp()

    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=config["Dataset"]["name"],
        distributed_data_parallelism=run_in_parallel,
    )

    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]
    config["NeuralNetwork"]["Architecture"]["output_dim"] = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["graph_features"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            dim_item = (
                config["Dataset"]["node_features"]["dim"][output_index[item]]
                * config["Dataset"]["num_nodes"]
            )
        else:
            raise ValueError("Unknown output type", output_type[item])
        config["NeuralNetwork"]["Architecture"]["output_dim"].append(dim_item)

    if (
        "denormalize_output" in config["NeuralNetwork"]["Variables_of_interest"]
        and config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"]
        == "True"
    ):
        dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
        with open(dataset_path, "rb") as f:
            node_minmax = pickle.load(f)
            graph_minmax = pickle.load(f)
        config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"] = []
        config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"] = []
        feature_indices = [
            i
            for i in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        ]
        for item in feature_indices:
            config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"].append(
                node_minmax[:, :, item].tolist()
            )
        for item in range(len(output_type)):
            if output_type[item] == "graph":
                config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"].append(
                    graph_minmax[:, output_index[item], None].tolist()
                )
            elif output_type[item] == "node":
                config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"].append(
                    node_minmax[:, :, output_index[item]].tolist()
                )
            else:
                raise ValueError("Unknown output type", output_type[item])

    else:
        config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = "False"
    config["NeuralNetwork"]["Architecture"]["output_type"] = config["NeuralNetwork"][
        "Variables_of_interest"
    ]["type"]

    model = generate_model(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        ),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
    )

    model_with_config_name = (
        model.__str__()
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-mnnn-"
        + str(config["NeuralNetwork"]["Architecture"]["max_neighbours"])
        + "-ncl-"
        + str(model.num_conv_layers)
        + "-hd-"
        + str(model.hidden_dim)
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-data-"
        + config["Dataset"]["name"]
        + "-node_ft-"
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
    )

    device_name, device = get_device()
    if run_in_parallel:
        if device_name == "cpu":
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device]
            )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["NeuralNetwork"]["Training"]["learning_rate"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = None
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        _, world_rank = get_comm_size_and_rank()
        if int(world_rank) == 0:
            writer = SummaryWriter("./logs/" + model_with_config_name)
    else:
        writer = SummaryWriter("./logs/" + model_with_config_name)

    with open("./logs/" + model_with_config_name + "/config.json", "w") as f:
        json.dump(config, f)

    if (
        "continue" in config["NeuralNetwork"]["Training"]
        and config["NeuralNetwork"]["Training"]["continue"] == 1
    ):
        # starting from an existing model
        modelstart = config["NeuralNetwork"]["Training"]["startfrom"]
        if not modelstart:
            modelstart = model_with_config_name

        state_dict = torch.load(
            f"./logs/{modelstart}/{modelstart}.pk",
            map_location="cpu",
        )
        model.load_state_dict(state_dict)

    print(
        f"Starting training with the configuration: \n{json.dumps(config, indent=4, sort_keys=True)}"
    )
    train_validate_test_normal(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        model_with_config_name,
    )

    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        _, world_rank = get_comm_size_and_rank()
        if int(world_rank) == 0:
            torch.save(
                model.module.state_dict(),
                "./logs/"
                + model_with_config_name
                + "/"
                + model_with_config_name
                + ".pk",
            )
    else:
        torch.save(
            model.state_dict(),
            "./logs/" + model_with_config_name + "/" + model_with_config_name + ".pk",
        )


if __name__ == "__main__":
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        run_normal_terminal_input()
    else:
        run_normal_config_file()
