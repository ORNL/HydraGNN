import sys, os, json

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.utils import (
    dataset_loading_and_splitting,
    train_validate_test_normal,
    setup_ddp,
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
    config["Dataset"]["atom_features"] = {}
    config["Dataset"]["properties"] = {}

    config["NeuralNetwork"] = {}
    config["NeuralNetwork"]["Architecture"] = {}
    config["NeuralNetwork"]["Target_dataset"] = {}
    config["NeuralNetwork"]["Training"] = {}

    config["Dataset"]["atom_features"]["name"] = [
        "num_of_protons",
        "charge_density",
        "magnetic_moment",
    ]
    config["Dataset"]["atom_features"]["dim"] = [1, 1, 1]
    config["Dataset"]["atom_features"]["locations"] = [0, 5, 6]
    config["Dataset"]["properties"]["name"] = ["free_energy"]
    config["Dataset"]["properties"]["dim"] = [1]
    config["Dataset"]["properties"]["locations"] = [0]

    config["NeuralNetwork"]["Target_dataset"]["input_atom_features"] = []
    chosen_atom_features = int(input("Selected value: "))
    config["NeuralNetwork"]["Target_dataset"]["input_atom_features"] = [
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
        input("Select the radius within which neighbours of an atom are chosen: ")
    )
    config["NeuralNetwork"]["Architecture"]["max_num_node_neighbours"] = int(
        input("Select the maximum number of atom neighbours: ")
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
    config["NeuralNetwork"]["Target_dataset"]["output_index"] = chosen_prediction_value
    config["NeuralNetwork"]["Target_dataset"]["type"] = [
        "graph" if item == 0 else "node" for item in chosen_prediction_value
    ]

    dataset_options = {
        1: Dataset.CuAu,
        2: Dataset.FePt,
        3: Dataset.CuAu_FePt_SHUFFLE,
        4: Dataset.CuAu_TRAIN_FePt_TEST,
        5: Dataset.FePt_TRAIN_CuAu_TEST,
        6: Dataset.FeSi,
        7: Dataset.FePt_FeSi_SHUFFLE,
        8: Dataset.unit_test,
    }
    print(
        "Select the dataset you want to use: 1) CuAu 2) FePt 3)Combine CuAu-FePt&Shuffle 4)CuAu-train, FePt-test 5)FePt-train, CuAu-test, 6)FeSi , 7) Combine FePt-FeSi&Shuffle"
    )
    chosen_dataset_option = int(input("Selected value: "))
    config["Dataset"]["name"] = dataset_options[chosen_dataset_option].value
    config["Dataset"]["num_atoms"] = int(input("Enter the number of atoms: "))
    config["Dataset"]["properties"]["dim"] = [
        1,
        config["Dataset"]["num_atoms"],
        config["Dataset"]["num_atoms"],
    ]

    print(
        "Do you want to use subsample of the dataset? If yes input the percentage of the original dataset if no enter 0."
    )
    subsample_percentage = float(input("Selected value: "))
    if subsample_percentage > 0:
        config["NeuralNetwork"]["Target_dataset"][
            "subsample_percentage"
        ] = subsample_percentage

    config["NeuralNetwork"]["Target_dataset"]["denormalize_output"] = "False"
    print("Select if you want to denormalize the output: 0) No (by default), 1) Yes")
    if int(input("Selected value: ")) == 1:
        config["NeuralNetwork"]["Target_dataset"]["denormalize_output"] = "True"

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=dataset_options[chosen_dataset_option],
    )

    output_type = config["NeuralNetwork"]["Target_dataset"]["type"]
    output_index = config["NeuralNetwork"]["Target_dataset"]["output_index"]
    config["NeuralNetwork"]["Architecture"]["output_dim"] = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["properties"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            dim_item = (
                config["Dataset"]["atom_features"]["dim"][output_index[item]]
                * config["Dataset"]["num_atoms"]
            )
        else:
            raise ValueError("Unknown output type", output_type[item])
        config["NeuralNetwork"]["Architecture"]["output_dim"].append(dim_item)

    if (
        "denormalize_output" in config["NeuralNetwork"]["Target_dataset"]
        and config["NeuralNetwork"]["Target_dataset"]["denormalize_output"] == "True"
    ):
        dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
        with open(dataset_path, "rb") as f:
            node_minmax = pickle.load(f)
            graph_minmax = pickle.load(f)
        config["NeuralNetwork"]["Target_dataset"]["x_minmax"] = []
        config["NeuralNetwork"]["Target_dataset"]["y_minmax"] = []
        feature_indices = [
            i for i in config["NeuralNetwork"]["Target_dataset"]["input_atom_features"]
        ]
        for item in feature_indices:
            config["NeuralNetwork"]["Target_dataset"]["x_minmax"].append(
                node_minmax[:, :, item].tolist()
            )
        for item in range(len(output_type)):
            if output_type[item] == "graph":
                config["NeuralNetwork"]["Target_dataset"]["y_minmax"].append(
                    graph_minmax[:, output_index[item], None].tolist()
                )
            elif output_type[item] == "node":
                config["NeuralNetwork"]["Target_dataset"]["y_minmax"].append(
                    node_minmax[:, :, output_index[item]].tolist()
                )
            else:
                raise ValueError("Unknown output type", output_type[item])

    else:
        config["NeuralNetwork"]["Target_dataset"]["denormalize_output"] = "False"

    input_dim = len(config["NeuralNetwork"]["Target_dataset"]["input_atom_features"])
    model_choices = {"1": "GIN", "2": "PNN", "3": "GAT", "4": "MFC"}
    print("Select which model you want to use: 1) GIN 2) PNN 3) GAT 4) MFC")
    chosen_model = model_choices[input("Selected value: ")]

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
        + str(config["NeuralNetwork"]["Architecture"]["max_num_node_neighbours"])
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
            for x in config["NeuralNetwork"]["Target_dataset"]["input_atom_features"]
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

    dataset_options = {
        1: Dataset.CuAu,
        2: Dataset.FePt,
        3: Dataset.CuAu_FePt_SHUFFLE,
        4: Dataset.CuAu_TRAIN_FePt_TEST,
        5: Dataset.FePt_TRAIN_CuAu_TEST,
        6: Dataset.FeSi,
        7: Dataset.FePt_FeSi_SHUFFLE,
        8: Dataset.unit_test,
    }
    chosen_dataset_option = None
    for dataset in dataset_options.values():
        if dataset.value == config["Dataset"]["name"]:
            chosen_dataset_option = dataset

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=chosen_dataset_option,
        distributed_data_parallelism=run_in_parallel,
    )

    output_type = config["NeuralNetwork"]["Target_dataset"]["type"]
    output_index = config["NeuralNetwork"]["Target_dataset"]["output_index"]
    config["NeuralNetwork"]["Architecture"]["output_dim"] = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["properties"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            dim_item = (
                config["Dataset"]["atom_features"]["dim"][output_index[item]]
                * config["Dataset"]["num_atoms"]
            )
        else:
            raise ValueError("Unknown output type", output_type[item])
        config["NeuralNetwork"]["Architecture"]["output_dim"].append(dim_item)

    if (
        "denormalize_output" in config["NeuralNetwork"]["Target_dataset"]
        and config["NeuralNetwork"]["Target_dataset"]["denormalize_output"] == "True"
    ):
        dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
        with open(dataset_path, "rb") as f:
            node_minmax = pickle.load(f)
            graph_minmax = pickle.load(f)
        config["NeuralNetwork"]["Target_dataset"]["x_minmax"] = []
        config["NeuralNetwork"]["Target_dataset"]["y_minmax"] = []
        feature_indices = [
            i for i in config["NeuralNetwork"]["Target_dataset"]["input_atom_features"]
        ]
        for item in feature_indices:
            config["NeuralNetwork"]["Target_dataset"]["x_minmax"].append(
                node_minmax[:, :, item].tolist()
            )
        for item in range(len(output_type)):
            if output_type[item] == "graph":
                config["NeuralNetwork"]["Target_dataset"]["y_minmax"].append(
                    graph_minmax[:, output_index[item], None].tolist()
                )
            elif output_type[item] == "node":
                config["NeuralNetwork"]["Target_dataset"]["y_minmax"].append(
                    node_minmax[:, :, output_index[item]].tolist()
                )
            else:
                raise ValueError("Unknown output type", output_type[item])

    else:
        config["NeuralNetwork"]["Target_dataset"]["denormalize_output"] = "False"

    model = generate_model(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(config["NeuralNetwork"]["Target_dataset"]["input_atom_features"]),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
    )

    model_with_config_name = (
        model.__str__()
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-mnnn-"
        + str(config["NeuralNetwork"]["Architecture"]["max_num_node_neighbours"])
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
            for x in config["NeuralNetwork"]["Target_dataset"]["input_atom_features"]
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
    save_state = False
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        world_rank = os.environ["OMPI_COMM_WORLD_RANK"]
        if int(world_rank) == 0:
            save_state = True
    else:
        save_state = True

    if save_state:
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
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
                "./logs/"
                + model_with_config_name
                + "/"
                + model_with_config_name
                + ".pk",
            )


if __name__ == "__main__":
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        run_normal_terminal_input()
    else:
        run_normal_config_file()
