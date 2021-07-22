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
    chosen_atom_features = int(input("Selected value: "))
    config["atom_features"] = [
        x.value for x in atom_features_options[chosen_atom_features]
    ]

    config["batch_size"] = int(input("Select batch size(8,16,32,64): "))
    config["hidden_dim"] = int(input("Select hidden dimension: "))
    config["num_conv_layers"] = int(input("Select number of convolutional layers: "))
    config["learning_rate"] = float(input("Select learning rate: "))
    config["radius"] = int(
        input("Select the radius within which neighbours of an atom are chosen: ")
    )
    config["max_num_node_neighbours"] = int(
        input("Select the maximum number of atom neighbours: ")
    )
    config["num_epoch"] = int(input("Select the number of epochs: "))
    config["perc_train"] = float(input("Select train percentage: "))

    predicted_value_option = {1: 1, 2: 32, 3: 32, 4: 33, 5: 33, 6: 65}
    print(
        "Select the values you want to predict: 1)free energy 2)charge density 3)magnetic moment 4)free energy+charge density 5)free energy+magnetic moment, 6)free energy+charge density+magnetic moment"
    )
    chosen_prediction_value = int(input("Selected value: "))
    config["output_dim"] = predicted_value_option[chosen_prediction_value]
    config["predicted_value_option"] = chosen_prediction_value

    dataset_options = {
        1: Dataset.CuAu,
        2: Dataset.FePt,
        3: Dataset.CuAu_FePt_SHUFFLE,
        4: Dataset.CuAu_TRAIN_FePt_TEST,
        5: Dataset.FePt_TRAIN_CuAu_TEST,
        6: Dataset.FeSi,
        7: Dataset.FePt_FeSi_SHUFFLE,
    }
    print(
        "Select the dataset you want to use: 1) CuAu 2) FePt 3)Combine CuAu-FePt&Shuffle 4)CuAu-train, FePt-test 5)FePt-train, CuAu-test, 6)FeSi , 7) Combine FePt-FeSi&Shuffle"
    )
    chosen_dataset_option = int(input("Selected value: "))
    config["dataset_option"] = dataset_options[chosen_dataset_option].value
    print(
        "Do you want to use subsample of the dataset? If yes input the percentage of the original dataset if no enter 0."
    )
    subsample_percentage = float(input("Selected value: "))
    if subsample_percentage > 0:
        config["subsample_percentage"] = subsample_percentage
    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=dataset_options[chosen_dataset_option],
    )

    input_dim = len(config["atom_features"])
    model_choices = {"1": "GIN", "2": "PNN", "3": "GAT", "4": "MFC"}
    print("Select which model you want to use: 1) GIN 2) PNN 3) GAT 4) MFC")
    chosen_model = model_choices[input("Selected value: ")]

    model = generate_model(
        model_type=chosen_model,
        input_dim=input_dim,
        dataset=train_loader.dataset,
        config=config,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model_with_config_name = (
        model.__str__()
        + "-r-"
        + str(config["radius"])
        + "-mnnn-"
        + str(config["max_num_node_neighbours"])
        + "-ncl-"
        + str(model.num_conv_layers)
        + "-hd-"
        + str(model.hidden_dim)
        + "-ne-"
        + str(config["num_epoch"])
        + "-lr-"
        + str(config["learning_rate"])
        + "-bs-"
        + str(config["batch_size"])
        + "-data-"
        + config["dataset_option"]
        + "-node_ft-"
        + "".join(str(x) for x in config["atom_features"])
        + "-pred_val-"
        + str(config["predicted_value_option"])
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
        config,
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
    predicted_value_option = {1: 1, 2: 32, 3: 32, 4: 33, 5: 33, 6: 65}
    config["output_dim"] = predicted_value_option[config["predicted_value_option"]]

    dataset_options = {
        1: Dataset.CuAu,
        2: Dataset.FePt,
        3: Dataset.CuAu_FePt_SHUFFLE,
        4: Dataset.CuAu_TRAIN_FePt_TEST,
        5: Dataset.FePt_TRAIN_CuAu_TEST,
        6: Dataset.FeSi,
        7: Dataset.FePt_FeSi_SHUFFLE,
    }
    chosen_dataset_option = None
    for dataset in dataset_options.values():
        if dataset.value == config["dataset_option"]:
            chosen_dataset_option = dataset

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=chosen_dataset_option,
        distributed_data_parallelism=run_in_parallel,
    )

    if "out_wunit" in config and config["out_wunit"]:
        dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['dataset_option']}.pkl"
        with open(dataset_path, "rb") as f:
            x_minmax = pickle.load(f)
            y_minmax = pickle.load(f)
        config["x_minmax"] = []
        config["y_minmax"] = []
        config["x_minmax"].append(x_minmax[:, :, config["atom_features"]].tolist())
        if config["predicted_value_option"] == 1:
            config["y_minmax"].append(y_minmax[:, 0].tolist())
        elif config["predicted_value_option"] == 2:
            config["y_minmax"].append(x_minmax[:, :, 1].tolist())
        elif config["predicted_value_option"] == 3:
            config["y_minmax"].append(x_minmax[:, :, 2].tolist())
        elif config["predicted_value_option"] == 4:
            config["y_minmax"].append(y_minmax[:, 0].tolist())
            config["y_minmax"].append(x_minmax[:, :, 1].tolist())
        elif config["predicted_value_option"] == 5:
            config["y_minmax"].append(y_minmax[:, 0].tolist())
            config["y_minmax"].append(x_minmax[:, :, 2].tolist())
        elif config["predicted_value_option"] == 6:
            config["y_minmax"].append(y_minmax[:, 0].tolist())
            config["y_minmax"].append(x_minmax[:, :, 1].tolist())
            config["y_minmax"].append(x_minmax[:, :, 2].tolist())
    else:
        config["out_wunit"] = "False"

    model = generate_model(
        model_type=config["model_type"],
        input_dim=len(config["atom_features"]),
        dataset=train_loader.dataset,
        config=config,
    )

    model_with_config_name = (
        model.__str__()
        + "-r-"
        + str(config["radius"])
        + "-mnnn-"
        + str(config["max_num_node_neighbours"])
        + "-ncl-"
        + str(model.num_conv_layers)
        + "-hd-"
        + str(model.hidden_dim)
        + "-ne-"
        + str(config["num_epoch"])
        + "-lr-"
        + str(config["learning_rate"])
        + "-bs-"
        + str(config["batch_size"])
        + "-data-"
        + config["dataset_option"]
        + "-node_ft-"
        + "".join(str(x) for x in config["atom_features"])
        + "-pred_val-"
        + str(config["predicted_value_option"])
    )

    device_name, device = get_device()
    if run_in_parallel:
        if device_name == "cpu":
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device]
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
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
        config,
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
