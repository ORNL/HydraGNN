import json

import torch

from utils.utils import (
    dataset_loading_and_splitting,
    test,
    setup_ddp,
)
from utils.models_setup import generate_model
from data_utils.dataset_descriptors import (
    Dataset,
)


def test_trained_model(config_file: str = None, chosen_model: torch.nn.Module = None):

    if config_file is None:
        raise RuntimeError("No configure file provided")

    if chosen_model is None:
        raise RuntimeError("No model type provided")

    run_in_parallel, world_size, world_rank = setup_ddp()

    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

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
    state_dict = torch.load(
        "./logs/" + model_with_config_name + "/" + model_with_config_name + ".pk",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)

    error, true_values, predicted_values = test(
        test_loader, model, config["NeuralNetwork"]["Architecture"]["output_dim"]
    )

    return error, true_values, predicted_values
