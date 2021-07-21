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
        8: Dataset.unit_test,
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
    state_dict = torch.load(
        "./logs/" + model_with_config_name + "/" + model_with_config_name + ".pk",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)

    error, true_values, predicted_values = test(
        test_loader, model, config["output_dim"]
    )

    return error, true_values, predicted_values
