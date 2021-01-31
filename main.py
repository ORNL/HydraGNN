import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ray import tune
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import numpy as np

from functools import partial
import os
import json

from utilities.utils import (
    load_data,
    dataset_splitting,
    train_validate_test_hyperopt,
    train_validate_test_normal,
)
from utilities.models_setup import generate_model
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    Dataset,
)
from data_loading_and_transformation.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader


def run_with_hyperparameter_optimization():
    config = {
        "batch_size": hp.choice("batch_size", [64]),
        "learning_rate": hp.choice("learning_rate", [0.005]),
        "num_conv_layers": hp.choice("num_conv_layers", [8, 10, 12, 14]),
        "hidden_dim": hp.choice("hidden_dim", [20]),
        "radius": hp.choice("radius", [5, 10, 15, 20, 25]),
        "max_num_node_neighbours": hp.choice(
            "max_num_node_neighbours", [5, 10, 15, 20, 25, 30]
        ),
    }

    algo = HyperOptSearch(space=config, metric="test_mae", mode="min")
    algo = ConcurrencyLimiter(searcher=algo, max_concurrent=10)

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="test_mae",
        mode="min",
        grace_period=10,
        reduction_factor=3,
    )

    reporter = CLIReporter(
        metric_columns=["train_mae", "val_mae", "test_mae", "training_iteration"]
    )

    result = tune.run(
        partial(train_validate_test_hyperopt, checkpoint_dir="./checkpoint-ray-tune"),
        resources_per_trial={"cpu": 0.5, "gpu": 0.1},
        search_alg=algo,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
    )


def run_normal():
    config = {}

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
    dataset_CuAu, dataset_FePt, dataset_FeSi = load_data(config)

    dataset_options = {
        1: Dataset.CuAu,
        2: Dataset.FePt,
        3: Dataset.CuAu_FePt_SHUFFLE,
        4: Dataset.CuAu_TRAIN_FePt_TEST,
        5: Dataset.FePt_TRAIN_CuAu_TEST,
        6: Dataset.FeSi,
    }
    print(
        "Select the dataset you want to use: 1) CuAu 2) FePt 3)Combine&Shuffle 4)CuAu-train, FePt-test 5)FePt-train, CuAu-test, 6)FeSi "
    )
    chosen_dataset_option = int(input("Selected value: "))
    config["dataset_option"] = dataset_options[chosen_dataset_option].value
    train_loader, val_loader, test_loader = dataset_splitting(
        dataset_CuAu=dataset_CuAu,
        dataset_FePt=dataset_FePt,
        dataset_FeSi=dataset_FeSi,
        batch_size=config["batch_size"],
        perc_train=config["perc_train"],
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
        + str(chosen_atom_features)
        + "-pred_val-"
        + str(chosen_prediction_value)
        + ".pk"
    )
    writer = SummaryWriter("./logs/" + model_with_config_name)

    with open("./logs/" + model_with_config_name + "/config.json", "w") as f:
        json.dump(config, f)

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


os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
type_of_run = {1: run_with_hyperparameter_optimization, 2: run_normal}

print(
    "Training and validation is conducted on first dataset and testing on the second dataset."
)
choice = int(
    input(
        "Select the type of run between hyperparameter optimization and normal run 1)Hyperopt 2)Normal: "
    )
)
type_of_run[choice]()
