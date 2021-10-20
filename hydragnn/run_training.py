##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import sys, os, json
import pickle

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import setup_ddp, get_comm_size_and_rank
from hydragnn.utils.print_utils import print_distributed
from hydragnn.models.create import create, get_device
from hydragnn.train.train_validate_test import train_validate_test


def run_training(config_file="./examples/configuration.json"):

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    world_size, world_rank = setup_ddp()

    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    verbosity = config["Verbosity"]["level"]
    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=config["Dataset"]["name"],
    )
    graph_size_variable = config["Dataset"]["variable_size"]

    output_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    output_index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]
    config["NeuralNetwork"]["Architecture"]["output_dim"] = []
    for item in range(len(output_type)):
        if output_type[item] == "graph":
            dim_item = config["Dataset"]["graph_features"]["dim"][output_index[item]]
        elif output_type[item] == "node":
            if graph_size_variable:
                if (
                    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"][
                        "type"
                    ]
                    == "mlp"
                ):
                    raise ValueError(
                        "mlp type of node feature prediction for variable graph size not yet supported",
                        graph_size_variable,
                    )
            dim_item = config["Dataset"]["node_features"]["dim"][output_index[item]]
        else:
            raise ValueError("Unknown output type", output_type[item])
        config["NeuralNetwork"]["Architecture"]["output_dim"].append(dim_item)

    if (
        "denormalize_output" in config["NeuralNetwork"]["Variables_of_interest"]
        and config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"]
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
        config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = False
    config["NeuralNetwork"]["Architecture"]["output_type"] = config["NeuralNetwork"][
        "Variables_of_interest"
    ]["type"]

    model = create(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        ),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
        verbosity_level=config["Verbosity"]["level"],
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
        + "-task_weights-"
        + "".join(
            str(weigh) + "-"
            for weigh in config["NeuralNetwork"]["Architecture"]["task_weights"]
        )
    )

    device_name, device = get_device(config["Verbosity"]["level"])
    if dist.is_initialized():
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

    print_distributed(
        verbosity,
        f"Starting training with the configuration: \n{json.dumps(config, indent=4, sort_keys=True)}",
    )

    if (
        "continue" in config["NeuralNetwork"]["Training"]
        and config["NeuralNetwork"]["Training"]["continue"] == 1
    ):  # starting from an existing model
        modelstart = config["NeuralNetwork"]["Training"]["startfrom"]
        if not modelstart:
            modelstart = model_with_config_name

        state_dict = torch.load(
            f"./logs/{modelstart}/{modelstart}.pk",
            map_location="cpu",
        )
        model.load_state_dict(state_dict)

    train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        model_with_config_name,
        verbosity,
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
