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
from functools import singledispatch

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.preprocess.utils import check_if_graph_size_constant
from hydragnn.utils.distributed import (
    setup_ddp,
    get_comm_size_and_rank,
    get_distributed_model,
    save_model,
    get_summary_writer,
)
from hydragnn.utils.print_utils import print_distributed
from hydragnn.utils.time_utils import print_timers
from hydragnn.utils.config_utils import (
    update_config_NN_outputs,
    update_config_minmax,
    get_model_output_name_config,
)
from hydragnn.models.create import create
from hydragnn.train.train_validate_test import train_validate_test


@singledispatch
def run_training(config):
    raise TypeError("Input must be filename string or configuration dictionary.")


@run_training.register
def _(config_file: str):

    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    run_training(config)


@run_training.register
def _(config: dict):

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    world_size, world_rank = setup_ddp()

    verbosity = config["Verbosity"]["level"]
    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=config["Dataset"]["name"],
    )

    graph_size_variable = check_if_graph_size_constant(
        train_loader, val_loader, test_loader
    )
    config = update_config_NN_outputs(config, graph_size_variable)

    if (
        "denormalize_output" in config["NeuralNetwork"]["Variables_of_interest"]
        and config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"]
    ):
        if "total" in config["Dataset"]["path"]["raw"].keys():
            dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}.pkl"
        else:
            ###used for min/max values loading below
            dataset_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}_train.pkl"
        config = update_config_minmax(dataset_path, config)
    else:
        config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] = False

    model = create(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        ),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
        verbosity_level=verbosity,
    )

    model_with_config_name = get_model_output_name_config(model, config)

    model = get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = get_summary_writer(model_with_config_name)

    if dist.is_initialized():
        dist.barrier()
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

    save_model(model, model_with_config_name)

    print_timers(verbosity)
