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

import json, os
from functools import singledispatch

import torch

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.time_utils import print_timers
from hydragnn.utils.function_utils import (
    check_if_graph_size_constant,
    update_config_NN_outputs,
    get_model_output_name,
)
from hydragnn.models.create import create
from hydragnn.train.train_validate_test import test
from hydragnn.postprocess.postprocess import output_denormalize


@singledispatch
def run_prediction(config):
    raise TypeError("Input must be filename string or configuration dictionary.")


@run_prediction.register
def _(config_file: str):

    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    run_prediction(config)


@run_prediction.register
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

    model = create(
        model_type=config["NeuralNetwork"]["Architecture"]["model_type"],
        input_dim=len(
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        ),
        dataset=train_loader.dataset,
        config=config["NeuralNetwork"]["Architecture"],
        verbosity_level=config["Verbosity"]["level"],
    )

    model_with_config_name = get_model_output_name(model, config)
    state_dict = torch.load(
        "./logs/" + model_with_config_name + "/" + model_with_config_name + ".pk",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)

    (
        error,
        error_sumofnodes_task,
        error_rmse_task,
        true_values,
        predicted_values,
    ) = test(test_loader, model, config["Verbosity"]["level"])

    ##output predictions with unit/not normalized
    if config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"] == "True":
        true_values, predicted_values = output_denormalize(
            config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"],
            true_values,
            predicted_values,
        )

    return error, error_sumofnodes_task, error_rmse_task, true_values, predicted_values
