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
from hydragnn.preprocess.utils import check_if_graph_size_constant
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.time_utils import print_timers
from hydragnn.utils.config_utils import (
    update_config_NN_outputs,
    normalize_output_config,
    get_log_name_config,
)
from hydragnn.utils.model import calculate_PNA_degree
from hydragnn.models.create import create_model_config
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
    train_loader, val_loader, test_loader = dataset_loading_and_splitting(config=config)

    graph_size_variable = check_if_graph_size_constant(
        train_loader, val_loader, test_loader
    )
    config = update_config_NN_outputs(config, graph_size_variable)

    config = normalize_output_config(config)

    config["NeuralNetwork"]["Architecture"]["input_dim"] = len(
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
    )
    max_neigh = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
    if config["NeuralNetwork"]["Architecture"]["model_type"] == "PNA":
        deg = calculate_PNA_degree(train_loader.dataset, max_neigh)
    else:
        deg = None
    model = create_model_config(
        config=config["NeuralNetwork"]["Architecture"],
        num_nodes=train_loader.dataset[0].num_nodes,
        max_neighbours=max_neigh,
        pna_deg=deg,
        verbosity=config["Verbosity"]["level"],
    )

    log_name = get_log_name_config(config)
    load_existing_model(model, log_name)

    (
        error,
        error_rmse_task,
        true_values,
        predicted_values,
    ) = test(test_loader, model, config["Verbosity"]["level"])

    ##output predictions with unit/not normalized
    if config["NeuralNetwork"]["Variables_of_interest"]["denormalize_output"]:
        true_values, predicted_values = output_denormalize(
            config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"],
            true_values,
            predicted_values,
        )

    return error, error_rmse_task, true_values, predicted_values
