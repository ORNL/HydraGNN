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

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config,
    get_log_name_config,
    parse_deepspeed_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import test
from hydragnn.postprocess.postprocess import output_denormalize

deepspeed_available = True
try:
    import deepspeed
except:
    deepspeed_available = False


@singledispatch
def run_prediction(config, use_deepspeed=False):
    raise TypeError("Input must be filename string or configuration dictionary.")


@run_prediction.register
def _(config_file: str, use_deepspeed=False):

    with open(config_file, "r") as f:
        config = json.load(f)

    run_prediction(config)


@run_prediction.register
def _(config: dict, use_deepspeed=False):

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    world_size, world_rank = setup_ddp(use_deepspeed=use_deepspeed)

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(config=config)

    config = update_config(config, train_loader, val_loader, test_loader)

    model = create_model_config(
        config=config["NeuralNetwork"], verbosity=config["Verbosity"]["level"]
    )

    if not use_deepspeed:
        model = get_distributed_model(
            model,
            config["Verbosity"]["level"],
            sync_batch_norm=config["NeuralNetwork"]["Architecture"]["SyncBatchNorm"],
        )

    else:
        assert deepspeed_available, "deepspeed package not installed"

        # create temporary deepspeed configuration
        ds_config = parse_deepspeed_config(config)

        try:
            # cannot use zero_optimization iwithout an optimizer, so we must disable it
            ds_config["zero_optimization"]["stage"] = 0
        except KeyError:
            pass

        model, _, _, _ = deepspeed.initialize(
            model=model, config=ds_config, dist_init_required=False
        )

    log_name = get_log_name_config(config)
    load_existing_model(model, log_name, use_deepspeed=use_deepspeed)

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
