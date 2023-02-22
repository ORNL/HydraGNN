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

import os, json
from functools import singledispatch

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import (
    setup_ddp,
    get_distributed_model,
    print_peak_memory,
)
from hydragnn.utils.model import (
    save_model,
    get_summary_writer,
    load_existing_model_config,
)
from hydragnn.utils.print_utils import print_distributed, setup_log
from hydragnn.utils.time_utils import print_timers
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
    save_config,
)
from hydragnn.utils.optimizer import select_optimizer
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import train_validate_test


@singledispatch
def run_training(config):
    raise TypeError("Input must be filename string or configuration dictionary.")


@run_training.register
def _(config_file: str):

    with open(config_file, "r") as f:
        config = json.load(f)

    run_training(config)


@run_training.register
def _(config: dict):

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    setup_log(get_log_name_config(config))
    world_size, world_rank = setup_ddp()

    train_loader, val_loader, test_loader = dataset_loading_and_splitting(config=config)

    config = update_config(config, train_loader, val_loader, test_loader)
    plot_init_solution = config["Visualization"]["plot_init_solution"]
    plot_hist_solution = config["Visualization"]["plot_hist_solution"]
    create_plots = config["Visualization"]["create_plots"]

    model = create_model_config(
        config=config["NeuralNetwork"], verbosity=config["Verbosity"]["level"]
    )
    print_peak_memory(
        config["Verbosity"]["level"], "Max memory allocated after creating local model"
    )
    model = get_distributed_model(
        model,
        config["Verbosity"]["level"],
        sync_batch_norm=config["NeuralNetwork"]["Architecture"]["SyncBatchNorm"],
    )
    print_peak_memory(
        config["Verbosity"]["level"],
        "Max memory allocated after creating distributed model",
    )

    optimizer = select_optimizer(
        model, config["NeuralNetwork"]["Training"]["Optimizer"]
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

    save_config(config, log_name)

    load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    print_distributed(
        config["Verbosity"]["level"],
        f"Starting training with the configuration: \n{json.dumps(config, indent=4, sort_keys=True)}",
    )

    train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        config["Verbosity"]["level"],
        plot_init_solution,
        plot_hist_solution,
        create_plots,
    )

    save_model(model, optimizer, log_name)

    print_timers(config["Verbosity"]["level"])
