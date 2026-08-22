##############################################################################
# Copyright (c) 2025, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.input_config_parsing.config_utils import (
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


def load_checkpoint_and_test(config, test_loader, use_deepspeed=False):
    """Exercise checkpoint loading and testing in integration tests."""
    world_size, world_rank = setup_ddp(use_deepspeed=use_deepspeed)

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

    enable_interatomic_potential = config["NeuralNetwork"]["Architecture"].get(
        "enable_interatomic_potential", False
    )
    num_tasks = 3 if enable_interatomic_potential else model.module.num_heads

    (
        error,
        error_rmse_task,
        true_values,
        predicted_values,
    ) = test(
        test_loader,
        model,
        config["Verbosity"]["level"],
        num_tasks=num_tasks,
        compute_grad_energy=enable_interatomic_potential,
        precision=config["NeuralNetwork"]["Training"].get("precision", "fp32"),
    )

    return error, error_rmse_task, true_values, predicted_values
