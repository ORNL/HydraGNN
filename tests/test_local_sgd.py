##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from hydragnn.utils.distributed import (
    configure_local_sgd,
    synchronize_local_sgd_parameters,
)
from hydragnn.utils.input_config_parsing.config_utils import (
    validate_local_sgd_config,
)


def _config(warmup_steps=1, synchronization_period=2):
    return {
        "NeuralNetwork": {
            "Architecture": {"SyncBatchNorm": False},
            "Training": {
                "LocalSGD": {
                    "enabled": True,
                    "warmup_steps": warmup_steps,
                    "synchronization_period": synchronization_period,
                }
            },
        }
    }


def _all_parameters_match(model):
    value = next(model.parameters()).detach().clone()
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return all(torch.equal(gathered[0], other) for other in gathered[1:])


def _local_sgd_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7)
        model = DDP(torch.nn.Linear(1, 1, bias=False))
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        model, optimizer = configure_local_sgd(model, base_optimizer, _config())

        synchronization = []
        for _ in range(5):
            optimizer.zero_grad()
            inputs = torch.ones(1, 1)
            targets = torch.tensor([[float(2 * rank)]])
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            synchronization.append(_all_parameters_match(model))

        # Step 0 uses global DDP gradient averaging. Step 1 is the first local
        # update and also the first periodic parameter average. Step 2 remains
        # local, step 3 performs the next parameter average, and step 4 is local.
        assert synchronization == [True, True, False, True, False]
        assert synchronize_local_sgd_parameters(optimizer)
        assert _all_parameters_match(model)
        assert not synchronize_local_sgd_parameters(optimizer)
        assert optimizer.state_dict()["step"] == 5

        parameter = next(model.parameters())
        exp_avg = optimizer.state[parameter]["exp_avg"].detach().clone()
        gathered_state = [torch.empty_like(exp_avg) for _ in range(world_size)]
        dist.all_gather(gathered_state, exp_avg)
        assert not torch.equal(gathered_state[0], gathered_state[1])
    finally:
        dist.destroy_process_group()


def test_local_sgd_reduces_gradient_sync_and_periodically_averages_parameters():
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        rendezvous_file = rendezvous.name
    try:
        mp.start_processes(
            _local_sgd_worker,
            args=(2, rendezvous_file),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
    finally:
        if os.path.exists(rendezvous_file):
            os.unlink(rendezvous_file)


def test_local_sgd_defaults_preserve_synchronous_training():
    training = {}
    validate_local_sgd_config(training)
    assert training["LocalSGD"] == {"enabled": False}


@pytest.mark.parametrize(
    "local_sgd,error,message",
    [
        (True, TypeError, "JSON object"),
        ({"enabled": 1}, TypeError, "boolean"),
        (
            {"enabled": True, "warmup_steps": -1},
            ValueError,
            "warmup_steps",
        ),
        (
            {"enabled": True, "synchronization_period": 0},
            ValueError,
            "synchronization_period",
        ),
    ],
)
def test_local_sgd_rejects_invalid_configuration(local_sgd, error, message):
    with pytest.raises(error, match=message):
        validate_local_sgd_config({"LocalSGD": local_sgd})


def test_local_sgd_rejects_deepspeed_before_distributed_setup():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="DeepSpeed"):
        configure_local_sgd(model, optimizer, _config(), use_deepspeed=True)
