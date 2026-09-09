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

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hydragnn.utils.distributed import (
    configure_local_sgd,
    setup_ddp,
    synchronize_local_sgd_parameters,
)
from hydragnn.utils.input_config_parsing.config_utils import (
    validate_local_sgd_config,
)


def _config(
    warmup_steps=1,
    synchronization_period=2,
    optimizer_state_policy="local",
    optimizer_state_bucket_bytes=25 * 1024 * 1024,
):
    return {
        "NeuralNetwork": {
            "Architecture": {"SyncBatchNorm": False},
            "Training": {
                "LocalSGD": {
                    "enabled": True,
                    "warmup_steps": warmup_steps,
                    "synchronization_period": synchronization_period,
                    "optimizer_state_policy": optimizer_state_policy,
                    "optimizer_state_bucket_bytes": optimizer_state_bucket_bytes,
                }
            },
        }
    }


def _all_parameters_match(model):
    value = next(model.parameters()).detach().clone()
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return all(torch.equal(gathered[0], other) for other in gathered[1:])


def _local_sgd_worker(rank, world_size, rendezvous_file=None):
    if rendezvous_file is not None:
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
        for step in range(5):
            optimizer.zero_grad()
            inputs = torch.ones(1, 1)
            targets = torch.tensor([[float(2 * rank)]])
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            synchronization.append(_all_parameters_match(model))
            if step == 0:
                # The first step is still in globally synchronized DDP warm-up;
                # epoch-boundary handling must not add a parameter collective.
                assert not synchronize_local_sgd_parameters(optimizer)

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
        # Standalone callers own their temporary process group. MPI tests use
        # HydraGNN's normal setup_ddp lifecycle and leave the shared group up.
        if rendezvous_file is not None:
            dist.destroy_process_group()


def _synchronized_optimizer_state_worker(rank, world_size, rendezvous_file=None):
    if rendezvous_file is not None:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_file}",
            rank=rank,
            world_size=world_size,
        )
    try:
        optimizers = (
            (torch.optim.SGD, {"momentum": 0.9}),
            (torch.optim.Adam, {}),
            (torch.optim.AdamW, {}),
            (torch.optim.Adamax, {}),
            (torch.optim.Adagrad, {}),
            (torch.optim.Adadelta, {}),
            (torch.optim.RMSprop, {"momentum": 0.9, "centered": True}),
        )
        reductions = {
            torch.optim.SGD: {"momentum_buffer": "mean"},
            torch.optim.Adam: {"exp_avg": "mean", "exp_avg_sq": "mean"},
            torch.optim.AdamW: {"exp_avg": "mean", "exp_avg_sq": "mean"},
            torch.optim.Adamax: {"exp_avg": "mean", "exp_inf": "max"},
            torch.optim.Adagrad: {"sum": "mean"},
            torch.optim.Adadelta: {"square_avg": "mean", "acc_delta": "mean"},
            torch.optim.RMSprop: {
                "square_avg": "mean",
                "momentum_buffer": "mean",
                "grad_avg": "mean",
            },
        }
        for optimizer_type, kwargs in optimizers:
            torch.manual_seed(11)
            model = DDP(torch.nn.Linear(2, 1, bias=False))
            base_optimizer = optimizer_type(model.parameters(), lr=0.1, **kwargs)
            reference_parameter = torch.nn.Parameter(
                next(model.parameters()).detach().clone()
            )
            reference_optimizer = optimizer_type(
                [reference_parameter], lr=0.1, **kwargs
            )
            model, optimizer = configure_local_sgd(
                model,
                base_optimizer,
                _config(
                    warmup_steps=0,
                    synchronization_period=1,
                    optimizer_state_policy="synchronize",
                    # A sub-element request is clamped to one element and still
                    # exercises multiple bounded buckets for this tiny model.
                    optimizer_state_bucket_bytes=1,
                ),
            )

            optimizer.zero_grad()
            inputs = torch.tensor([[1.0, 2.0]])
            targets = torch.tensor([[float(3 * rank)]])
            reference_optimizer.zero_grad()
            torch.nn.functional.mse_loss(
                inputs @ reference_parameter.t(), targets
            ).backward()
            reference_optimizer.step()
            torch.nn.functional.mse_loss(model(inputs), targets).backward()
            optimizer.step()

            assert _all_parameters_match(model)
            parameter = next(model.parameters())
            for key, value in optimizer.state[parameter].items():
                if not torch.is_tensor(value):
                    continue
                gathered = [torch.empty_like(value) for _ in range(world_size)]
                dist.all_gather(gathered, value)
                assert all(torch.equal(gathered[0], other) for other in gathered[1:]), (
                    optimizer_type.__name__,
                    key,
                )
                reference_value = reference_optimizer.state[reference_parameter][key]
                references = [
                    torch.empty_like(reference_value) for _ in range(world_size)
                ]
                dist.all_gather(references, reference_value)
                if key == "step":
                    expected = references[0]
                elif reductions[optimizer_type][key] == "mean":
                    expected = torch.stack(references).mean(dim=0)
                else:
                    expected = torch.stack(references).max(dim=0).values
                torch.testing.assert_close(value, expected)

        model = DDP(torch.nn.Linear(1, 1))
        unsupported = torch.optim.LBFGS(model.parameters())
        with pytest.raises(ValueError, match="LBFGS"):
            configure_local_sgd(
                model,
                unsupported,
                _config(optimizer_state_policy="synchronize"),
            )

        model = DDP(torch.nn.Linear(1, 1, bias=False))
        base_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        parameter = next(model.parameters())
        base_optimizer.state[parameter].update(
            {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
                "unknown_history": torch.zeros_like(parameter),
            }
        )
        model, optimizer = configure_local_sgd(
            model,
            base_optimizer,
            _config(
                warmup_steps=0,
                synchronization_period=1,
                optimizer_state_policy="synchronize",
            ),
        )
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        with pytest.raises(ValueError, match="unknown_history"):
            optimizer.step()
    finally:
        if rendezvous_file is not None:
            dist.destroy_process_group()


@pytest.mark.mpi
def test_local_sgd_reduces_gradient_sync_and_periodically_averages_parameters():
    world_size, rank = setup_ddp()
    assert world_size == 2
    _local_sgd_worker(rank, world_size)


@pytest.mark.mpi
def test_local_sgd_synchronizes_supported_optimizer_state_policies():
    world_size, rank = setup_ddp()
    assert world_size == 2
    _synchronized_optimizer_state_worker(rank, world_size)


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
        (
            {"enabled": True, "optimizer_state_policy": "copy_rank_zero"},
            ValueError,
            "optimizer_state_policy",
        ),
        (
            {"enabled": True, "optimizer_state_bucket_bytes": 0},
            ValueError,
            "optimizer_state_bucket_bytes",
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
