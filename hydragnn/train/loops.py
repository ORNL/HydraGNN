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
"""Per-epoch training, validation, and test loops."""

import os
import pickle

import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.profiler import record_function

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.distributed import get_comm_size_and_rank, get_device, print_peak_memory
from hydragnn.loss_reporting import (
    accumulate_loss_report,
    finalize_loss_report,
    reset_loss_report,
)
from hydragnn.models.attention.gps import redraw_performer_projections
from hydragnn.train.fsdp import (
    is_fsdp2_enabled,
    set_reshard_after_backward,
    verify_fsdp2_sharding,
)
from hydragnn.train.head_indices import get_head_indices
from hydragnn.train.precision import (
    first_float_tensor_dtype,
    first_optimizer_state_dtype,
    get_autocast_and_scaler,
    move_batch_to_device,
    resolve_precision,
)
from hydragnn.train.reduction import gather_tensor_ranks, reduce_values_ranks
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.utils.profiling_and_tracing.profile import Profiler


def get_nbatch(loader):
    """Calculate the number of batches produced by a loader."""
    nbatch = len(loader)

    if os.getenv("HYDRAGNN_MAX_NUM_BATCH") is not None:
        nbatch = min(nbatch, int(os.environ["HYDRAGNN_MAX_NUM_BATCH"]))

    return nbatch


def train(
    loader,
    model,
    opt,
    verbosity,
    num_tasks=None,
    profiler=None,
    use_deepspeed=False,
    compute_grad_energy=False,
    precision="fp32",
    performer_redraw_interval=1000,
):
    if profiler is None:
        profiler = Profiler()

    precision, param_dtype, _ = resolve_precision(precision)
    autocast_context, scaler = get_autocast_and_scaler(precision)

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(num_tasks, device=get_device())
    num_samples_local = 0
    model.train()
    reset_loss_report(model.module)

    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    model_param_dtype = None
    for parameter in model.parameters():
        model_param_dtype = parameter.dtype
        break
    if rank0:
        print(
            f"[precision-diagnostic] requested={precision} model_param_dtype={model_param_dtype}"
        )
    batch_dtype_logged = False
    optimizer_state_dtype_logged = False

    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    fsdp2_force_workaround = compute_grad_energy and is_fsdp2_enabled()
    fsdp2_workaround_available = True
    verify_sharding = bool(int(os.getenv("HYDRAGNN_VERIFY_FSDP2_SHARDING", "0")))
    if verify_sharding:
        if not is_fsdp2_enabled():
            raise ValueError(
                "HYDRAGNN_VERIFY_FSDP2_SHARDING requires FSDP2 to be enabled"
            )
        verify_fsdp2_sharding(model, "parameters")

    local_nbatch = get_nbatch(loader)
    nbatch = MPI.COMM_WORLD.allreduce(local_nbatch, op=MPI.MIN)
    syncopt = {"cudasync": False}
    ## 0: default (no detailed tracing), 1: sync tracing
    trace_level = int(os.getenv("HYDRAGNN_TRACE_LEVEL", "0"))
    if trace_level > 0:
        syncopt = {"cudasync": True}
    tr.start("dataload", **syncopt)
    if use_ddstore:
        tr.start("epoch_begin")
        loader.dataset.ddstore.epoch_begin()
        tr.stop("epoch_begin")
    for ibatch, data in iterate_tqdm(
        enumerate(loader), verbosity, desc="Train", total=nbatch
    ):
        if ibatch >= nbatch:
            break
        if use_ddstore:
            tr.start("epoch_end")
            loader.dataset.ddstore.epoch_end()
            tr.stop("epoch_end")
        if trace_level > 0:
            tr.start("dataload_sync", **syncopt)
            MPI.COMM_WORLD.Barrier()
            tr.stop("dataload_sync")
        tr.stop("dataload", **syncopt)
        tr.start("zero_grad")
        with record_function("zero_grad"):
            if use_deepspeed:
                pass
            else:
                opt.zero_grad()
        tr.stop("zero_grad")
        tr.start("get_head_indices")
        if not compute_grad_energy:
            with record_function("get_head_indices"):
                head_index = get_head_indices(model, data)
        tr.stop("get_head_indices")
        tr.start("forward", **syncopt)
        with record_function("forward"):
            # Advance Performer redraw schedules once per training batch. Keep
            # this outside model.forward(), which may be replayed by activation
            # checkpointing and would otherwise advance the schedule twice.
            redraw_performer_projections(model, performer_redraw_interval)
            if trace_level > 0:
                tr.start("h2d", **syncopt)
            data = move_batch_to_device(data, param_dtype)
            if rank0 and not batch_dtype_logged:
                batch_float_dtype = first_float_tensor_dtype(data)
                print(
                    f"[precision-diagnostic] first_batch_float_dtype={batch_float_dtype}"
                )
                batch_dtype_logged = True
            if trace_level > 0:
                tr.stop("h2d", **syncopt)
            if compute_grad_energy:  # for force and energy prediction
                data.pos.requires_grad = True
                if fsdp2_force_workaround and fsdp2_workaround_available:
                    fsdp2_workaround_available = set_reshard_after_backward(
                        model, False
                    )
                # Perform forward pass and backward pass under autocast
                with autocast_context:
                    pred = model(data)
                    loss, tasks_loss = model.module.energy_force_loss(pred, data)
            else:
                # Perform forward pass and backward pass under autocast
                with autocast_context:
                    pred = model(data)
                    loss, tasks_loss = model.module.loss(pred, data.y, head_index)
            if trace_level > 0:
                tr.start("forward_sync", **syncopt)
                MPI.COMM_WORLD.Barrier()
                tr.stop("forward_sync")
        tr.stop("forward", **syncopt)
        tr.start("backward", **syncopt)
        with record_function("backward"):
            if fsdp2_force_workaround and fsdp2_workaround_available:
                set_reshard_after_backward(model, True)
            if verify_sharding and ibatch == 0:
                verify_fsdp2_sharding(model, "gradients")
            if use_deepspeed:
                model.backward(loss)
            else:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if fsdp2_force_workaround and fsdp2_workaround_available:
                set_reshard_after_backward(model, True)
            if trace_level > 0:
                tr.start("backward_sync", **syncopt)
                MPI.COMM_WORLD.Barrier()
                tr.stop("backward_sync")
        tr.stop("backward", **syncopt)
        tr.start("opt_step", **syncopt)
        # print_peak_memory(verbosity, "Max memory allocated before optimizer step")
        if use_deepspeed:
            model.step()
        else:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        if rank0 and not optimizer_state_dtype_logged:
            optimizer_state_dtype = first_optimizer_state_dtype(opt)
            print(
                f"[precision-diagnostic] first_optimizer_state_float_dtype={optimizer_state_dtype}"
            )
            optimizer_state_dtype_logged = True
        print_peak_memory(verbosity, "Max memory allocated after optimizer step")
        tr.stop("opt_step", **syncopt)
        profiler.step()
        with torch.no_grad():
            total_error += loss * data.num_graphs
            num_samples_local += data.num_graphs
            for itask in range(len(tasks_loss)):
                tasks_error[itask] += tasks_loss[itask] * data.num_graphs
            accumulate_loss_report(model.module, data.num_graphs)
        if ibatch < (nbatch - 1):
            tr.start("dataload", **syncopt)
        if use_ddstore:
            if ibatch < (nbatch - 1):
                tr.start("epoch_begin")
            loader.dataset.ddstore.epoch_begin()
            if ibatch < (nbatch - 1):
                tr.stop("epoch_begin")
    if use_ddstore:
        loader.dataset.ddstore.epoch_end()

    train_error = total_error / num_samples_local
    tasks_error = tasks_error / num_samples_local

    train_error = reduce_values_ranks(train_error)
    tasks_error = reduce_values_ranks(tasks_error)
    finalize_loss_report(model.module, get_device())

    return train_error, tasks_error


@torch.no_grad()
def validate(
    loader,
    model,
    verbosity,
    num_tasks=None,
    reduce_ranks=True,
    compute_grad_energy=False,
    precision="fp32",
):
    precision, param_dtype, _ = resolve_precision(precision)
    autocast_context, scaler = get_autocast_and_scaler(precision)

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(num_tasks, device=get_device())
    num_samples_local = 0
    model.eval()
    reset_loss_report(model.module)
    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    local_nbatch = get_nbatch(loader)
    nbatch = MPI.COMM_WORLD.allreduce(local_nbatch, op=MPI.MIN)

    if use_ddstore:
        loader.dataset.ddstore.epoch_begin()
    for ibatch, data in iterate_tqdm(
        enumerate(loader), verbosity, desc="Validate", total=nbatch
    ):
        if ibatch >= nbatch:
            break
        if use_ddstore:
            loader.dataset.ddstore.epoch_end()
        data = move_batch_to_device(data, param_dtype)
        if compute_grad_energy:  # for force and energy prediction
            with torch.enable_grad():
                data.pos.requires_grad = True
                with autocast_context:
                    pred = model(data)
                    error, tasks_loss = model.module.energy_force_loss(
                        pred, data, create_graph=False
                    )
        else:
            with autocast_context:
                head_index = get_head_indices(model, data)
                pred = model(data)
                error, tasks_loss = model.module.loss(pred, data.y, head_index)
        error = error.detach()
        if torch.is_tensor(tasks_loss):
            tasks_loss = tasks_loss.detach()
        else:
            tasks_loss = [task.detach() for task in tasks_loss]

        total_error += error * data.num_graphs
        num_samples_local += data.num_graphs
        for itask in range(len(tasks_loss)):
            tasks_error[itask] += tasks_loss[itask] * data.num_graphs
        accumulate_loss_report(model.module, data.num_graphs)
        if use_ddstore:
            loader.dataset.ddstore.epoch_begin()
    if use_ddstore:
        loader.dataset.ddstore.epoch_end()

    val_error = total_error / num_samples_local
    tasks_error = tasks_error / num_samples_local
    if reduce_ranks:
        val_error = reduce_values_ranks(val_error)
        tasks_error = reduce_values_ranks(tasks_error)
    finalize_loss_report(model.module, get_device())
    return val_error, tasks_error


@torch.no_grad()
def test(
    loader,
    model,
    verbosity,
    num_tasks=None,
    reduce_ranks=True,
    return_samples=True,
    compute_grad_energy=False,
    precision="fp32",
):
    precision, param_dtype, _ = resolve_precision(precision)
    autocast_context, scaler = get_autocast_and_scaler(precision)

    if num_tasks is None:
        num_tasks = (
            len(model.module.task_names)
            if compute_grad_energy
            else model.module.num_heads
        )

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(num_tasks, device=get_device())
    num_samples_local = 0
    model.eval()
    reset_loss_report(model.module)
    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    local_nbatch = get_nbatch(loader)
    nbatch = MPI.COMM_WORLD.allreduce(local_nbatch, op=MPI.MIN)
    _, rank = get_comm_size_and_rank()

    if int(os.getenv("HYDRAGNN_DUMP_TESTDATA", "0")) == 1:
        f = open(f"testdata_rank{rank}.pickle", "wb")
    if use_ddstore:
        loader.dataset.ddstore.epoch_begin()
    for ibatch, data in iterate_tqdm(
        enumerate(loader), verbosity, desc="Test", total=nbatch
    ):
        if ibatch >= nbatch:
            break
        if use_ddstore:
            loader.dataset.ddstore.epoch_end()
        data = move_batch_to_device(data, param_dtype)
        if compute_grad_energy:  # for force and energy prediction
            with torch.enable_grad():
                data.pos.requires_grad = True
                with autocast_context:
                    pred = model(data)
                    error, tasks_loss = model.module.energy_force_loss(
                        pred, data, create_graph=False
                    )
        else:
            with autocast_context:
                head_index = get_head_indices(model, data)
                pred = model(data)
                error, tasks_loss = model.module.loss(pred, data.y, head_index)
        ## FIXME: temporary
        if int(os.getenv("HYDRAGNN_DUMP_TESTDATA", "0")) == 1:
            if model.module.var_output:
                pred = pred[0]
            offset = 0
            for i in range(len(data)):
                n = len(data[i].pos)
                y0 = data[i].y[1:].flatten()
                y1 = pred[1][offset : offset + n].flatten()
                y2 = torch.norm(
                    data[i].y[1:].reshape(-1, 3) - pred[1][offset : offset + n, :],
                    dim=1,
                ).mean()
                data_to_save = dict()
                data_to_save["energy_true"] = data[i].y[0].detach().cpu().item()
                data_to_save["forces_true"] = y0.detach().cpu()
                data_to_save["energy_pred"] = pred[0][i].detach().cpu().item()
                data_to_save["forces_pred"] = y1.detach().cpu()
                data_to_save["forces_average_error_per_atom"] = y2.detach().cpu()
                pickle.dump(data_to_save, f)
                if rank == 0:
                    print(
                        rank,
                        ibatch,
                        i,
                        data[i].x.shape,
                        data[i].y[0].item(),
                        pred[0][i].item(),
                        y2.item(),
                    )
                offset += n

        error = error.detach()
        if torch.is_tensor(tasks_loss):
            tasks_loss = tasks_loss.detach()
        else:
            tasks_loss = [task.detach() for task in tasks_loss]

        total_error += error * data.num_graphs
        num_samples_local += data.num_graphs
        for itask in range(len(tasks_loss)):
            tasks_error[itask] += tasks_loss[itask] * data.num_graphs
        accumulate_loss_report(model.module, data.num_graphs)
        if use_ddstore:
            loader.dataset.ddstore.epoch_begin()
    if use_ddstore:
        loader.dataset.ddstore.epoch_end()

    if int(os.getenv("HYDRAGNN_DUMP_TESTDATA", "0")) == 1:
        f.close()

    test_error = total_error / num_samples_local
    tasks_error = tasks_error / num_samples_local

    true_values = [[] for _ in range(num_tasks)]
    predicted_values = [[] for _ in range(num_tasks)]

    if return_samples:
        if use_ddstore:
            loader.dataset.ddstore.epoch_begin()
        for ibatch, data in iterate_tqdm(
            enumerate(loader), verbosity, desc="Sample", total=nbatch
        ):
            if ibatch >= nbatch:
                break
            if use_ddstore:
                loader.dataset.ddstore.epoch_end()
            data = move_batch_to_device(data, param_dtype)
            if compute_grad_energy:
                with torch.enable_grad():
                    data.pos.requires_grad = True
                    with autocast_context:
                        pred = model(data)
                        values = model.module.prediction_target_pairs(
                            pred, data, create_graph=False
                        )
                        for itask, name in enumerate(model.module.task_names):
                            task_pred, task_true = values[name]
                            true_values[itask].append(task_true.reshape(-1, 1))
                            predicted_values[itask].append(task_pred.reshape(-1, 1))
            else:
                head_index = get_head_indices(model, data)
                ytrue = data.y
                pred = model(data)
                if model.module.var_output:
                    pred = pred[0]
                for ihead in range(model.module.num_heads):
                    head_val = ytrue[head_index[ihead]]
                    head_pre = pred[ihead].reshape(head_val.shape)
                    true_values[ihead].append(head_val)
                    predicted_values[ihead].append(head_pre)
            if use_ddstore:
                loader.dataset.ddstore.epoch_begin()
        if use_ddstore:
            loader.dataset.ddstore.epoch_end()
        for itask in range(
            num_tasks
        ):  ###More general for both MLIP and non-conservative model
            predicted_values[itask] = torch.cat(predicted_values[itask], dim=0)
            true_values[itask] = torch.cat(true_values[itask], dim=0)

    if reduce_ranks:
        test_error = reduce_values_ranks(test_error)
        tasks_error = reduce_values_ranks(tasks_error)
        if len(true_values[0]) > 0:
            for itask in range(num_tasks):
                true_values[itask] = gather_tensor_ranks(true_values[itask])
                predicted_values[itask] = gather_tensor_ranks(predicted_values[itask])

    finalize_loss_report(model.module, get_device())

    return test_error, tasks_error, true_values, predicted_values
