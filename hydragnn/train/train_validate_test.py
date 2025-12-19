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

from tqdm import tqdm
import numpy as np
import pdb
import torch

from hydragnn.preprocess.serialized_dataset_loader import SerializedDataLoader
from hydragnn.postprocess.postprocess import output_denormalize
from hydragnn.postprocess.visualizer import Visualizer
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.profiling_and_tracing.profile import Profiler
from hydragnn.utils.distributed import get_device, check_remaining
from hydragnn.utils.model.model import Checkpoint, EarlyStopping

import os

from torch.profiler import record_function

from hydragnn.utils.distributed import get_comm_size_and_rank, print_peak_memory
import torch.distributed as dist
import pickle

import hydragnn.utils.profiling_and_tracing.tracer as tr
import time
from mpi4py import MPI
from contextlib import nullcontext

try:
    from torch.amp import GradScaler
except (ImportError, ModuleNotFoundError):
    GradScaler = None

PRECISION_MAP = {
    "bf16": {"param_dtype": torch.float32, "autocast_dtype": torch.bfloat16},
    "fp32": {"param_dtype": torch.float32, "autocast_dtype": None},
    "fp64": {"param_dtype": torch.float64, "autocast_dtype": None},
}


def resolve_precision(precision: str):
    """Normalize precision string and return parameter/autocast dtypes."""

    if precision is None:
        precision = "fp32"
    prec = str(precision).lower()
    aliases = {
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        "float64": "fp64",
        "double": "fp64",
    }
    prec = aliases.get(prec, prec)
    if prec not in PRECISION_MAP:
        raise ValueError(
            f"Unsupported precision {precision}. Choose from {list(PRECISION_MAP.keys())}."
        )
    info = PRECISION_MAP[prec]
    return prec, info["param_dtype"], info["autocast_dtype"]


def move_batch_to_device(data, param_dtype):
    device = get_device()
    if param_dtype == torch.float64:
        return data.to(device, dtype=param_dtype)
    return data.to(device)


def get_autocast_and_scaler(precision):
    precision, _, autocast_dtype = resolve_precision(precision)

    if precision == "bf16":
        device_type = str(get_device())
        cpu_bf16 = bool(getattr(torch.backends.cpu, "has_bf16", False))
        use_bf16 = (
            torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 7
        ) or (hasattr(torch, "xpu") and torch.xpu.is_available())
        use_bf16 = use_bf16 or (device_type == "cpu" and cpu_bf16)
        if not use_bf16:
            print(
                f"Requested bf16 but unsupported on {device_type}; falling back to full precision."
            )

        autocast = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if use_bf16
            else nullcontext()
        )
        return autocast, None

    return nullcontext(), None


def get_nbatch(loader):
    ## calculate numbrer of batches for a given loader
    m = len(loader.sampler)
    nbatch = (m - 1) // loader.batch_size + 1
    extra = -1 if m - nbatch * loader.batch_size > 0 and loader.drop_last else 0
    nbatch = nbatch + extra

    if os.getenv("HYDRAGNN_MAX_NUM_BATCH") is not None:
        nbatch = min(nbatch, int(os.environ["HYDRAGNN_MAX_NUM_BATCH"]))

    return nbatch


def train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config,
    model_with_config_name,
    verbosity=0,
    plot_init_solution=True,
    plot_hist_solution=False,
    create_plots=False,
    use_deepspeed=False,
    compute_grad_energy=False,
    precision="fp32",
):
    num_epoch = config["Training"]["num_epoch"]
    EarlyStop = (
        config["Training"]["EarlyStopping"]
        if "EarlyStopping" in config["Training"]
        else False
    )

    CheckRemainingTime = (
        config["Training"]["CheckRemainingTime"]
        if "CheckRemainingTime" in config["Training"]
        else False
    )

    SaveCheckpoint = (
        config["Training"]["Checkpoint"]
        if "Checkpoint" in config["Training"]
        else False
    )

    precision, _, _ = resolve_precision(precision)

    device = get_device()
    if compute_grad_energy:
        num_tasks = 3  # [energy, energy per atom, forces]
        task_dims = [1, 1, 1]
        task_weights = [
            model.module.energy_weight,
            model.module.energy_peratom_weight,
            model.module.force_weight,
        ]
        output_names = [
            config["Variables_of_interest"]["output_names"][0],
            "energy_peratom",
            "forces",
        ]
    else:
        num_tasks = model.module.num_heads
        task_dims = model.module.head_dims
        task_weights = model.module.loss_weights
        output_names = config["Variables_of_interest"]["output_names"]

    # total loss tracking for train/vali/test
    total_loss_train = torch.zeros(num_epoch, device=device)
    total_loss_val = torch.zeros(num_epoch, device=device)
    total_loss_test = torch.zeros(num_epoch, device=device)
    # loss tracking for each head/task
    task_loss_train = torch.zeros((num_epoch, num_tasks), device=device)
    task_loss_test = torch.zeros((num_epoch, num_tasks), device=device)
    task_loss_val = torch.zeros((num_epoch, num_tasks), device=device)

    # preparing for results visualization
    ## collecting node feature
    if create_plots:
        node_feature = []
        nodes_num_list = []
        ## (2022/05) : FIXME: using test_loader.datast caused a bottleneck for large data
        for data in iterate_tqdm(
            test_loader.dataset, verbosity, desc="Collecting node feature"
        ):
            node_feature.extend(data.x.tolist())
            nodes_num_list.append(data.num_nodes)

        visualizer = Visualizer(
            model_with_config_name,
            node_feature=node_feature,
            num_heads=num_tasks,
            head_dims=task_dims,
            num_nodes_list=nodes_num_list,
        )
        visualizer.num_nodes_plot()

    if create_plots and plot_init_solution:  # visualizing of initial conditions
        _, _, true_values, predicted_values = test(
            test_loader,
            model,
            verbosity,
            compute_grad_energy=compute_grad_energy,
            num_tasks=num_tasks,
        )

        visualizer.create_scatter_plots(
            true_values,
            predicted_values,
            output_names=output_names,
            iepoch=-1,
        )

    profiler = Profiler("./logs/" + model_with_config_name)
    if "Profile" in config:
        profiler.setup(config["Profile"])

    if EarlyStop:
        earlystopper = EarlyStopping()
        if "patience" in config["Training"]:
            earlystopper = EarlyStopping(patience=config["Training"]["patience"])

    if SaveCheckpoint:
        checkpoint = Checkpoint(
            name=model_with_config_name,
            use_deepspeed=use_deepspeed,
        )
        if "checkpoint_warmup" in config["Training"]:
            checkpoint = Checkpoint(
                name=model_with_config_name,
                warmup=config["Training"]["checkpoint_warmup"],
                use_deepspeed=use_deepspeed,
            )

    timer = Timer("train_validate_test")
    timer.start()

    epoch_start = config["Training"].get("epoch_start", 0)
    for epoch in range(epoch_start, num_epoch):
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)
        ## timer per epoch
        t0 = time.time()
        profiler.set_current_epoch(epoch)
        for dataloader in [train_loader, val_loader, test_loader]:
            if getattr(dataloader.sampler, "set_epoch", None) is not None:
                dataloader.sampler.set_epoch(epoch)

        with profiler as prof:
            tr.enable()
            tr.start("train")
            train_loss, train_taskserr = train(
                train_loader,
                model,
                optimizer,
                verbosity,
                num_tasks=num_tasks,
                profiler=prof,
                use_deepspeed=use_deepspeed,
                compute_grad_energy=compute_grad_energy,
                precision=precision,
            )
            tr.stop("train")
            tr.disable()
            if epoch == 0:
                tr.reset()

        if int(os.getenv("HYDRAGNN_VALTEST", "1")) == 0:
            continue

        val_loss, val_taskserr = validate(
            val_loader,
            model,
            verbosity,
            num_tasks=num_tasks,
            reduce_ranks=True,
            compute_grad_energy=compute_grad_energy,
            precision=precision,
        )
        test_loss, test_taskserr, true_values, predicted_values = test(
            test_loader,
            model,
            verbosity,
            num_tasks=num_tasks,
            reduce_ranks=True,
            return_samples=plot_hist_solution,
            compute_grad_energy=compute_grad_energy,
            precision=precision,
        )
        scheduler.step(val_loss)
        if writer is not None:
            writer.add_scalar("train error", train_loss, epoch)
            writer.add_scalar("validate error", val_loss, epoch)
            writer.add_scalar("test error", test_loss, epoch)
            for ivar in range(num_tasks):
                writer.add_scalar(
                    "train error of task" + str(ivar), train_taskserr[ivar], epoch
                )
        print_distributed(
            verbosity,
            f"Epoch: {epoch:02d}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, "
            f"Test Loss: {test_loss:.8f}",
        )
        print_distributed(
            verbosity,
            "Tasks Train Loss:",
            [taskerr.item() for taskerr in train_taskserr],
        )
        print_distributed(
            verbosity, "Tasks Val Loss:", [taskerr.item() for taskerr in val_taskserr]
        )
        print_distributed(
            verbosity, "Tasks Test Loss:", [taskerr.item() for taskerr in test_taskserr]
        )

        total_loss_train[epoch] = train_loss
        total_loss_val[epoch] = val_loss
        total_loss_test[epoch] = test_loss
        task_loss_train[epoch, :] = train_taskserr
        task_loss_val[epoch, :] = val_taskserr
        task_loss_test[epoch, :] = test_taskserr

        ###tracking the solution evolving with training
        if plot_hist_solution:
            visualizer.create_scatter_plots(
                true_values,
                predicted_values,
                output_names=output_names,
                iepoch=epoch,
            )

        if SaveCheckpoint:
            if checkpoint(model, optimizer, reduce_values_ranks(val_loss).item()):
                print_distributed(
                    verbosity, "Creating Checkpoint: %f" % checkpoint.min_perf_metric
                )
            print_distributed(
                verbosity, "Best Performance Metric: %f" % checkpoint.min_perf_metric
            )

        if EarlyStop:
            if earlystopper(reduce_values_ranks(val_loss)):
                print_distributed(
                    verbosity,
                    "Early stopping executed at epoch = %d due to val_loss not decreasing"
                    % epoch,
                )
                break

        if CheckRemainingTime:
            should_stop = check_remaining(t0)
            if should_stop:
                print_distributed(
                    verbosity,
                    "No time left. Early stop.",
                )
                break

    timer.stop()

    if create_plots:
        # reduce loss statistics across all processes
        total_loss_train = reduce_values_ranks(total_loss_train)
        total_loss_val = reduce_values_ranks(total_loss_val)
        total_loss_test = reduce_values_ranks(total_loss_test)
        task_loss_train = reduce_values_ranks(task_loss_train)
        task_loss_val = reduce_values_ranks(task_loss_val)
        task_loss_test = reduce_values_ranks(task_loss_test)

        # At the end of training phase, do the one test run for visualizer to get latest predictions
        test_loss, test_taskserr, true_values, predicted_values = test(
            test_loader,
            model,
            verbosity,
            precision=precision,
            compute_grad_energy=compute_grad_energy,
            num_tasks=num_tasks,
        )

        ##output predictions with unit/not normalized
        if config["Variables_of_interest"]["denormalize_output"]:
            true_values, predicted_values = output_denormalize(
                config["Variables_of_interest"]["y_minmax"],
                true_values,
                predicted_values,
            )

    _, rank = get_comm_size_and_rank()
    if create_plots and rank == 0:
        ######result visualization######
        visualizer.create_plot_global(
            true_values,
            predicted_values,
            output_names=output_names,
        )
        visualizer.create_scatter_plots(
            true_values,
            predicted_values,
            output_names=output_names,
        )
        ######plot loss history#####
        visualizer.plot_history(
            total_loss_train,
            total_loss_val,
            total_loss_test,
            task_loss_train,
            task_loss_val,
            task_loss_test,
            task_weights,
            output_names,
        )


def get_head_indices(model, data):
    """In data.y (the true value here), all feature variables for a mini-batch are concatenated together as a large list.
    To calculate loss function, we need to know true value for each feature in every head.
    This function is to get the feature/head index/location in the large list."""
    if all(ele == "graph" for ele in model.module.head_type):
        return get_head_indices_graph(model, data)
    else:
        return get_head_indices_node_or_mixed(model, data)


def get_head_indices_graph(model, data):
    """this is for cases when outputs are all at graph level"""
    # total length
    nsize = data.y.shape[0]
    # feature index for all heads
    head_index = [None] * model.module.num_heads
    if model.module.num_heads == 1:
        head_index[0] = torch.arange(nsize)
        return head_index
    # dimensions of all heads
    head_dims = model.module.head_dims
    head_dimsum = sum(head_dims)

    batch_size = data.batch.max() + 1
    for ihead in range(model.module.num_heads):
        head_each = torch.arange(head_dims[ihead])
        head_ind_temporary = head_each.repeat(batch_size)
        head_shift_temporary = sum(head_dims[:ihead]) + torch.repeat_interleave(
            torch.arange(batch_size) * head_dimsum, head_dims[ihead]
        )
        head_index[ihead] = head_ind_temporary + head_shift_temporary
    return head_index


def get_head_indices_node_or_mixed(model, data):
    """this is for cases when outputs are node level or mixed graph-node level"""
    batch_size = data.batch.max() + 1
    y_loc = data.y_loc
    # head size for each sample
    total_size = y_loc[:, -1]
    # feature index for all heads
    head_index = [None] * model.module.num_heads
    if model.module.num_heads == 1:
        head_index[0] = torch.arange(data.y.shape[0])
        return head_index
    # intermediate work list
    head_ind_temporary = [None] * batch_size
    # track the start loc of each sample
    sample_start = torch.cumsum(total_size, dim=0) - total_size
    sample_start = sample_start.view(-1, 1)
    # shape (batch_size, model.module.num_heads), start and end of each head for each sample
    start_index = sample_start + y_loc[:, :-1]
    end_index = sample_start + y_loc[:, 1:]

    # a large index tensor pool for all element in data.y
    index_range = torch.arange(0, end_index[-1, -1], device=y_loc.device)
    for ihead in range(model.module.num_heads):
        for isample in range(batch_size):
            head_ind_temporary[isample] = index_range[
                start_index[isample, ihead] : end_index[isample, ihead]
            ]
        head_index[ihead] = torch.cat(head_ind_temporary, dim=0)

    return head_index


@torch.no_grad()
def reduce_values_ranks_dist(local_tensor):
    if dist.get_world_size() > 1:
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        local_tensor = local_tensor / dist.get_world_size()
    return local_tensor


@torch.no_grad()
def reduce_values_ranks_mpi(local_tensor):
    if dist.get_world_size() > 1:
        from mpi4py import MPI

        local_tensor = MPI.COMM_WORLD.allreduce(
            local_tensor.detach().cpu().numpy(), op=MPI.SUM
        )
        local_tensor = torch.tensor(local_tensor) / dist.get_world_size()
    return local_tensor


def reduce_values_ranks(local_tensor):
    backend = os.getenv("HYDRAGNN_AGGR_BACKEND", "torch")
    if backend == "mpi":
        return reduce_values_ranks_mpi(local_tensor)
    else:
        return reduce_values_ranks_dist(local_tensor)


@torch.no_grad()
def gather_tensor_ranks(head_values):
    if dist.get_world_size() > 1:
        head_values = head_values.to(get_device())
        size_local = torch.tensor(
            [head_values.shape[0]], dtype=torch.int64, device=head_values.device
        )
        size_all = [torch.ones_like(size_local) for _ in range(dist.get_world_size())]
        dist.all_gather(size_all, size_local)
        size_all = torch.cat(size_all, 0)
        max_size = size_all.max()

        padded = torch.empty(
            max_size,
            *head_values.shape[1:],
            dtype=head_values.dtype,
            device=head_values.device,
        )
        padded[: head_values.shape[0]] = head_values

        tensor_list = [torch.ones_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, padded)
        tensor_list = torch.cat(tensor_list, 0)

        head_values = torch.zeros(
            size_all.sum(),
            *head_values.shape[1:],
            dtype=head_values.dtype,
            device=head_values.device,
        )
        for i, size in enumerate(size_all):
            start_idx = i * max_size
            end_idx = start_idx + size.item()
            if end_idx > start_idx:
                head_values[
                    size_all[:i].sum() : size_all[:i].sum() + size.item()
                ] = tensor_list[start_idx:end_idx]

    return head_values


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
):
    if profiler is None:
        profiler = Profiler()

    precision, param_dtype, _ = resolve_precision(precision)
    autocast_context, scaler = get_autocast_and_scaler(precision)

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(num_tasks, device=get_device())
    num_samples_local = 0
    model.train()

    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )

    nbatch = get_nbatch(loader)
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
            if trace_level > 0:
                tr.start("h2d", **syncopt)
            data = move_batch_to_device(data, param_dtype)
            if trace_level > 0:
                tr.stop("h2d", **syncopt)
            if compute_grad_energy:  # for force and energy prediction
                data.pos.requires_grad = True
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
            if use_deepspeed:
                model.backward(loss)
            else:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
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
        print_peak_memory(verbosity, "Max memory allocated after optimizer step")
        tr.stop("opt_step", **syncopt)
        profiler.step()
        with torch.no_grad():
            total_error += loss * data.num_graphs
            num_samples_local += data.num_graphs
            for itask in range(len(tasks_loss)):
                tasks_error[itask] += tasks_loss[itask] * data.num_graphs
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
    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    nbatch = get_nbatch(loader)

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
                    error, tasks_loss = model.module.energy_force_loss(pred, data)
        else:
            with autocast_context:
                head_index = get_head_indices(model, data)
                pred = model(data)
                error, tasks_loss = model.module.loss(pred, data.y, head_index)
        total_error += error * data.num_graphs
        num_samples_local += data.num_graphs
        for itask in range(len(tasks_loss)):
            tasks_error[itask] += tasks_loss[itask] * data.num_graphs
        if use_ddstore:
            loader.dataset.ddstore.epoch_begin()
    if use_ddstore:
        loader.dataset.ddstore.epoch_end()

    val_error = total_error / num_samples_local
    tasks_error = tasks_error / num_samples_local
    if reduce_ranks:
        val_error = reduce_values_ranks(val_error)
        tasks_error = reduce_values_ranks(tasks_error)
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

    if compute_grad_energy:
        import torch_scatter

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(num_tasks, device=get_device())
    num_samples_local = 0
    model.eval()
    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    nbatch = get_nbatch(loader)
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
                    error, tasks_loss = model.module.energy_force_loss(pred, data)
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

        total_error += error * data.num_graphs
        num_samples_local += data.num_graphs
        for itask in range(len(tasks_loss)):
            tasks_error[itask] += tasks_loss[itask] * data.num_graphs
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
                        # Support both node and graph heads; enforce sum pooling for graph heads
                        if model.module.head_type[0] == "node":
                            node_energy_pred = pred[0]
                            graph_energy_pred = (
                                torch_scatter.scatter_add(
                                    node_energy_pred, data.batch, dim=0
                                )
                                .squeeze()
                                .float()
                            )
                        elif model.module.head_type[0] == "graph":
                            if getattr(
                                model.module.model, "graph_pooling", "mean"
                            ) not in ["add"]:
                                raise ValueError(
                                    "Graph head force loss requires sum pooling (graph_pooling='add')."
                                )
                            if isinstance(pred, dict) and "graph" in pred:
                                graph_energy_pred = pred["graph"][0].squeeze().float()
                            elif isinstance(pred, (list, tuple)):
                                graph_energy_pred = pred[0].squeeze().float()
                            else:
                                graph_energy_pred = pred.squeeze().float()
                        else:
                            raise ValueError(
                                "Force predictions are only supported for node or graph energy heads."
                            )

                        graph_energy_true = data.energy.squeeze().float()

                        ncount = torch.bincount(data.batch)
                        graph_energy_peratom_pred = graph_energy_pred / ncount
                        graph_energy_peratom_true = graph_energy_true / ncount

                        forces_true = data.forces.float()
                        forces_pred = torch.autograd.grad(
                            graph_energy_pred,
                            data.pos,
                            grad_outputs=torch.ones_like(graph_energy_pred),
                            retain_graph=graph_energy_pred.requires_grad,
                            # Retain graph only if needed (it will be needed during training, but not during validation/testing)
                            create_graph=True,
                        )[0].float()
                        assert (
                            forces_pred is not None
                        ), "No gradients were found for data.pos. Does your model use positions for prediction?"
                        forces_pred = -forces_pred
                        forces_true = forces_true.flatten()
                        forces_pred = forces_pred.flatten()
                        true_values[0].append(graph_energy_true.reshape(-1, 1))
                        true_values[1].append(graph_energy_peratom_true.reshape(-1, 1))
                        true_values[2].append(forces_true.reshape(-1, 1))
                        predicted_values[0].append(graph_energy_pred.reshape(-1, 1))
                        predicted_values[1].append(
                            graph_energy_peratom_pred.reshape(-1, 1)
                        )
                        predicted_values[2].append(forces_pred.reshape(-1, 1))
            else:
                head_index = get_head_indices(model, data)
                ytrue = data.y
                pred = model(data)
                if model.module.var_output:
                    pred = pred[0]
                for ihead in range(model.module.num_heads):
                    head_pre = pred[ihead].reshape(-1, 1)
                    head_val = ytrue[head_index[ihead]]
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

    return test_error, tasks_error, true_values, predicted_values
