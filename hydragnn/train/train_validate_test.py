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

import torch

from hydragnn.preprocess.serialized_dataset_loader import SerializedDataLoader
from hydragnn.postprocess.postprocess import output_denormalize
from hydragnn.postprocess.visualizer import Visualizer
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.profile import Profiler
from hydragnn.utils.distributed import get_device, print_peak_memory
from hydragnn.preprocess.load_data import HydraDataLoader
from hydragnn.utils.model import Checkpoint, EarlyStopping

import os

from torch.profiler import record_function
import contextlib
from unittest.mock import MagicMock
from hydragnn.utils.distributed import get_comm_size_and_rank
import torch.distributed as dist
import pickle

import hydragnn.utils.tracer as tr


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
):
    num_epoch = config["Training"]["num_epoch"]
    EarlyStop = (
        config["Training"]["EarlyStopping"]
        if "EarlyStopping" in config["Training"]
        else False
    )

    SaveCheckpoint = (
        config["Training"]["Checkpoint"]
        if "Checkpoint" in config["Training"]
        else False
    )

    device = get_device()
    # total loss tracking for train/vali/test
    total_loss_train = torch.zeros(num_epoch, device=device)
    total_loss_val = torch.zeros(num_epoch, device=device)
    total_loss_test = torch.zeros(num_epoch, device=device)
    # loss tracking for each head/task
    task_loss_train = torch.zeros((num_epoch, model.module.num_heads), device=device)
    task_loss_test = torch.zeros((num_epoch, model.module.num_heads), device=device)
    task_loss_val = torch.zeros((num_epoch, model.module.num_heads), device=device)

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
            num_heads=model.module.num_heads,
            head_dims=model.module.head_dims,
            num_nodes_list=nodes_num_list,
        )
        visualizer.num_nodes_plot()

    if create_plots and plot_init_solution:  # visualizing of initial conditions
        _, _, true_values, predicted_values = test(test_loader, model, verbosity)
        visualizer.create_scatter_plots(
            true_values,
            predicted_values,
            output_names=config["Variables_of_interest"]["output_names"],
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
        checkpoint = Checkpoint(name=model_with_config_name)
        if "checkpoint_warmup" in config["Training"]:
            checkpoint = Checkpoint(
                name=model_with_config_name,
                warmup=config["Training"]["checkpoint_warmup"],
            )

    timer = Timer("train_validate_test")
    timer.start()

    for epoch in range(0, num_epoch):
        profiler.set_current_epoch(epoch)
        for dataloader in [train_loader, val_loader, test_loader]:
            if getattr(dataloader.sampler, "set_epoch", None) is not None:
                dataloader.sampler.set_epoch(epoch)

        with profiler as prof:
            tr.enable()
            tr.start("train")
            train_loss, train_taskserr = train(
                train_loader, model, optimizer, verbosity, profiler=prof
            )
            tr.stop("train")
            tr.disable()
            if epoch == 0:
                tr.reset()

        val_loss, val_taskserr = validate(
            val_loader, model, verbosity, reduce_ranks=True
        )
        test_loss, test_taskserr, true_values, predicted_values = test(
            test_loader,
            model,
            verbosity,
            reduce_ranks=True,
            return_samples=plot_hist_solution,
        )
        scheduler.step(val_loss)
        if writer is not None:
            writer.add_scalar("train error", train_loss, epoch)
            writer.add_scalar("validate error", val_loss, epoch)
            writer.add_scalar("test error", test_loss, epoch)
            for ivar in range(model.module.num_heads):
                writer.add_scalar(
                    "train error of task" + str(ivar), train_taskserr[ivar], epoch
                )
        print_distributed(
            verbosity,
            f"Epoch: {epoch:02d}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, "
            f"Test Loss: {test_loss:.8f}",
        )
        print_distributed(
            verbosity, "Tasks Loss:", [taskerr.item() for taskerr in train_taskserr]
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
                output_names=config["Variables_of_interest"]["output_names"],
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
            test_loader, model, verbosity
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
            output_names=config["Variables_of_interest"]["output_names"],
        )
        visualizer.create_scatter_plots(
            true_values,
            predicted_values,
            output_names=config["Variables_of_interest"]["output_names"],
        )
        ######plot loss history#####
        visualizer.plot_history(
            total_loss_train,
            total_loss_val,
            total_loss_test,
            task_loss_train,
            task_loss_val,
            task_loss_test,
            model.module.loss_weights,
            config["Variables_of_interest"]["output_names"],
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
    profiler=None,
):
    if profiler is None:
        profiler = Profiler()

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(model.module.num_heads, device=get_device())
    num_samples_local = 0
    model.train()

    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    tr.start("dataload")
    if use_ddstore:
        loader.dataset.ddstore.epoch_begin()
    for data in iterate_tqdm(loader, verbosity, desc="Train"):
        if use_ddstore:
            loader.dataset.ddstore.epoch_end()
        tr.stop("dataload")
        tr.start("zero_grad")
        with record_function("zero_grad"):
            opt.zero_grad()
        tr.stop("zero_grad")
        tr.start("get_head_indices")
        with record_function("get_head_indices"):
            head_index = get_head_indices(model, data)
        tr.stop("get_head_indices")
        tr.start("forward")
        with record_function("forward"):
            data = data.to(get_device())
            pred = model(data)
            loss, tasks_loss = model.module.loss(pred, data.y, head_index)
        tr.stop("forward")
        tr.start("backward")
        with record_function("backward"):
            loss.backward()
        tr.stop("backward")
        tr.start("opt_step")
        # print_peak_memory(verbosity, "Max memory allocated before optimizer step")
        opt.step()
        # print_peak_memory(verbosity, "Max memory allocated after optimizer step")
        tr.stop("opt_step")
        profiler.step()
        with torch.no_grad():
            total_error += loss * data.num_graphs
            num_samples_local += data.num_graphs
            for itask in range(len(tasks_loss)):
                tasks_error[itask] += tasks_loss[itask] * data.num_graphs
        tr.start("dataload")
        if use_ddstore:
            loader.dataset.ddstore.epoch_begin()
    if use_ddstore:
        loader.dataset.ddstore.epoch_end()
    tr.stop("dataload")

    train_error = total_error / num_samples_local
    tasks_error = tasks_error / num_samples_local
    return train_error, tasks_error


@torch.no_grad()
def validate(loader, model, verbosity, reduce_ranks=True):

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(model.module.num_heads, device=get_device())
    num_samples_local = 0
    model.eval()
    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    if use_ddstore:
        loader.dataset.ddstore.epoch_begin()
    for data in iterate_tqdm(loader, verbosity, desc="Validate"):
        if use_ddstore:
            loader.dataset.ddstore.epoch_end()
        head_index = get_head_indices(model, data)
        data = data.to(get_device())
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
def test(loader, model, verbosity, reduce_ranks=True, return_samples=True):

    total_error = torch.tensor(0.0, device=get_device())
    tasks_error = torch.zeros(model.module.num_heads, device=get_device())
    num_samples_local = 0
    model.eval()
    use_ddstore = (
        hasattr(loader.dataset, "ddstore")
        and hasattr(loader.dataset.ddstore, "epoch_begin")
        and bool(int(os.getenv("HYDRAGNN_USE_ddstore", "0")))
    )
    if use_ddstore:
        loader.dataset.ddstore.epoch_begin()
    for data in iterate_tqdm(loader, verbosity, desc="Test"):
        if use_ddstore:
            loader.dataset.ddstore.epoch_end()
        head_index = get_head_indices(model, data)
        data = data.to(get_device())
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

    test_error = total_error / num_samples_local
    tasks_error = tasks_error / num_samples_local
    true_values = [[] for _ in range(model.module.num_heads)]
    predicted_values = [[] for _ in range(model.module.num_heads)]
    if return_samples:
        for data in loader:
            head_index = get_head_indices(model, data)
            data = data.to(get_device())
            ytrue = data.y
            pred = model(data)
            for ihead in range(model.module.num_heads):
                head_pre = pred[ihead].reshape(-1, 1)
                head_val = ytrue[head_index[ihead]]
                true_values[ihead].append(head_val)
                predicted_values[ihead].append(head_pre)
        for ihead in range(model.module.num_heads):
            predicted_values[ihead] = torch.cat(predicted_values[ihead], dim=0)
            true_values[ihead] = torch.cat(true_values[ihead], dim=0)

    if reduce_ranks:
        test_error = reduce_values_ranks(test_error)
        tasks_error = reduce_values_ranks(tasks_error)
        if len(true_values[0]) > 0:
            for ihead in range(model.module.num_heads):
                true_values[ihead] = gather_tensor_ranks(true_values[ihead])
                predicted_values[ihead] = gather_tensor_ranks(predicted_values[ihead])

    return test_error, tasks_error, true_values, predicted_values
