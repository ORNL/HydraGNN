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
from hydragnn.utils.model import get_model_or_module
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.profile import Profiler

import os

from torch.profiler import record_function
import contextlib
from unittest.mock import MagicMock


def train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    writer,
    scheduler,
    config,
    model_with_config_name,
    verbosity=0,
    plot_init_solution=True,
    plot_hist_solution=False,
):
    num_epoch = config["Training"]["num_epoch"]
    # total loss tracking for train/vali/test
    total_loss_train = [None] * num_epoch
    total_loss_val = [None] * num_epoch
    total_loss_test = [None] * num_epoch
    # loss tracking for each head/task
    task_loss_train = [None] * num_epoch
    task_loss_test = [None] * num_epoch
    task_loss_val = [None] * num_epoch

    model = get_model_or_module(model)

    # preparing for results visualization
    ## collecting node feature
    node_feature = []
    nodes_num_list = []
    for data in test_loader.dataset:
        node_feature.extend(data.x.tolist())
        nodes_num_list.append(data.num_nodes)

    visualizer = Visualizer(
        model_with_config_name,
        node_feature=node_feature,
        num_heads=model.num_heads,
        head_dims=model.head_dims,
        num_nodes_list=nodes_num_list,
    )
    visualizer.num_nodes_plot()

    if plot_init_solution:  # visualizing of initial conditions
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

    timer = Timer("train_validate_test")
    timer.start()

    for epoch in range(0, num_epoch):
        profiler.set_current_epoch(epoch)
        for sampler in sampler_list:
            sampler.set_epoch(epoch)

        with profiler as prof:
            train_rmse, train_taskserr = train(
                train_loader, model, optimizer, verbosity, profiler=prof
            )
        val_rmse, val_taskserr = validate(val_loader, model, verbosity)
        test_rmse, test_taskserr, true_values, predicted_values = test(
            test_loader, model, verbosity
        )
        scheduler.step(val_rmse)
        if writer is not None:
            writer.add_scalar("train error", train_rmse, epoch)
            writer.add_scalar("validate error", val_rmse, epoch)
            writer.add_scalar("test error", test_rmse, epoch)
            for ivar in range(model.num_heads):
                writer.add_scalar(
                    "train error of task" + str(ivar), train_taskserr[ivar], epoch
                )
        print_distributed(
            verbosity,
            f"Epoch: {epoch:02d}, Train RMSE: {train_rmse:.8f}, Val RMSE: {val_rmse:.8f}, "
            f"Test RMSE: {test_rmse:.8f}",
        )
        print_distributed(verbosity, "Tasks RMSE:", train_taskserr)

        total_loss_train[epoch] = train_rmse
        total_loss_val[epoch] = val_rmse
        total_loss_test[epoch] = test_rmse
        task_loss_train[epoch] = train_taskserr
        task_loss_val[epoch] = val_taskserr
        task_loss_test[epoch] = test_taskserr

        ###tracking the solution evolving with training
        if plot_hist_solution:
            visualizer.create_scatter_plots(
                true_values,
                predicted_values,
                output_names=config["Variables_of_interest"]["output_names"],
                iepoch=epoch,
            )

    timer.stop()

    # At the end of training phase, do the one test run for visualizer to get latest predictions
    test_rmse, test_taskserr, true_values, predicted_values = test(
        test_loader, model, verbosity
    )

    ##output predictions with unit/not normalized
    if config["Variables_of_interest"]["denormalize_output"]:
        true_values, predicted_values = output_denormalize(
            config["Variables_of_interest"]["y_minmax"], true_values, predicted_values
        )

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
        model.loss_weights,
        config["Variables_of_interest"]["output_names"],
    )


def get_head_indices(model, data):
    """In data.y (the true value here), all feature variables for a mini-batch are concatenated together as a large list.
    To calculate loss function, we need to know true value for each feature in every head.
    This function is to get the feature/head index/location in the large list."""
    batch_size = data.batch.max() + 1
    y_loc = data.y_loc
    # head size for each sample
    total_size = y_loc[:, -1]
    # feature index for all heads
    head_index = [None] * model.num_heads
    # intermediate work list
    head_ind_temporary = [None] * batch_size
    # track the start loc of each sample
    sample_start = torch.cumsum(total_size, dim=0) - total_size
    sample_start = sample_start.view(-1, 1)
    # shape (batch_size, model.num_heads), start and end of each head for each sample
    start_index = sample_start + y_loc[:, :-1]
    end_index = sample_start + y_loc[:, 1:]

    # a large index tensor pool for all element in data.y
    index_range = torch.arange(0, end_index[-1, -1], device=y_loc.device)
    for ihead in range(model.num_heads):
        for isample in range(batch_size):
            head_ind_temporary[isample] = index_range[
                start_index[isample, ihead] : end_index[isample, ihead]
            ]
        head_index[ihead] = torch.cat(head_ind_temporary, dim=0)

    return head_index


def train(
    loader,
    model,
    opt,
    verbosity,
    profiler=Profiler(),
):
    tasks_error = np.zeros(model.num_heads)

    model.train()

    total_error = 0
    for data in iterate_tqdm(loader, verbosity):
        with record_function("zero_grad"):
            opt.zero_grad()
        with record_function("get_head_indices"):
            head_index = get_head_indices(model, data)
        with record_function("forward"):
            pred = model(data)
            loss, tasks_rmse = model.loss_rmse(pred, data.y, head_index)
        with record_function("backward"):
            loss.backward()
        opt.step()
        profiler.step()
        total_error += loss.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
    )


@torch.no_grad()
def validate(loader, model, verbosity):

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    model.eval()
    for data in iterate_tqdm(loader, verbosity):
        head_index = get_head_indices(model, data)

        pred = model(data)
        error, tasks_rmse = model.loss_rmse(pred, data.y, head_index)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
    )


@torch.no_grad()
def test(loader, model, verbosity):

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    model.eval()
    true_values = [[] for _ in range(model.num_heads)]
    predicted_values = [[] for _ in range(model.num_heads)]
    IImean = [i for i in range(sum(model.head_dims))]
    if model.ilossweights_nll == 1:
        IImean = [i for i in range(sum(model.head_dims) + model.num_heads)]
        [
            IImean.remove(sum(model.head_dims[: ihead + 1]) + (ihead + 1) * 1 - 1)
            for ihead in range(model.num_heads)
        ]
    for data in iterate_tqdm(loader, verbosity):
        head_index = get_head_indices(model, data)

        pred = model(data)
        error, tasks_rmse = model.loss_rmse(pred, data.y, head_index)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
        ytrue = data.y
        istart = 0
        for ihead in range(model.num_heads):
            head_pre = pred[ihead]
            pred_shape = head_pre.shape
            iend = istart + pred_shape[0] * pred_shape[1]
            head_val = ytrue[head_index[ihead]]
            istart = iend
            true_values[ihead].extend(head_val.tolist())
            predicted_values[ihead].extend(pred[ihead].tolist())

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        true_values,
        predicted_values,
    )
