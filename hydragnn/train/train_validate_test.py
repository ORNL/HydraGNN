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
    for data in test_loader:
        node_feature.extend(data.x.tolist())
        nodes_num_list.extend(data.num_nodes_list.tolist())

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

    timer = Timer("train_validate_test")
    timer.start()

    for epoch in range(0, num_epoch):
        train_rmse, train_taskserr = train(train_loader, model, optimizer, verbosity)
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
    head_index = []
    for ihead in range(model.num_heads):
        _head_ind = []
        for isample in range(batch_size):
            istart = sum(total_size[:isample]) + y_loc[isample, ihead]
            iend = sum(total_size[:isample]) + y_loc[isample, ihead + 1]
            [_head_ind.append(ind) for ind in range(istart, iend)]
        head_index.append(_head_ind)

    return head_index


def train(loader, model, opt, verbosity):
    device = next(model.parameters()).device
    tasks_error = np.zeros(model.num_heads)

    model.train()

    total_error = 0
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)
        opt.zero_grad()
        head_index = get_head_indices(model, data)

        pred = model(data)
        loss, tasks_rmse = model.loss_rmse(pred, data.y, head_index)

        loss.backward()
        opt.step()
        total_error += loss.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
    )


@torch.no_grad()
def validate(loader, model, verbosity):

    device = next(model.parameters()).device

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    model.eval()
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)
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

    device = next(model.parameters()).device

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
        data = data.to(device)
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
