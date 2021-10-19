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
from hydragnn.postprocess.visualizer import Visualizer
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm


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
    total_loss_train = []
    total_loss_val = []
    total_loss_test = []
    # loss tracking of summation across all nodes for node feature predictions
    task_loss_train_sum = []
    task_loss_test_sum = []
    task_loss_val_sum = []
    # loss tracking for each head/task
    task_loss_train = []
    task_loss_test = []
    task_loss_val = []

    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model.module
    else:
        model = model
    # preparing for results visualization
    ## collecting node feature
    node_feature = []
    for data in test_loader.dataset:
        node_feature.append(data.x)
    visualizer = Visualizer(
        model_with_config_name,
        node_feature=node_feature,
        num_heads=model.num_heads,
        head_dims=model.head_dims,
    )

    if plot_init_solution:  # visualizing of initial conditions
        test_rmse = test(test_loader, model, verbosity)
        true_values = test_rmse[3]
        predicted_values = test_rmse[4]
        visualizer.create_scatter_plots(
            true_values,
            predicted_values,
            output_names=config["Variables_of_interest"]["output_names"],
            iepoch=-1,
        )
    for epoch in range(0, num_epoch):
        train_mae, train_taskserr, train_taskserr_nodes = train(
            train_loader, model, optimizer, verbosity
        )
        val_mae, val_taskserr, val_taskserr_nodes = validate(
            val_loader, model, verbosity
        )
        test_rmse = test(test_loader, model, verbosity)
        scheduler.step(val_mae)
        if writer is not None:
            writer.add_scalar("train error", train_mae, epoch)
            writer.add_scalar("validate error", val_mae, epoch)
            writer.add_scalar("test error", test_rmse[0], epoch)
            for ivar in range(model.num_heads):
                writer.add_scalar(
                    "train error of task" + str(ivar), train_taskserr[ivar], epoch
                )
        print_distributed(
            verbosity,
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}, "
            f"Test RMSE: {test_rmse[0]:.8f}",
        )
        print_distributed(verbosity, "Tasks MAE:", train_taskserr)

        total_loss_train.append(train_mae)
        total_loss_val.append(val_mae)
        total_loss_test.append(test_rmse[0])
        task_loss_train_sum.append(train_taskserr)
        task_loss_val_sum.append(val_taskserr)
        task_loss_test_sum.append(test_rmse[1])

        task_loss_train.append(train_taskserr_nodes)
        task_loss_val.append(val_taskserr_nodes)
        task_loss_test.append(test_rmse[2])

        ###tracking the solution evolving with training
        if plot_hist_solution:
            true_values = test_rmse[3]
            predicted_values = test_rmse[4]
            visualizer.create_scatter_plots(
                true_values,
                predicted_values,
                output_names=config["Variables_of_interest"]["output_names"],
                iepoch=epoch,
            )

    # At the end of training phase, do the one test run for visualizer to get latest predictions
    test_rmse, test_taskserr, test_taskserr_nodes, true_values, predicted_values = test(
        test_loader, model, verbosity
    )

    ##output predictions with unit/not normalized
    if config["Variables_of_interest"]["denormalize_output"] == "True":
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
        task_loss_train_sum,
        task_loss_val_sum,
        task_loss_test_sum,
        task_loss_train,
        task_loss_val,
        task_loss_test,
        model.loss_weights,
        config["Variables_of_interest"]["output_names"],
    )


def train(loader, model, opt, verbosity):

    device = next(model.parameters()).device
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)

    model.train()

    total_error = 0
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)
        opt.zero_grad()

        pred = model(data)
        loss, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)

        loss.backward()
        opt.step()
        total_error += loss.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs
    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
    )


@torch.no_grad()
def validate(loader, model, verbosity):

    device = next(model.parameters()).device

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
    model.eval()
    for data in iterate_tqdm(loader, verbosity):
        data = data.to(device)

        pred = model(data)
        error, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
    )


@torch.no_grad()
def test(loader, model, verbosity):

    device = next(model.parameters()).device

    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
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

        pred = model(data)
        error, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs

        ytrue = torch.reshape(data.y, (-1, sum(model.head_dims)))
        for ihead in range(model.num_heads):
            isum = sum(model.head_dims[: ihead + 1])
            true_values[ihead].extend(
                ytrue[:, isum - model.head_dims[ihead] : isum].tolist()
            )
            predicted_values[ihead].extend(
                pred[:, IImean[isum - model.head_dims[ihead] : isum]].tolist()
            )

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
        true_values,
        predicted_values,
    )
