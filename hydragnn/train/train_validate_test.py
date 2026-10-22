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
"""Top-level driver that runs the epoch loop, checkpointing, and plotting."""

import os
import time

import torch

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.config.variable_schema import parse_variable_schema
from hydragnn.distributed import (
    check_remaining,
    get_comm_size_and_rank,
    get_device,
    synchronize_local_sgd_parameters,
)
from hydragnn.loss_reporting import configure_supervised_components, format_loss_report
from hydragnn.models.utils.model import Checkpoint, EarlyStopping
from hydragnn.postprocess.visualizer import Visualizer
from hydragnn.train.loops import test, train, validate
from hydragnn.train.precision import resolve_precision
from hydragnn.train.reduction import reduce_values_ranks
from hydragnn.utils.print.print_utils import iterate_tqdm, print_distributed
from hydragnn.utils.profiling_and_tracing.profile import Profiler
from hydragnn.utils.profiling_and_tracing.time_utils import Timer


def _print_named_task_losses(verbosity, task_names, losses_by_split):
    """Print each task loss with its name and a consistent split format."""
    for task_index, task_name in enumerate(task_names):
        fields = [
            f"{split_name} Loss: {losses[task_index].item():.8f}"
            for split_name, losses in losses_by_split.items()
        ]
        print_distributed(verbosity, f"{task_name} " + ", ".join(fields))


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
    variable_schema = parse_variable_schema(config["Variables"])
    config = config["NeuralNetwork"]
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
    configured_output_names = [spec.name for spec in variable_schema.outputs]
    configure_supervised_components(model.module, variable_schema.outputs)

    device = get_device()
    if compute_grad_energy:
        num_tasks = len(model.module.task_names)
        task_dims = [1] * num_tasks
        task_weights = model.module.task_weights
        output_names = list(model.module.task_names)
    else:
        num_tasks = model.module.num_heads
        task_dims = model.module.head_dims
        task_weights = model.module.loss_weights
        output_names = configured_output_names
    task_names = output_names

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

    ## inference only
    if int(os.getenv("HYDRAGNN_EVALONLY", "0")) == 1:
        for loader, dataset_name in zip(
            [train_loader, val_loader, test_loader], ["trainset", "valset", "testset"]
        ):
            loss, taskserr, true_values, predicted_values = test(
                loader,
                model,
                verbosity,
                num_tasks=num_tasks,
                reduce_ranks=True,
                return_samples=plot_hist_solution,
                compute_grad_energy=compute_grad_energy,
                precision=precision,
            )
            print_distributed(
                verbosity,
                f"Loss for {dataset_name}: {loss:.8f}",
            )
            _print_named_task_losses(verbosity, task_names, {dataset_name: taskserr})
            print_distributed(
                verbosity,
                format_loss_report(
                    -1,
                    dataset_name.removesuffix("set"),
                    loss,
                    model.module.last_epoch_loss_report,
                ),
            )
        return

    timer = Timer("train_validate_test")
    timer.start()

    epoch_start = config["Training"].get("epoch_start", 0)
    for epoch in range(epoch_start, num_epoch):
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)
        ## timer per epoch
        t0 = time.time()
        profiler.set_current_epoch(epoch)
        for dataloader in [train_loader, val_loader, test_loader]:
            sampler = getattr(dataloader, "batch_sampler", None)
            if getattr(sampler, "set_epoch", None) is not None:
                sampler.set_epoch(epoch)
            else:
                sampler = dataloader.sampler
                if getattr(sampler, "set_epoch", None) is not None:
                    sampler.set_epoch(epoch)

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
                performer_redraw_interval=config["Training"].get(
                    "global_attn_redraw_interval", 1000
                ),
            )
            train_loss_report = dict(model.module.last_epoch_loss_report)
            tr.stop("train")
            tr.disable()
            if epoch == 0:
                tr.reset()

        # Validation, checkpoints, and the model returned after the final
        # epoch must observe one globally consistent replica. Avoid a redundant
        # collective when the last optimizer step already performed its
        # scheduled post-local-SGD parameter average.
        synchronize_local_sgd_parameters(optimizer)

        if int(os.getenv("HYDRAGNN_VALTEST", "1")) == 0:
            continue

        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

        val_loss, val_taskserr = validate(
            val_loader,
            model,
            verbosity,
            num_tasks=num_tasks,
            reduce_ranks=True,
            compute_grad_energy=compute_grad_energy,
            precision=precision,
        )
        val_loss_report = dict(model.module.last_epoch_loss_report)
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
        test_loss_report = dict(model.module.last_epoch_loss_report)
        scheduler.step(val_loss)
        if writer is not None:
            writer.add_scalar("train error", train_loss, epoch)
            writer.add_scalar("validate error", val_loss, epoch)
            writer.add_scalar("test error", test_loss, epoch)
            for ivar, task_name in enumerate(task_names):
                writer.add_scalar(
                    "train error of " + task_name, train_taskserr[ivar], epoch
                )
        print_distributed(
            verbosity,
            f"Epoch: {epoch:02d}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, "
            f"Test Loss: {test_loss:.8f}",
        )
        _print_named_task_losses(
            verbosity,
            task_names,
            {
                "Train": train_taskserr,
                "Val": val_taskserr,
                "Test": test_taskserr,
            },
        )
        for split, total, report in (
            ("train", train_loss, train_loss_report),
            ("validation", val_loss, val_loss_report),
            ("test", test_loss, test_loss_report),
        ):
            print_distributed(
                verbosity, format_loss_report(epoch, split, total, report)
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
