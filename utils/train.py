#!/usr/bin/env python3

""" Run HydraGNN's main train/test/validate
    loop on the given dataset / model combination.
"""
import os, json
import pickle, csv
from pathlib import Path

import logging
import sys
from tqdm import tqdm

info = logging.info


import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

from itertools import chain
import argparse
import time

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer

# from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.distributed import (
    setup_ddp,
    get_distributed_model,
    print_peak_memory,
)

from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

# from debug_dict import DebugDict


def run(argv):
    assert len(argv) == 3, f"Usage: {argv[0]} <config.json> <dataset.bp>"

    cfgfile = argv[1]
    dataset = argv[2]
    log_name = "experiment"
    (Path("logs") / log_name).mkdir(exist_ok=True, parents=True)
    verbosity = 1

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    config = json.loads(Path(cfgfile).read_text())
    # print(config)
    # config = DebugDict(config)
    # world_size, world_rank = setup_ddp()
    comm_size, rank = setup_ddp()
    # rank = comm.Get_rank()
    # comm_size = comm.Get_size()

    use_torch_backend = False  # Fix to MPI backend
    if True:  # fix to adios format
        shmem = ddstore = False
        if use_torch_backend:
            shmem = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "torch"
            os.environ["HYDRAGNN_USE_ddstore"] = "0"
        else:
            ddstore = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
            os.environ["HYDRAGNN_USE_ddstore"] = "1"

        opt = {"preload": False, "shmem": shmem, "ddstore": ddstore}
        comm = MPI.COMM_WORLD
        trainset = AdiosDataset(dataset, "trainset", comm, **opt)
        valset = AdiosDataset(dataset, "valset", comm)
        testset = AdiosDataset(dataset, "testset", comm)
        # comm.Barrier()

    print("Loaded dataset.")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    # first hurdle - we need to get metadata (what features are present) from adios datasets.
    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["Optimizer"]["batch_size"],
    )
    print("Created Dataloaders")
    # comm.Barrier()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    # comm.Barrier()
    print("Updated Config")

    if rank == 0:
        hydragnn.utils.save_config(config, log_name)
    comm.Barrier()

    timer.stop()

    # most important step - create the NN model based on the config file's
    # NeuralNetwork parameters.
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    # tell pytorch to parallelize training over torch.distributed
    model = get_distributed_model(model, verbosity)

    learning_rate = config["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # ? what is this doing to the model?
    hydragnn.utils.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    if args.mae:
        import matplotlib.pyplot as plt

        ##################################################################################################################
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for isub, (loader, setname) in enumerate(
            zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )
            ihead = 0
            head_true = np.asarray(true_values[ihead].cpu()).squeeze()
            head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = graph_feature_names[ifeat]

            ax = axs[isub]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            print(varname, ": ev, mae=", error_mae, ", rmse= ", error_rmse)

            ax.scatter(
                head_true,
                head_pred,
                s=7,
                linewidth=0.5,
                edgecolor="b",
                facecolor="none",
            )
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2f}".format(error_mae),
            )
        if rank == 0:
            fig.savefig(os.path.join("logs", log_name, varname + "_all.png"))
        plt.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()


if __name__ == "__main__":
    import sys

    run(sys.argv)
