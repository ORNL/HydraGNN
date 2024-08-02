#!/usr/bin/env python3

""" Run HydraGNN's main train/test/validate
    loop on the given dataset / model combination.
"""
import os, json
import pickle, csv

import logging
import sys
from tqdm import tqdm

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
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
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


def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--csv", help="CSV dataset", action="store_const", dest="format", const="csv"
    )
    parser.set_defaults(format="adios")
    """
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        "--shmem",
        help="shmem dataset",
        action="store_const",
        dest="dataset",
        const="shmem",
    )
    group1.add_argument(
        "--ddstore",
        help="ddstore dataset",
        action="store_const",
        dest="dataset",
        const="ddstore",
    )
    group1.add_argument(
        "--simple",
        help="no special dataset",
        action="store_const",
        dest="dataset",
        const="simple",
    )
    parser.set_defaults(dataset="simple")
    """
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    args = parser.parse_args()


    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        shmem = ddstore = False
        if args.dataset == "shmem":
            shmem = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "torch"
            os.environ["HYDRAGNN_USE_ddstore"] = "0"
        elif args.dataset == "ddstore":
            ddstore = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
            os.environ["HYDRAGNN_USE_ddstore"] = "1"

        opt = {"preload": False, "shmem": shmem, "ddstore": ddstore}
        fname = os.path.join(os.path.dirname(__file__), "dataset", "clintox.bp")
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm)
        testset = AdiosDataset(fname, "testset", comm)
        comm.Barrier()
    elif args.format == "pickle":
        basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
        trainset = SimplePickleDataset(basedir, "trainset")
        valset = SimplePickleDataset(basedir, "valset")
        testset = SimplePickleDataset(basedir, "testset")
        pna_deg = trainset.pna_deg
        if args.dataset == "ddstore":
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.dataset == "ddstore":
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    comm.Barrier()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)
    comm.Barrier()

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

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