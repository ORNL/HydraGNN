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
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json

import logging
import sys
from mpi4py import MPI
import argparse

import hydragnn
from hydragnn.utils.print.print_utils import log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import hydragnn.utils.profiling_and_tracing.tracer as tr
from ising_preprocess import prepare_ising_dataset


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios saving and no train",
    )
    parser.add_argument(
        "--natom",
        type=int,
        default=3,
        help="number_atoms_per_dimension",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=10,
        help="configurational_histogram_cutoff",
    )
    parser.add_argument("--seed", type=int, help="seed", default=43)
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--log", help="log name")
    parser.add_argument("--everyone", action="store_true", help="gptimer")
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
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, "ising_model.json")
    with open(input_filename, "r") as f:
        config = json.load(f)

    log_name = get_log_name_config(config)
    if args.log is not None:
        log_name = args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    number_atoms_per_dimension = args.natom
    configurational_histogram_cutoff = args.cutoff

    modelname = "ising_model_%d_%d" % (
        number_atoms_per_dimension,
        configurational_histogram_cutoff,
    )

    model_path = os.path.join(os.path.dirname(__file__), "dataset")
    artifact_exists = (
        os.path.isdir(os.path.join(model_path, f"{modelname}.pickle"))
        if args.format == "pickle"
        else os.path.exists(os.path.join(model_path, f"{modelname}.bp"))
    )
    if args.preonly or not artifact_exists:
        preprocessing_error = None
        if rank == 0:
            try:
                total = prepare_ising_dataset(
                    number_atoms_per_dimension,
                    configurational_histogram_cutoff,
                    config,
                    seed=args.seed,
                    sampling=args.sampling,
                )
                trainset, valset, testset = split_dataset(
                    total,
                    config["NeuralNetwork"]["Training"]["perc_train"],
                    config["Dataset"].get("compositional_stratified_splitting", False),
                )
                deg = gather_deg(trainset)
                if args.format == "pickle":
                    basedir = os.path.join(model_path, f"{modelname}.pickle")
                    attrs = {"pna_deg": deg}
                    for label, split in zip(
                        ("trainset", "valset", "testset"),
                        (trainset, valset, testset),
                    ):
                        SimplePickleWriter(
                            split,
                            basedir,
                            label,
                            use_subdir=True,
                            comm=MPI.COMM_SELF,
                            attrs=attrs if label == "trainset" else {},
                        )
                elif args.format == "adios":
                    fname = os.path.join(model_path, f"{modelname}.bp")
                    writer = AdiosWriter(fname, MPI.COMM_SELF)
                    writer.add("trainset", trainset)
                    writer.add("valset", valset)
                    writer.add("testset", testset)
                    writer.add_global("pna_deg", deg)
                    writer.save()
                else:
                    raise ValueError(f"Unknown data format: {args.format}")
            except Exception as error:
                preprocessing_error = f"{type(error).__name__}: {error}"
        preprocessing_error = comm.bcast(preprocessing_error, root=0)
        if preprocessing_error is not None:
            raise RuntimeError(f"Ising preprocessing failed: {preprocessing_error}")
        comm.Barrier()
    if args.preonly:
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        info("Adios load")
        opt = {
            "preload": False,
            "shmem": False,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir, "trainset")
        valset = SimplePickleDataset(basedir, "valset")
        testset = SimplePickleDataset(basedir, "testset")
        minmax_node_feature = trainset.minmax_node_feature
        minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg

        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.minmax_node_feature = minmax_node_feature
            trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity, config=config
    )

    writer = hydragnn.utils.model.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config,
        log_name,
        verbosity,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
