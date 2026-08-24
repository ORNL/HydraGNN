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
import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.serializeddataset import SerializedDataset
from eam_preprocess import prepare_eam_dataset, split_and_write_eam_dataset

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset, AdiosWriter
except ImportError:
    pass

import torch
import torch.distributed as dist


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="NiNb_EAM_energy.json"
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        default=None,
        help="directory containing raw CFG records; defaults to Dataset.path.total",
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
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.print.setup_log(get_log_name_config(config))
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

    os.environ["SERIALIZED_DATA_PATH"] = dirpwd + "/dataset"
    datasetname = config["Dataset"]["name"]
    fname_adios = dirpwd + "/dataset/%s.bp" % (datasetname)
    if not args.loadexistingsplit:
        preprocessing_error = None
        if rank == 0:
            try:
                raw_data_path = args.raw_data or config["Dataset"]["path"].get("total")
                if raw_data_path is None:
                    raise ValueError(
                        "EAM preprocessing requires --raw-data or Dataset.path.total"
                    )
                if not os.path.isabs(raw_data_path):
                    raw_data_path = os.path.join(dirpwd, raw_data_path)
                prepared = prepare_eam_dataset(raw_data_path, config)
                if args.format == "pickle":
                    basedir = os.path.join(dirpwd, "dataset", "serialized_dataset")
                    split_and_write_eam_dataset(prepared, config, basedir)
                elif args.format == "adios":
                    splits = hydragnn.preprocess.split_dataset(
                        prepared,
                        config["NeuralNetwork"]["Training"]["perc_train"],
                        config["Dataset"].get(
                            "compositional_stratified_splitting", False
                        ),
                    )
                    writer = AdiosWriter(fname_adios, MPI.COMM_SELF)
                    for label, split in zip(("trainset", "valset", "testset"), splits):
                        writer.add(label, split)
                    writer.save()
                else:
                    raise ValueError(f"Unknown data format: {args.format}")
            except Exception as error:
                preprocessing_error = f"{type(error).__name__}: {error}"
        preprocessing_error = comm.bcast(preprocessing_error, root=0)
        if preprocessing_error is not None:
            raise RuntimeError(f"EAM preprocessing failed: {preprocessing_error}")
    comm.Barrier()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        info("Adios load")
        opt = {
            "preload": True,
            "shmem": False,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % datasetname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "serialized_dataset"
        )
        trainset = SerializedDataset(basedir, datasetname, "trainset")
        valset = SerializedDataset(basedir, datasetname, "valset")
        testset = SerializedDataset(basedir, datasetname, "testset")
    else:
        raise ValueError("Unknown data format: %d" % args.format)
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

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
    config["Variables"].pop("minmax_node_feature", None)
    config["Variables"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

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
        create_plots=True,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.time_utils.print_timers(verbosity)
    if writer is not None:
        writer.close()

    sys.exit(0)
