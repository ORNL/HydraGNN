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
"""Shared training pipeline behind ``hydragnn train`` and the example scripts.

:func:`run_training` and :func:`write_split_datasets` handle homogeneous
(``Data``) and heterogeneous (``HeteroData``) samples.
"""

import argparse
import json
import logging
import os
import sys

FORMATS = ("adios", "pickle", "hdf5")
SPLITS = ("trainset", "valset", "testset")


def build_parser(
    description=None,
    default_inputfile=None,
    default_format="adios",
    preprocessing=True,
    formats=FORMATS,
):
    """Return a parser with the flags shared by every training script."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputfile",
        type=str,
        default=default_inputfile,
        required=default_inputfile is None,
        help="JSON configuration file",
    )
    if preprocessing:
        parser.add_argument(
            "--preonly", action="store_true", help="preprocess only (no training)"
        )
        parser.add_argument(
            "--sampling", type=float, default=None, help="sampling ratio"
        )
        parser.add_argument(
            "--energy_per_atom",
            action="store_true",
            help="normalize energy by number of atoms",
        )
        parser.add_argument("--modelname", default=None, help="dataset name")
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, default=None, help="ddstore width")
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", default=None, help="log name")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=None, help="number of epochs")
    parser.add_argument(
        "--everyone", action="store_true", help="write GPTL timers on every rank"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp64", "bf16"],
        default=None,
        help="override Training.precision",
    )
    group = parser.add_mutually_exclusive_group()
    for fmt in formats:
        group.add_argument(
            "--" + fmt,
            action="store_const",
            dest="format",
            const=fmt,
            help="%s dataset" % fmt,
        )
    parser.set_defaults(format=default_format)
    return parser


def load_config(path, args=None):
    """Load a JSON config and apply ``--batch_size``/``--num_epoch`` overrides."""
    with open(path, "r") as f:
        config = json.load(f)
    training = config["NeuralNetwork"]["Training"]
    if args is not None:
        if args.batch_size is not None:
            training["batch_size"] = args.batch_size
        if args.num_epoch is not None:
            training["num_epoch"] = args.num_epoch
    return config


def setup_run(args, default_log, use_deepspeed=False):
    """Initialize distributed state and logging; return ``(rank, log_name, writer)``."""
    import hydragnn
    from hydragnn.utils.print.print_utils import log

    _, rank = hydragnn.distributed.setup_ddp(use_deepspeed=use_deepspeed)
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % rank,
        datefmt="%H:%M:%S",
    )
    log_name = default_log if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.models.utils.get_summary_writer(log_name)
    log("Command: {0}\n".format(" ".join(sys.argv)), rank=0)
    return rank, log_name, writer


def dataset_path(script_dir, name, fmt):
    """Return the conventional ``dataset/<name>.bp|.pickle|.h5`` location."""
    suffix = {"adios": "bp", "pickle": "pickle", "hdf5": "h5"}[fmt]
    return os.path.join(script_dir, "dataset", "%s.%s" % (name, suffix))


def write_split_datasets(path, fmt, trainset, valset, testset, comm=None):
    """Write rank-local splits to ``path`` and store the PNA degree histogram."""
    from mpi4py import MPI
    from hydragnn.preprocess.graph_samples_checks_and_updates import (
        gather_deg,
        gather_hetero_deg,
    )

    comm = MPI.COMM_WORLD if comm is None else comm
    # Empty shards must follow the same branch as the others to avoid collective hangs.
    hetero = comm.allreduce(
        len(trainset) > 0 and hasattr(trainset[0], "node_types"), op=MPI.LOR
    )
    if hetero and fmt == "adios":
        raise NotImplementedError(
            "ADIOS cannot store HeteroData samples; use the pickle format."
        )
    if hetero:
        deg = gather_hetero_deg(trainset, 0, comm=comm)
    else:
        deg = gather_deg(trainset)
    splits = dict(zip(SPLITS, (trainset, valset, testset)))
    if fmt == "adios":
        from hydragnn.data.adiosdataset import AdiosWriter

        writer = AdiosWriter(path, comm)
        for label, split in splits.items():
            writer.add(label, split)
        writer.add_global("pna_deg", deg)
        writer.save()
    elif fmt == "pickle":
        from hydragnn.data.pickledataset import SimplePickleWriter

        for label, split in splits.items():
            SimplePickleWriter(
                list(split),
                path,
                label,
                use_subdir=True,
                comm=comm,
                attrs={"pna_deg": deg} if label == "trainset" else dict(),
            )
    elif fmt == "hdf5":
        from hydragnn.data.hdf5dataset import HDF5Writer

        # HDF5 has no global attributes; update_config recomputes pna_deg.
        writer = HDF5Writer(path, comm)
        for label, split in splits.items():
            writer.add(label, list(split))
        writer.save()
    else:
        raise NotImplementedError("Unsupported format: %s" % fmt)


def load_split_datasets(
    path,
    fmt,
    comm=None,
    shmem=False,
    ddstore=False,
    ddstore_width=None,
    preload_train=False,
):
    """Open the train/val/test splits written by :func:`write_split_datasets`."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD if comm is None else comm
    if fmt == "adios":
        from hydragnn.data.adiosdataset import AdiosDataset

        if shmem and ddstore:
            raise ValueError("Cannot use both ddstore and shmem")
        opt = dict(preload=False, shmem=shmem, ddstore=ddstore)
        opt["ddstore_width"] = ddstore_width
        splits = [AdiosDataset(path, label, comm, **opt) for label in SPLITS]
    elif fmt == "pickle":
        from hydragnn.data.pickledataset import SimplePickleDataset

        splits = [
            SimplePickleDataset(
                basedir=path,
                label=label,
                preload=preload_train and label == "trainset",
            )
            for label in SPLITS
        ]
        if ddstore:
            from hydragnn.data.distdataset import DistDataset

            pna_deg = splits[0].pna_deg
            splits = [
                DistDataset(split, label, comm, ddstore_width=ddstore_width)
                for split, label in zip(splits, SPLITS)
            ]
            splits[0].pna_deg = pna_deg
    elif fmt == "hdf5":
        from hydragnn.data.hdf5dataset import HDF5Dataset

        splits = [HDF5Dataset(path, label) for label in SPLITS]
    else:
        raise NotImplementedError("Unsupported format: %s" % fmt)

    if ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
    return tuple(splits)


def run_training(
    config,
    trainset,
    valset,
    testset,
    log_name,
    writer,
    precision=None,
    everyone=False,
    comm=None,
    variables=True,
    loader_transform=None,
    deferred_loss_provider=None,
    model_wrapper=None,
    find_unused_parameters=False,
    before_train=None,
    after_train=None,
    use_deepspeed=False,
):
    """Build loaders, model and optimizer, then train, validate, test and save.

    ``HeteroData`` samples get their metadata and ``node_input_dims`` passed to
    the model, and ``HeteroPNA`` gets a degree histogram over all edge types.

    Hooks for scripts that need more than the default pipeline:

    * ``variables=False`` skips compiling ``Variables`` on each sample, for
      datasets whose samples are already in model format (implied for hetero).
    * ``loader_transform(loader)`` wraps each loader after creation.
    * ``deferred_loss_provider`` names a ``Training.loss`` provider that needs
      the built model and is attached by ``model_wrapper`` instead.
    * ``model_wrapper(model, config)`` returns the model to train.
    * ``before_train(model, optimizer, loaders, config)`` runs after loading any
      checkpoint; a truthy return skips training and saving.
    * ``after_train(model)`` runs before the model is saved.
    """
    import torch
    from mpi4py import MPI

    import hydragnn
    import hydragnn.utils.profiling_and_tracing.tracer as tr
    from hydragnn.preprocess.graph_samples_checks_and_updates import (
        gather_hetero_deg,
    )
    from hydragnn.utils.optimizer import select_optimizer
    from hydragnn.utils.print.print_utils import log
    from hydragnn.utils.profiling_and_tracing.time_utils import Timer

    comm = MPI.COMM_WORLD if comm is None else comm
    rank = comm.Get_rank()
    verbosity = config["Verbosity"]["level"]
    training = config["NeuralNetwork"]["Training"]

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()
    log(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset)),
        rank=0,
    )
    sample = trainset[0]
    hetero = hasattr(sample, "node_types")
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        training["batch_size"],
        batching=training.get("Batching"),
        # The Variables schema compiler only handles homogeneous samples.
        variables=config["Variables"] if variables and not hetero else None,
    )
    if loader_transform is not None:
        train_loader, val_loader, test_loader = (
            loader_transform(loader)
            for loader in (train_loader, val_loader, test_loader)
        )
    if precision is not None:
        training["precision"] = precision
    config = hydragnn.config.update_config(
        config, train_loader, val_loader, test_loader
    )
    arch = config["NeuralNetwork"]["Architecture"]
    # update_config only computes homogeneous PNA degrees.
    if arch["mpnn_type"] == "HeteroPNA" and not arch.get("pna_deg"):
        if hasattr(trainset, "pna_deg"):
            arch["pna_deg"] = list(trainset.pna_deg)
        else:
            arch["pna_deg"] = gather_hetero_deg(trainset, verbosity)
        arch["max_neighbours"] = len(arch["pna_deg"]) - 1
    precision = config["NeuralNetwork"]["Training"]["precision"]
    comm.Barrier()
    hydragnn.config.save_config(config, log_name)
    timer.stop()

    model_config = config["NeuralNetwork"]
    if deferred_loss_provider is not None:
        model_config = hydragnn.domain_losses.defer_domain_loss(
            model_config, deferred_loss_provider
        )
    model = hydragnn.models.create_model_config(
        config=model_config,
        verbosity=verbosity,
        metadata=sample.metadata() if hetero else None,
        node_input_dims=arch.get("node_input_dims") if hetero else None,
    )
    if model_wrapper is not None:
        model = model_wrapper(model, config)
    optimizer = select_optimizer(
        model, config["NeuralNetwork"]["Training"]["Optimizer"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    model, optimizer = hydragnn.distributed.distributed_model_wrapper(
        model,
        optimizer,
        verbosity,
        find_unused_parameters=find_unused_parameters,
        use_deepspeed=use_deepspeed,
        config=config,
    )
    hydragnn.models.utils.print_model(model)
    hydragnn.models.utils.load_existing_model_config(
        model,
        config["NeuralNetwork"]["Training"],
        optimizer=optimizer,
        use_deepspeed=use_deepspeed,
    )
    loaders = (train_loader, val_loader, test_loader)
    if before_train is not None and before_train(model, optimizer, loaders, config):
        if writer is not None:
            writer.close()
        return model, config

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
        create_plots=False,
        compute_grad_energy=hydragnn.domain_losses.uses_interatomic_potential(config),
        precision=precision,
        use_deepspeed=use_deepspeed,
    )

    if after_train is not None:
        after_train(model)
    hydragnn.models.utils.save_model(
        model, optimizer, log_name, use_deepspeed=use_deepspeed
    )
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        if everyone or rank == 0:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    return model, config


def train_from_args(args, config, path, log_name, writer, preload_train=False):
    """Load the preprocessed splits at ``path`` and run :func:`run_training`."""
    splits = load_split_datasets(
        path,
        args.format,
        shmem=args.shmem,
        ddstore=args.ddstore,
        ddstore_width=args.ddstore_width,
        preload_train=preload_train,
    )
    return run_training(
        config,
        *splits,
        log_name,
        writer,
        precision=args.precision,
        everyone=args.everyone,
    )


def main(argv=None):
    """Entry point for the ``hydragnn`` console script."""
    parser = argparse.ArgumentParser(prog="hydragnn")
    commands = parser.add_subparsers(dest="command", required=True)
    train = build_parser(preprocessing=False)
    commands.add_parser(
        "train",
        parents=[train],
        add_help=False,
        help="train a GNN on a preprocessed ADIOS, pickle, or HDF5 dataset",
        description="Train a GNN on a dataset written by an example's --preonly step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).add_argument(
        "--dataset",
        required=True,
        help="path to a dataset written with --preonly (.bp directory, .pickle, or .h5)",
    )
    args = parser.parse_args(argv)

    import torch.distributed as dist

    config = load_config(args.inputfile, args)
    default_log = os.path.splitext(os.path.basename(args.inputfile))[0]
    _, log_name, writer = setup_run(args, default_log)
    train_from_args(args, config, args.dataset, log_name, writer)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
