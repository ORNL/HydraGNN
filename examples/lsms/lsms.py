import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.lsmsdataset import LSMSDataset
from hydragnn.utils.datasets.serializeddataset import (
    SerializedWriter,
    SerializedDataset,
)
from hydragnn.preprocess.load_data import split_dataset

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


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
    parser.add_argument("--inputfile", help="input file", type=str, default="lsms.json")
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

    datasetname = config["Dataset"]["name"]
    for dataset_type, raw_data_path in config["Dataset"]["path"].items():
        config["Dataset"]["path"][dataset_type] = os.path.join(dirpwd, raw_data_path)

    if not args.loadexistingsplit and rank == 0:
        ## Only rank=0 is enough for pre-processing
        total = LSMSDataset(config)

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=config["Dataset"]["compositional_stratified_splitting"],
        )
        print(len(total), len(trainset), len(valset), len(testset))

        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % datasetname
            )
            adwriter = AdiosWriter(fname, MPI.COMM_SELF)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.save()
        elif args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "serialized_dataset"
            )
            SerializedWriter(
                trainset,
                basedir,
                datasetname,
                "trainset",
                minmax_node_feature=total.minmax_node_feature,
                minmax_graph_feature=total.minmax_graph_feature,
            )
            SerializedWriter(
                valset,
                basedir,
                datasetname,
                "valset",
            )
            SerializedWriter(
                testset,
                basedir,
                datasetname,
                "testset",
            )
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
    ## Set minmax
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

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
        model, optimizer, verbosity
    )

    # Print details of neural network architecture
    print_model(model)

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
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    dist.destroy_process_group()
    sys.exit(0)
