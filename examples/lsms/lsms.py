import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.print_utils import log
from hydragnn.utils.lsmsdataset import LSMSDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.utils import gather_deg
import hydragnn.utils.tracer as tr

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--inputfile", help="input file", type=str, default="lsms_multi_tasking.json")

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

    graph_feature_names = ["mixing_enthalpy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "charge_transfer", "magnetic_moment"]
    node_feature_dims = [1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/FePt_enthalpy")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    modelname = "fept_lsms"
    log_name = modelname if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    datasetname = config["Dataset"]["name"]
    for dataset_type, raw_data_path in config["Dataset"]["path"].items():
        config["Dataset"]["path"][dataset_type] = os.path.join(dirpwd, raw_data_path)

    if args.preonly:
        ## Only rank=0 is enough for pre-processing
        total = LSMSDataset(config, dist=True)

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=config["Dataset"]["compositional_stratified_splitting"],
        )
        print(len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

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
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=var_config)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=var_config)
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)

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

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)