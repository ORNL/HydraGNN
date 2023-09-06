import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json

import logging
import sys
from mpi4py import MPI
import argparse

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.model import print_model
from hydragnn.utils.lsmsdataset import LSMSDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch
import torch.distributed as dist

import warnings

## For create_configurations
import shutil
from sympy.utilities.iterables import multiset_permutations
import scipy.special
import math

from create_configurations import E_dimensionless

import hydragnn.utils.tracer as tr
from hydragnn.utils import nsplit


def write_to_file(total_energy, atomic_features, count_config, dir, prefix):

    numpy_string_total_value = np.array2string(total_energy)

    filetxt = numpy_string_total_value

    for index in range(0, atomic_features.shape[0]):
        numpy_row = atomic_features[index, :]
        numpy_string_row = np.array2string(
            numpy_row, separator="\t", suppress_small=True
        )
        filetxt += "\n" + numpy_string_row.lstrip("[").rstrip("]")

        filename = os.path.join(dir, prefix + str(count_config) + ".txt")
        with open(filename, "w") as f:
            f.write(filetxt)


def create_dataset_mpi(
    L,
    histogram_cutoff,
    dir,
    spin_function=lambda x: x,
    scale_spin=False,
    comm=None,
    seed=None,
):
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if seed is not None:
        np.random.seed(seed)

    count_config = 0
    rx = list(nsplit(range(0, L ** 3), comm_size))[rank]
    info("rx", rx.start, rx.stop)
    for num_downs in range(rx.start, rx.stop):
        subdir = os.path.join(dir, str(num_downs))
        os.makedirs(subdir, exist_ok=True)

    for num_downs in iterate_tqdm(
        range(rx.start, rx.stop), verbosity_level=2, desc="Creating dataset"
    ):
        prefix = "output_%d_" % num_downs
        subdir = os.path.join(dir, str(num_downs))

        primal_configuration = np.ones((L ** 3,))
        for down in range(0, num_downs):
            primal_configuration[down] = -1.0

        # If the current composition has a total number of possible configurations above
        # the hard cutoff threshold, a random configurational subset is picked
        if scipy.special.binom(L ** 3, num_downs) > histogram_cutoff:
            for num_config in range(0, histogram_cutoff):
                config = np.random.permutation(primal_configuration)
                config = np.reshape(config, (L, L, L))
                total_energy, atomic_features = E_dimensionless(
                    config, L, spin_function, scale_spin
                )

                write_to_file(
                    total_energy, atomic_features, count_config, subdir, prefix
                )

                count_config = count_config + 1

        # If the current composition has a total number of possible configurations smaller
        # than the hard cutoff, then all possible permutations are generated
        else:
            for config in multiset_permutations(primal_configuration):
                config = np.array(config)
                config = np.reshape(config, (L, L, L))
                total_energy, atomic_features = E_dimensionless(
                    config, L, spin_function, scale_spin
                )

                write_to_file(
                    total_energy, atomic_features, count_config, subdir, prefix
                )

                count_config = count_config + 1


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
        default=1000,
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
    hydragnn.utils.setup_log(log_name)
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

    if args.preonly:
        """
        Parallel ising data generation step:
        1. Generate ising data (*.txt) in parallel (create_dataset_mpi)
        2. Read raw dataset in parallel (*.txt) (RawDataset)
        3. Split into a train, valid, and test set (split_dataset)
        4. Save as Adios file in parallel
        """
        sys.setrecursionlimit(1000000)
        dir = os.path.join(os.path.dirname(__file__), "./dataset/%s" % modelname)
        if rank == 0:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)
        comm.Barrier()

        info("Generating ... ")
        info("number_atoms_per_dimension", number_atoms_per_dimension)
        info("configurational_histogram_cutoff", configurational_histogram_cutoff)

        # Use sine function as non-linear extension of Ising model
        # Use randomized scaling of the spin magnitudes
        spin_func = lambda x: math.sin(math.pi * x / 2)
        create_dataset_mpi(
            number_atoms_per_dimension,
            configurational_histogram_cutoff,
            dir,
            spin_function=spin_func,
            scale_spin=True,
            comm=comm,
            seed=args.seed,
        )
        comm.Barrier()

        config["Dataset"]["path"]["total"] = dir
        total = LSMSDataset(config, dist=True, sampling=args.sampling)

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=config["Dataset"]["compositional_stratified_splitting"],
        )
        print(len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        attrs = dict()
        attrs["minmax_node_feature"] = total.minmax_node_feature
        attrs["minmax_graph_feature"] = total.minmax_graph_feature
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            use_subdir=True,
            attrs=attrs,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            use_subdir=True,
        )

        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

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

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    ## Set minmax read from bp file
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature
    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    del config["NeuralNetwork"]["Variables_of_interest"]["minmax_node_feature"]
    del config["NeuralNetwork"]["Variables_of_interest"]["minmax_graph_feature"]
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

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = hydragnn.utils.get_summary_writer(log_name)

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
        config["NeuralNetwork"],
        log_name,
        verbosity,
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
