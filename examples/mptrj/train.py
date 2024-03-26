import os, re, json
import logging
import sys
from mpi4py import MPI
import argparse

import glob

import random
import numpy as np

import torch
from torch import tensor
from torch_geometric.data import Data

from torch_geometric.transforms import Distance, Spherical, LocalCartesian

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg
from hydragnn.preprocess.utils import RadiusGraph, RadiusGraphPBC
from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.tracer as tr

from hydragnn.utils.print_utils import iterate_tqdm, log

from jarvis.db.jsonutils import loadjson, dumpjson
from pymatgen.core.structure import Structure
from jarvis.core.atoms import pmg_to_atoms
from utils.generate_dictionary import generate_dictionary_elements

inverted_dict = generate_dictionary_elements()

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import subprocess
from hydragnn.utils import nsplit


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)


class MPTrjDataset(AbstractBaseDataset):
    def __init__(self, dirpath, var_config, data_type, dist=False, tmpfs=None):
        super().__init__()

        self.var_config = var_config
        self.data_path = os.path.join(dirpath, data_type)

        self.radius_graph = RadiusGraph(5.0, loop=False, max_num_neighbors=50)

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        d = None
        if tmpfs is None:
            d = loadjson(os.path.join(dirpath, "MPtrj_2022.9_full.json"))
        else:
            d = loadjson(os.path.join(tmpfs, "MPtrj_2022.9_full.json"))

        mpids = list(d.keys())

        dataset = []

        if not self.dist:
            mpids_loc = mpids
        else:
            mpids_loc = list(nsplit(mpids, self.world_size))[self.rank]

        for i in mpids_loc:

            tmp = d[i]

            for j, k in tmp.items():

                info = {}

                info["jid"] = j

                info["total_energy"] = k["energy_per_atom"]

                info["forces"] = k["force"]

                info["stresses"] = k["stress"]

                info["atoms"] = pmg_to_atoms(
                    Structure.from_dict(k["structure"])
                ).to_dict()

                info["magmom"] = k["magmom"]

                # Convert lists to PyTorch tensors
                lattice_mat = torch.tensor(
                    info["atoms"]["lattice_mat"], dtype=torch.float32
                )
                coords = torch.tensor(info["atoms"]["coords"], dtype=torch.float32)

                # Multiply 'lattice_mat' by the transpose of 'coords'
                result = torch.matmul(lattice_mat, coords.T)

                # Transpose the result
                positions = result.T.clone().detach()

                # Extracting data from info dictionary
                total_energy = info["total_energy"]
                forces = info["forces"]
                stresses = info["stresses"]
                magmom = info["magmom"]
                atoms_dict = info["atoms"]

                # Converting positions and atomic numbers to torch tensors
                atomic_numbers = torch.tensor(
                    [inverted_dict[element] for element in atoms_dict["elements"]],
                    dtype=torch.float32,
                ).view(-1, 1)
                energy = torch.tensor(total_energy, dtype=torch.float32).unsqueeze(0)
                forces = torch.tensor(forces, dtype=torch.float32)
                x = torch.cat([atomic_numbers, positions, forces], dim=1)

                # Creating the Data object
                data = Data(
                    supercell_size=lattice_mat,
                    energy=energy,
                    force=forces,
                    # stress=torch.tensor(stresses, dtype=torch.float32),
                    # magmom=torch.tensor(magmom, dtype=torch.float32),
                    pos=positions,
                    atomic_numbers=atomic_numbers,  # Reshaping atomic_numbers to Nx1 tensor
                    x=x,
                    y=energy,
                )

                data = self.radius_graph(data)
                data = transform_coordinates(data)

                self.dataset.append(data)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="mptrj_energy.json"
    )
    parser.add_argument(
        "--train_path",
        help="path to training data",
        type=str,
        default="s2ef_train_200K_uncompressed",
    )
    parser.add_argument(
        "--test_path",
        help="path to testing data",
        type=str,
        default="s2ef_val_id_uncompressed",
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--tmpfs",
        default=None,
        help="Transient storage space such as /mnt/bb/$USER which can be used as a temporary scratch space for caching and/or extracting data. The location must exist before use by HydraGNN.",
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
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
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

    log_name = "MPTrj" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "MPTrj" if args.modelname is None else args.modelname
    if args.preonly:
        ## local data
        total = MPTrjDataset(
            os.path.join(datadir),
            var_config,
            data_type=args.train_path,
            dist=True,
            tmpfs=args.tmpfs,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

        ## adios
        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % modelname
            )
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        ## pickle
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

    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
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

    # Print details of neural network architecture
    print_model(model)

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
