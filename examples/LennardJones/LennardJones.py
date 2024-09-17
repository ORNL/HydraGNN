import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random

import logging
import sys
import argparse

import hydragnn
from hydragnn.utils.print_utils import iterate_tqdm, log
from hydragnn.utils.time_utils import Timer

from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.abstractrawdataset import AbstractBaseDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance, Spherical, LocalCartesian
import torch
import torch.distributed as dist

from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr

# Using LJ dataset creation
from configurational_data import deterministic_graph_data
from LJpotential import LJpotential
from AtomicStructure import AtomicStructureHandler


def create_dataset(path, config):
    # Angstrom unit
    primitive_bravais_lattice_constant_x = 3.8
    primitive_bravais_lattice_constant_y = 3.8
    primitive_bravais_lattice_constant_z = 3.8
    radius_cutoff = config["NeuralNetwork"]["Architecture"]["radius"]
    number_configurations = 1000
    atom_types = [1]
    formula = LJpotential(1.0, 3.4)
    atomic_structure_handler = AtomicStructureHandler(
        atom_types,
        [
            primitive_bravais_lattice_constant_x,
            primitive_bravais_lattice_constant_y,
            primitive_bravais_lattice_constant_z,
        ],
        radius_cutoff,
        formula,
    )
    deterministic_graph_data(
        path,
        atom_types,
        atomic_structure_handler=atomic_structure_handler,
        radius_cutoff=radius_cutoff,
        relative_maximum_atomic_displacement=1e-1,
        number_configurations=number_configurations,
    )


# FIXME: this works fine for now because we train on disordered atomic structures with potentials and forces computed with Lennard-Jones


torch.set_default_dtype(torch.float32)


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
create_graph_fromXYZ = RadiusGraph(r=5.0)  # radius cutoff in angstrom
compute_edge_lengths = Distance(norm=False, cat=True)
spherical_coordinates = Spherical(norm=False, cat=False)
cartesian_coordinates = LocalCartesian(norm=False, cat=False)


class LJDataset(AbstractBaseDataset):
    """LJDataset dataset class"""

    def __init__(self, dirpath, dist=False, sampling=None):
        super().__init__()

        self.dist = dist
        self.world_size = 1
        self.rank = 1
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        dirfiles = sorted(os.listdir(dirpath))

        rx = list(nsplit((dirfiles), self.world_size))[self.rank]

        for file in rx:
            filepath = os.path.join(dirpath, file)
            self.dataset.append(self.transform_input_to_data_object_base(filepath))

    def transform_input_to_data_object_base(self, filepath):

        # Using readline()
        file = open(filepath, "r")

        torch_data = torch.empty((0, 8), dtype=torch.float32)
        torch_supercell = torch.zeros((0, 3), dtype=torch.float32)

        count = 0

        while True:
            count += 1

            # Get next line from file
            line = file.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break

            if count == 1:
                total_energy = float(line)
            elif count == 2:
                energy_per_atom = float(line)
            elif 2 < count < 6:
                array_line = np.fromstring(line, dtype=float, sep="\t")
                torch_supercell = torch.cat(
                    [torch_supercell, torch.from_numpy(array_line).unsqueeze(0)], axis=0
                )
            elif count > 5:
                array_line = np.fromstring(line, dtype=float, sep="\t")
                torch_data = torch.cat(
                    [torch_data, torch.from_numpy(array_line).unsqueeze(0)], axis=0
                )
            # print("Line{}: {}".format(count, line.strip()))

        file.close()

        num_nodes = torch_data.shape[0]

        energy_pre_translation_factor = 0.0
        energy_pre_scaling_factor = 1.0 / num_nodes
        energy_per_atom_pretransformed = (
            energy_per_atom - energy_pre_translation_factor
        ) * energy_pre_scaling_factor
        grad_energy_post_scaling_factor = (
            1.0 / energy_pre_scaling_factor * torch.ones(num_nodes, 1)
        )
        forces = torch_data[:, [5, 6, 7]]
        forces_pre_scaling_factor = 1.0
        forces_pre_scaled = forces * forces_pre_scaling_factor

        data = Data(
            supercell_size=torch_supercell.to(torch.float32),
            num_nodes=num_nodes,
            grad_energy_post_scaling_factor=grad_energy_post_scaling_factor,
            forces_pre_scaling_factor=torch.tensor(forces_pre_scaling_factor).to(
                torch.float32
            ),
            forces=forces,
            forces_pre_scaled=forces_pre_scaled,
            pos=torch_data[:, [1, 2, 3]].to(torch.float32),
            x=torch.cat([torch_data[:, [0, 4]]], axis=1).to(torch.float32),
            y=torch.tensor(total_energy).unsqueeze(0).to(torch.float32),
            energy_per_atom=torch.tensor(energy_per_atom_pretransformed)
            .unsqueeze(0)
            .to(torch.float32),
            energy=torch.tensor(total_energy).unsqueeze(0).to(torch.float32),
        )
        data = create_graph_fromXYZ(data)
        data = compute_edge_lengths(data)
        data.edge_attr = data.edge_attr.to(torch.float32)
        # data = spherical_coordinates(data)
        data = cartesian_coordinates(data)

        return data

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
    parser.add_argument("--inputfile", help="input file", type=str, default="LJ.json")
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
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
    parser.set_defaults(format="pickle")  # Changed this for my PC
    args = parser.parse_args()

    graph_feature_names = ["total_energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "potential", "forces"]
    node_feature_dims = [1, 1, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/data")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Variables_of_interest"][
        "graph_feature_names"
    ] = graph_feature_names
    config["NeuralNetwork"]["Variables_of_interest"][
        "graph_feature_dims"
    ] = graph_feature_dims
    config["NeuralNetwork"]["Variables_of_interest"][
        "node_feature_names"
    ] = node_feature_names
    config["NeuralNetwork"]["Variables_of_interest"][
        "node_feature_dims"
    ] = node_feature_dims

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

    log_name = "LJ" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "LJ"
    # Check for dataset for each format
    lookdir = os.path.join(dirpwd, "dataset")
    if args.format == "pickle":
        dataset_exists = os.path.exists(os.path.join(lookdir, "LJ.pickle"))
    if args.format == "adios":
        dataset_exists = os.path.exists(os.path.join(lookdir, "%s.bp" % modelname))

    # Create dataset if preonly specified or dataset does not exist
    if not dataset_exists:

        ## local data
        create_dataset(os.path.join(lookdir, 'data'), config)
        total = LJDataset(
            os.path.join(datadir),
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg.tolist()

        setnames = ["trainset", "valset", "testset"]

        if args.format == "pickle":

            ## pickle
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

        if args.format == "adios":
            ## adios
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
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        var_config = config["NeuralNetwork"]["Variables_of_interest"]
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", preload=True, var_config=var_config
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
        compute_grad_energy=True,
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
