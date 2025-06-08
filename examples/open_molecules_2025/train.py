import os, re, json
import logging
import sys
from mpi4py import MPI
import argparse

###############################################################################################
import importlib.util

# Path to the outer directory that contains 'ase_db_backends'
local_path = "/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/open_molecules_2025/fairchem/core/datasets"

# Ensure this path is searched first
sys.path.insert(0, local_path)

# Now explicitly load your local ase_db_backends package
package_init = os.path.join(local_path, "ase_db_backends", "ase_db_backends", "__init__.py")
spec = importlib.util.spec_from_file_location("ase_db_backends", package_init)
mod = importlib.util.module_from_spec(spec)
sys.modules["ase_db_backends"] = mod
spec.loader.exec_module(mod)

###############################################################################################

import numpy as np

import random

import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

from torch_geometric.data import Data
from torch_geometric.transforms import Distance, Spherical, LocalCartesian
from torch_geometric.transforms import AddLaplacianEigenvectorPE

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    PBCLocalCartesian,
    pbc_as_tensor,
    gather_deg,
)
from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.print.print_utils import iterate_tqdm, log

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import subprocess
from hydragnn.utils.distributed import nsplit

## FIMME
torch.backends.cudnn.enabled = False

from fairchem.core.datasets import AseDBDataset
import glob


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=False)
transform_coordinates = LocalCartesian(norm=False, cat=False)
# transform_coordinates = Distance(norm=False, cat=False)

transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
# transform_coordinates_pbc = PBCDistance(norm=False, cat=False)

dataset_names = [
    "rattled-300-subsampled",
]

class OMol2025(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        data_type,
        graphgps_transform=None,
        energy_per_atom=True,
        dist=False,
    ):
        super().__init__()

        assert (data_type == "train") or (
            data_type == "val"
        ), "data_type must be a string either equal to 'train' or to 'val'"

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.data_path = os.path.join(dirpath, data_type)
        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )
        self.radius_graph_pbc = RadiusGraphPBC(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        config_kwargs = {}  # see tutorial on additional configuration

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        ## Parallelizing over data files for training data. For val set, we parallelize over each molecules.
        if data_type == "train":
            total_file_list = glob.glob(
                os.path.join(dirpath, data_type, "*.aselmdb"), recursive=True
            )
            rx = list(nsplit(total_file_list, self.world_size))[self.rank]
            print(self.rank, "total:", len(total_file_list), "local:", len(rx))
            dataset_list = rx
        else:
            dataset_list = dataset_names

        torch.distributed.barrier()

        for dataset_index, dataname in enumerate(dataset_list):

            # print(f"Rank {self.rank} reading {data_type}/{dataname} ... ", flush=True)

            dataset = AseDBDataset(
                config=dict(
                    src=os.path.join(dirpath, data_type, dataname), **config_kwargs
                )
            )
            # print(self.rank, "dataname:", dataname, "total:", dataset.num_samples)

            if data_type == "train":
                rx = list(range(dataset.num_samples))
            else:
                rx = list(nsplit(list(range(dataset.num_samples)), self.world_size))[
                    self.rank
                ]
                print(
                    self.rank,
                    "dataname:",
                    dataname,
                    "total:",
                    dataset.num_samples,
                    "local:",
                    len(rx),
                )

            for index in iterate_tqdm(
                rx,
                verbosity_level=2,
                desc=f"Rank{self.rank} Dataset {dataset_index}/{len(dataset_list)}",
            ):
                try:
                    pos = torch.tensor(
                        dataset.get_atoms(index).get_positions(), dtype=torch.float32
                    )
                    natoms = torch.IntTensor([pos.shape[0]])
                    atomic_numbers = torch.tensor(
                        dataset.get_atoms(index).get_atomic_numbers(),
                        dtype=torch.float32,
                    ).unsqueeze(1)
                    energy = torch.tensor(
                        dataset.get_atoms(index).get_total_energy(), dtype=torch.float32
                    ).unsqueeze(0)
                    energy_per_atom = energy.detach().clone() / natoms
                    forces = torch.tensor(
                        dataset.get_atoms(index).get_forces(), dtype=torch.float32
                    )
                    chemical_formula = dataset.get_atoms(index).get_chemical_formula()

                    cell = None
                    try:
                        cell = torch.tensor(
                            dataset.get_atoms(index).get_cell(), dtype=torch.float32
                        ).view(3, 3)
                    except:
                        print(
                            f"Atomic structure {chemical_formula} does not have cell",
                            flush=True,
                        )

                    pbc = None
                    try:
                        pbc = pbc_as_tensor(dataset.get_atoms(index).get_pbc())
                    except:
                        print(
                            f"Atomic structure {chemical_formula} does not have pbc",
                            flush=True,
                        )

                    # If either cell or pbc were not read, we set to defaults which are not none.
                    if cell is None or pbc is None:
                        cell = torch.eye(3, dtype=torch.float32)
                        pbc = torch.tensor([False, False, False], dtype=torch.bool)

                    x = torch.cat([atomic_numbers, pos, forces], dim=1)

                    # Calculate chemical composition
                    atomic_number_list = atomic_numbers.tolist()
                    assert len(atomic_number_list) == natoms
                    ## 118: number of atoms in the periodic table
                    hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
                    chemical_composition = (
                        torch.tensor(hist).unsqueeze(1).to(torch.float32)
                    )

                    data_object = Data(
                        dataset_name="omat24",
                        natoms=natoms,
                        pos=pos,
                        cell=cell,
                        pbc=pbc,
                        edge_index=None,
                        edge_attr=None,
                        atomic_numbers=atomic_numbers,
                        chemical_composition=chemical_composition,
                        smiles_string=None,
                        x=x,
                        energy=energy,
                        energy_per_atom=energy_per_atom,
                        forces=forces,
                    )

                    if self.energy_per_atom:
                        data_object.y = data_object.energy_per_atom
                    else:
                        data_object.y = data_object.energy

                    if data_object.pbc.any():
                        try:
                            data_object = self.radius_graph_pbc(data_object)
                            data_object = transform_coordinates_pbc(data_object)
                        except:
                            print(
                                f"Structure could not successfully apply one or both of the pbc radius graph and positional transform",
                                flush=True,
                            )
                            data_object = self.radius_graph(data_object)
                            data_object = transform_coordinates(data_object)
                    else:
                        data_object = self.radius_graph(data_object)
                        data_object = transform_coordinates(data_object)

                    # Default edge_shifts for when radius_graph_pbc is not activated
                    if not hasattr(data_object, "edge_shifts"):
                        data_object.edge_shifts = torch.zeros(
                            (data_object.edge_index.size(1), 3), dtype=torch.float32
                        )

                    # FIXME: PBC from bool --> int32 to be accepted by ADIOS
                    data_object.pbc = data_object.pbc.int()

                    # LPE
                    if self.graphgps_transform is not None:
                        data_object = self.graphgps_transform(data_object)

                    if self.check_forces_values(data_object.forces):
                        self.dataset.append(data_object)
                    else:
                        print(
                            f"L2-norm of force tensor is {data_object.forces.norm()} and exceeds threshold {self.forces_norm_threshold} - atomistic structure: {chemical_formula}",
                            flush=True,
                        )

                except Exception as e:
                    print(f"Rank {self.rank} reading - exception: ", e)

        print(self.rank, "Done. Wait others.", flush=True)
        torch.distributed.barrier()

        random.shuffle(self.dataset)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

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
        "--inputfile", help="input file", type=str, default="omol25_energy.json"
    )
    parser.add_argument(
        "--energy_per_atom",
        help="option to normalize energy by number of atoms",
        type=bool,
        default=False,
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--num_epoch", type=int, help="num_epoch", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--compute_grad_energy", type=bool, help="compute_grad_energy", default=False
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

    # Transformation to create positional and structural laplacian encoders
    """
    graphgps_transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )
    """

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch

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

    log_name = "OMat24" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "OMat24" if args.modelname is None else args.modelname
    if args.preonly:
        ## local data
        trainset = OMol2025(
            os.path.join(datadir),
            config,
            data_type="train",
            # graphgps_transform=graphgps_transform,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset1, valset2 = split_dataset(
            dataset=trainset,
            perc_train=0.9,
            stratify_splitting=False,
        )
        valset = [*valset1, *valset2]
        testset = OMol2025(
            os.path.join(datadir),
            config,
            data_type="val",
            # graphgps_transform=graphgps_transform,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## Need as a list
        testset = testset[:]
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        print("Before COMM.Barrier()", flush=True)
        comm.Barrier()
        print("After COMM.Barrier()", flush=True)

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

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # Print details of neural network architecture
    print_model(model)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.model.load_existing_model_config(
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
        compute_grad_energy=args.compute_grad_energy,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
