import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random

import h5py

import logging
import sys
import argparse

import numpy as np

import torch
import torch.distributed as dist

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
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
)
from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.print.print_utils import iterate_tqdm, log

from hydragnn.utils.descriptors_and_embeddings import xyz2mol

from rdkit import Chem

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.utils.distributed import nsplit

torch.set_default_dtype(torch.float32)

from utils.create_graph_data import Dataloader


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
# transform_coordinates = Spherical(norm=False, cat=False)
transform_coordinates = LocalCartesian(norm=False, cat=False)
# transform_coordinates = Distance(norm=False, cat=False)


class Transition1xDataset(AbstractBaseDataset):
    """Transition1xDataset dataset class"""

    def __init__(
        self,
        dirpath,
        var_config,
        graphgps_transform=None,
        energy_per_atom=True,
        dist=False,
    ):
        super().__init__()

        self.data_path = os.path.join(dirpath, "transition1x-release.h5")
        self.energy_per_atom = energy_per_atom

        self.world_size = 1
        self.rank = 0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(5.0, loop=False, max_num_neighbors=50)

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        # loop through all configurations in the data set
        dataloader = Dataloader(
            self.data_path, comm_rank=self.rank, comm_size=self.world_size
        )

        for i, configuration in enumerate(dataloader):

            data_object = None

            pos = None
            try:
                pos = torch.tensor(configuration["positions"]).to(torch.float32)
                assert pos.shape[0] > 0, "pos tensor does not have any atoms"
                assert (
                    pos.shape[1] == 3
                ), "pos tensor does not have 3 coordinates per atom"
            except:
                print(
                    f"Structure {configuration} does not have positional sites",
                    flush=True,
                )
                continue
            natoms = torch.IntTensor([pos.shape[0]])

            atomic_numbers = None
            try:
                atomic_numbers = (
                    torch.tensor([configuration["atomic_numbers"]]).to(torch.float32)
                ).t()
                assert (
                    pos.shape[0] == atomic_numbers.shape[0]
                ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
            except:
                print(
                    f"Structure {configuration} does not have positional atomic numbers",
                    flush=True,
                )
                continue

            forces = None
            try:
                forces = torch.tensor(configuration["wB97x_6-31G(d).forces"]).to(
                    torch.float32
                )
            except:
                print(f"Structure {configuration} does not have forces", flush=True)
                continue

            total_energy = None
            try:
                total_energy = configuration["wB97x_6-31G(d).energy"]
            except:
                print(
                    f"Structure {configuration} does not have total energy", flush=True
                )
                continue
            total_energy_tensor = (
                torch.tensor(total_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
            )
            total_energy_per_atom_tensor = total_energy_tensor.detach().clone() / natoms

            x = torch.cat([atomic_numbers, pos, forces], dim=1)

            # Calculate chemical composition
            atomic_number_list = atomic_numbers.tolist()
            assert len(atomic_number_list) == natoms
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)
            pos_list = pos.tolist()
            atomic_number_list_int = [int(item[0]) for item in atomic_number_list]
            try:
                mol = xyz2mol(
                    atomic_number_list_int,
                    pos_list,
                    charge=0,
                    allow_charged_fragments=True,
                    use_graph=False,
                    use_huckel=False,
                    embed_chiral=True,
                    use_atom_maps=False,
                )

                assert (
                    len(mol) == 1
                ), f"molecule with atomic numbers {atomic_number_list_int}  and positions {pos_list} does not produce RDKit.mol object"
                smiles_string = Chem.MolToSmiles(mol[0])
            except:
                smiles_string = None

            try:
                # check forces values
                assert self.check_forces_values(
                    forces
                ), f"transition1x dataset - formula:{configuration['formula']} - confid:{configuration['rxn']} - L2-norm of atomic forces exceeds {self.forces_norm_threshold}"
            except:
                continue

            data_object = Data(
                dataset_name="transition1x",
                natoms=natoms,
                pos=pos,
                cell=None,  # even if not needed, cell needs to be defined because ADIOS requires consistency across datasets
                pbc=None,  # even if not needed, pbc needs to be defined because ADIOS requires consistency across datasets
                edge_index=None,
                edge_attr=None,
                edge_shifts=None,  # even if not needed, edge_shift needs to be defined because ADIOS requires consistency across datasets
                atomic_numbers=atomic_numbers,
                chemical_composition=chemical_composition,
                smiles_string=smiles_string,
                x=x,
                energy=total_energy_tensor,
                energy_per_atom=total_energy_per_atom_tensor,
                forces=forces,
            )

            if self.energy_per_atom:
                data_object.y = data_object.energy_per_atom
            else:
                data_object.y = data_object.energy

            data_object = self.radius_graph(data_object)

            # Build edge attributes
            data_object = transform_coordinates(data_object)

            # LPE
            if self.graphgps_transform is not None:
                data_object = self.graphgps_transform(data_object)

            self.dataset.append(data_object)

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
        "--inputfile", help="input file", type=str, default="transition1x_energy.json"
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
    parser.add_argument("--everyone", action="store_true", help="gptimer")
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
    node_feature_names = [
        "atomic_number",
        "coordinates",
        "forces",
        "hCHG",
        "hVDIP",
        "hRAT",
    ]
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
    graphgps_transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

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

    log_name = "transition1x" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "transition1x"
    if args.preonly:

        ## local data
        total = Transition1xDataset(
            os.path.join(datadir),
            var_config,
            graphgps_transform=graphgps_transform,
            energy_per_atom=args.energy_per_atom,
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
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))
    if args.ddstore:
        opt = {"ddstore_width": args.ddstore_width}
        trainset = DistDataset(trainset, "trainset", comm, **opt)
        valset = DistDataset(valset, "valset", comm, **opt)
        testset = DistDataset(testset, "testset", comm, **opt)
        # trainset.minmax_node_feature = minmax_node_feature
        # trainset.minmax_graph_feature = minmax_graph_feature
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
