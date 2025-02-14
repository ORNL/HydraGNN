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

import hydragnn
from hydragnn.utils.print.print_utils import iterate_tqdm, log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer

from hydragnn.utils.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

import numpy as np

import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance, Spherical, LocalCartesian
from torch_geometric.transforms import AddLaplacianEigenvectorPE


try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.utils.distributed import nsplit
import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.descriptors_and_embeddings import xyz2mol

from rdkit import Chem

# FIXME: this works fine for now because we train on QM7-X molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
qm7x_node_types = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "Cl": 5}

torch.set_default_dtype(torch.float32)

EPBE0_atom = {
    6: -1027.592489146,
    17: -12516.444619523,
    1: -13.641404161,
    7: -1484.274819088,
    8: -2039.734879322,
    16: -10828.707468187,
}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=False)
transform_coordinates = LocalCartesian(norm=False, cat=False)
# transform_coordinates = Distance(norm=False, cat=False)


from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset


class QM7XDataset(AbstractBaseDataset):
    """QM7-XDataset datasets class"""

    def __init__(
        self,
        dirpath,
        var_config,
        graphgps_transform=None,
        energy_per_atom=True,
        dist=False,
    ):
        super().__init__()

        self.qm7x_node_types = qm7x_node_types
        self.var_config = var_config
        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(5.0, loop=False, max_num_neighbors=50)

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        if os.path.isdir(dirpath):
            dirfiles = sorted(os.listdir(dirpath))
        else:
            filelist = dirpath
            info("Reading filelist:", filelist)
            dirpath = os.path.dirname(dirpath)
            dirlist = list()
            with open(filelist, "r") as f:
                lines = f.readlines()
                for line in lines:
                    dirlist.append(line.rstrip())

        setids_files = [x for x in dirfiles if x.endswith("hdf5")]

        self.read_setids(dirpath, setids_files)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def read_setids(self, dirpath, setids_files):

        for setid in setids_files:
            ## load HDF5 file
            fMOL = h5py.File(dirpath + "/" + setid, "r")

            ## get IDs of HDF5 files and loop through
            mol_ids = list(fMOL.keys())

            if self.dist:
                random.shuffle(mol_ids)

                x = torch.tensor(len(mol_ids), requires_grad=False).to(get_device())
                y = x.clone().detach().requires_grad_(False)
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
                assert x == y
                log("molecule dirlist", len(mol_ids))

            for mol_id in iterate_tqdm(mol_ids, verbosity_level=2, desc="Load"):
                self.dataset.extend(self.hdf5_to_graph(fMOL, mol_id))

    def hdf5_to_graph(self, fMOL, molid):

        subset = []

        ## get IDs of individual configurations/conformations of molecule
        conf_ids = list(fMOL[molid].keys())

        rx = list(nsplit(range(len(conf_ids)), comm_size))[rank]

        for confid in conf_ids[rx.start : rx.stop]:
            ## get atomic positions and numbers
            pos = torch.from_numpy(np.array(fMOL[molid][confid]["atXYZ"])).to(
                torch.float32
            )
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)
            edge_shifts = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            atomic_numbers = (
                torch.Tensor(fMOL[molid][confid]["atNUM"])
                .unsqueeze(1)
                .to(torch.float32)
            )

            natoms = torch.IntTensor([pos.shape[0]])

            ## get quantum mechanical properties and add them to properties buffer
            forces = torch.from_numpy(np.array(fMOL[molid][confid]["pbe0FOR"])).to(
                torch.float32
            )
            Eatoms = (
                torch.tensor(sum([EPBE0_atom[zi.item()] for zi in atomic_numbers]))
                .unsqueeze(0)
                .to(torch.float32)
            )  # eatoms
            EPBE0 = (
                torch.tensor(float(list(fMOL[molid][confid]["ePBE0"])[0]))
                .unsqueeze(0)
                .to(torch.float32)
            )  # energy
            EMBD = (
                torch.tensor(float(list(fMOL[molid][confid]["eMBD"])[0]))
                .unsqueeze(0)
                .to(torch.float32)
            )  # embd
            hCHG = torch.from_numpy(np.array(fMOL[molid][confid]["hCHG"])).to(
                torch.float32
            )  # charge
            POL = float(list(fMOL[molid][confid]["mPOL"])[0])
            hVDIP = torch.from_numpy(np.array(fMOL[molid][confid]["hVDIP"])).to(
                torch.float32
            )  # dipole moment
            HLGAP = (
                torch.tensor(fMOL[molid][confid]["HLgap"])
                .unsqueeze(1)
                .to(torch.float32)
            )  # HL gap
            hRAT = torch.from_numpy(np.array(fMOL[molid][confid]["hRAT"])).to(
                torch.float32
            )  # hirshfeld ratios

            try:
                # check forces values
                assert self.check_forces_values(
                    forces
                ), f"qm7x dataset - molid:{molid} - confid:{confid} - L2-norm of atomic forces exceeds {self.forces_norm_threshold}"

                energy = torch.tensor(EPBE0, dtype=torch.float32).unsqueeze(0)
                energy_per_atom = energy.detach().clone() / natoms

                x = torch.cat((atomic_numbers, pos, forces, hCHG, hVDIP, hRAT), dim=1)

                # Calculate chemical composition
                atomic_number_list = atomic_numbers.tolist()
                assert len(atomic_number_list) == natoms
                ## 118: number of atoms in the periodic table
                hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
                chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)
                pos_list = pos.tolist()
                atomic_number_list_int = [int(item[0]) for item in atomic_number_list]
                """
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
                """

                data_object = Data(
                    # dataset_name="qm7x",
                    dataset_name=torch.IntTensor([1]),
                    natoms=natoms,
                    pos=pos,
                    # cell=None,  # even if not needed, cell needs to be defined because ADIOS requires consistency across datasets
                    # pbc=None,  # even if not needed, pbc needs to be defined because ADIOS requires consistency across datasets
                    # edge_index=None,
                    # edge_attr=None,
                    # edge_shifts=None,  # even if not needed, edge_shift needs to be defined because ADIOS requires consistency across datasets
                    atomic_numbers=atomic_numbers,  # Reshaping atomic_numbers to Nx1 tensor
                    chemical_composition=chemical_composition,
                    # smiles_string=smiles_string,
                    x=x,
                    energy=energy,
                    energy_per_atom=energy_per_atom,
                    forces=forces,
                )

                if self.energy_per_atom:
                    data_object.y = data_object.energy_per_atom
                else:
                    data_object.y = data_object.energy

                data_object = self.radius_graph(data_object)

                data_object = transform_coordinates(data_object)

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

                subset.append(data_object)
            except AssertionError as e:
                print(f"Assertion error occurred: {e}")

        return subset

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
    parser.add_argument("--inputfile", help="input file", type=str, default="qm7x.json")
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
    node_feature_dims = [1, 3, 3, 1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/QM7-X")
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
    graphgps_transform = None

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

    log_name = "qm7x" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "qm7x"
    if args.preonly:

        ## local data
        total = QM7XDataset(
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
