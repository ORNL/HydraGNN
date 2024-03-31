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
from hydragnn.utils.print_utils import iterate_tqdm, log
from hydragnn.utils.time_utils import Timer

from hydragnn.utils.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg

import numpy as np

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance
import torch
import torch.distributed as dist

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr

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


from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
create_graph_fromXYZ = RadiusGraph(r=5.0)  # radius cutoff in angstrom
compute_edge_lengths = Distance(norm=False, cat=True)


class QM7XDataset(AbstractBaseDataset):
    """QM7-XDataset dataset class"""

    def __init__(self, dirpath, var_config, energy_per_atom=True, dist=False):
        super().__init__()

        self.qm7x_node_types = qm7x_node_types
        self.var_config = var_config
        self.energy_per_atom = energy_per_atom

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

    def read_setids(self, dirpath, setids_files):

        for setid in setids_files:
            ## load HDF5 file
            fMOL = h5py.File(dirpath + "/" + setid, "r")

            ## get IDs of HDF5 files and loop through
            mol_ids = list(fMOL.keys())

            if self.dist:
                ## Random shuffle dirlist to avoid the same test/validation set
                random.seed(43)
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
            xyz = torch.from_numpy(np.array(fMOL[molid][confid]["atXYZ"])).to(
                torch.float32
            )
            Z = (
                torch.Tensor(fMOL[molid][confid]["atNUM"])
                .unsqueeze(1)
                .to(torch.float32)
            )

            natoms = xyz.shape[0]

            ## get quantum mechanical properties and add them to properties buffer
            forces = torch.from_numpy(np.array(fMOL[molid][confid]["pbe0FOR"])).to(
                torch.float32
            )
            Eatoms = (
                torch.tensor(sum([EPBE0_atom[zi.item()] for zi in Z]))
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

            data = Data(pos=xyz, x=Z)
            data.x = torch.cat((data.x, xyz, forces, hCHG, hVDIP, hRAT), dim=1)

            if self.energy_per_atom:
                data.y = EPBE0 / natoms
            else:
                data.y = EPBE0

            data = create_graph_fromXYZ(data)

            # Add edge length as edge feature
            data = compute_edge_lengths(data)
            data.edge_attr = data.edge_attr.to(torch.float32)

            subset.append(data)

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
        default=True,
    )

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

    log_name = "qm7x" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "qm7x"
    if args.preonly:

        ## local data
        total = QM7XDataset(
            os.path.join(datadir),
            var_config,
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
