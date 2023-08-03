import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random
import pickle
import io

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

from rdkit.Chem.rdmolfiles import MolFromPDBFile

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_rdkit_molecule,
)
from hydragnn.utils.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

import hydragnn.utils.tracer as tr

# FIXME: deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import glob
import webdataset as wds

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
dftb_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


from hydragnn.utils.abstractbasedataset import AbstractBaseDataset


def dftb_to_graph(moldir, dftb_node_types, var_config):
    pdb_filename = os.path.join(moldir, "smiles.pdb")
    mol = MolFromPDBFile(
        pdb_filename, sanitize=False, proximityBonding=True, removeHs=True
    )
    spectrum_filename = os.path.join(moldir, "EXC.DAT")
    ytarget = np.loadtxt(
        spectrum_filename, skiprows=4, usecols=(0, 1), dtype=np.float32
    )
    ytarget = torch.tensor(ytarget.T.ravel())
    data = generate_graphdata_from_rdkit_molecule(
        mol, ytarget, dftb_node_types, var_config
    )
    data.ID = torch.tensor((int(os.path.basename(moldir).replace("mol_", "")),))
    return data


class DFTBDataset(AbstractBaseDataset):
    """DFTBDataset dataset class"""

    def __init__(self, dirpath, dftb_node_types, var_config, dist=False, sampling=None):
        super().__init__()

        self.dftb_node_types = dftb_node_types
        self.var_config = var_config
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        if os.path.isdir(dirpath):
            dirlist = sorted(os.listdir(dirpath))
        else:
            filelist = dirpath
            info("Reading filelist:", filelist)
            dirpath = os.path.dirname(dirpath)
            dirlist = list()
            with open(filelist, "r") as f:
                lines = f.readlines()
                for line in lines:
                    dirlist.append(line.rstrip())

        if self.dist:
            ## Random shuffle dirlist to avoid the same test/validation set
            random.seed(43)
            random.shuffle(dirlist)
            if sampling is not None:
                dirlist = np.random.choice(dirlist, int(len(dirlist) * sampling))

            x = torch.tensor(len(dirlist), requires_grad=False).to(get_device())
            y = x.clone().detach().requires_grad_(False)
            torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
            assert x == y
            dirlist = list(nsplit(dirlist, self.world_size))[self.rank]
            log("local dirlist", len(dirlist))

        for subdir in iterate_tqdm(dirlist, verbosity_level=2, desc="Load"):
            data_object = dftb_to_graph(
                os.path.join(dirpath, subdir), dftb_node_types, var_config
            )
            self.dataset.append(data_object)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


def decoder(sample):
    data_object = pickle.load(io.BytesIO(sample["pkl"]))
    return data_object


def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


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
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--distds", action="store_true", help="distds dataset")
    parser.add_argument("--distds_width", type=int, help="distds width", default=None)
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
    group.add_argument(
        "--webdataset",
        help="webdataset",
        action="store_const",
        dest="format",
        const="webdataset",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["frequencies", "intensities"]
    graph_feature_dim = [50, 50]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(dirpwd, "dataset/dftb_aisd_electronic_excitation_spectrum")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, "dftb_discrete_uv_spectrum.json")
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_names"] = [
        graph_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dim
    (
        var_config["input_node_feature_names"],
        var_config["input_node_feature_dims"],
    ) = get_node_attribute_name(dftb_node_types)
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

    log_name = "dftb_eV_fullx" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "dftb_discrete_uv_spectrum"
    if args.preonly:
        ## local data
        total = DFTBDataset(
            os.path.join(datafile, "mollist.txt"),
            dftb_node_types,
            var_config,
            dist=True,
        )
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        ## adios
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

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
        sys.exit(0)

    tr.initialize()
    tr.start("preload")
    num_epoch = config["NeuralNetwork"]["Training"]["num_epoch"]
    batch_size = config["NeuralNetwork"]["Training"]["batch_size"]
    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.distds), "Cannot use both distds and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "distds": args.distds,
            "distds_width": args.distds_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir, "trainset")
        if args.distds:
            opt = {"distds_width": args.distds_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
    elif args.format == "webdataset":
        info("WebDataset load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        urls = glob.glob(os.path.join(basedir, "*.tar"))
        trainset = (
            wds.WebDataset(urls, nodesplitter=nodesplitter)
            .map(decoder)
            .batched(batch_size, partial=False)
        )
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    if args.distds:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_DISTDS"] = "1"

    if args.format == "webdataset":
        train_loader = wds.WebLoader(trainset, num_workers=0, batch_size=None)
    else:
        info("trainset size: %d" % len(trainset))

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        pin_memory = True
        persistent_workers = False
        num_workers = 0
        if os.getenv("HYDRAGNN_NUM_WORKERS") is not None:
            num_workers = int(os.environ["HYDRAGNN_NUM_WORKERS"])
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    tr.stop("preload")

    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)

    loader = train_loader
    use_distds = False
    try:
        use_distds = (
            hasattr(loader.dataset, "ddstore")
            and hasattr(loader.dataset.ddstore, "epoch_begin")
            and bool(int(os.getenv("HYDRAGNN_USE_DISTDS", "0")))
        )
    except:
        pass
    for epoch in range(num_epoch):
        try:
            if getattr(loader.sampler, "set_epoch", None) is not None:
                loader.sampler.set_epoch(epoch)
        except:
            pass

        tr.enable()
        tr.start("dataload")
        if use_distds:
            tr.start("epoch_begin")
            loader.dataset.ddstore.epoch_begin()
            tr.stop("epoch_begin")
        for i, data in enumerate(iterate_tqdm(loader, verbosity, desc="Train")):
            print(rank, i, len(data), len(data[0]))
            if use_distds:
                tr.start("epoch_end")
                loader.dataset.ddstore.epoch_end()
                tr.stop("epoch_end")
            if use_distds:
                tr.start("epoch_begin")
                loader.dataset.ddstore.epoch_begin()
                tr.stop("epoch_begin")
        if use_distds:
            tr.start("epoch_end")
            loader.dataset.ddstore.epoch_end()
            tr.stop("epoch_end")
        tr.stop("dataload")
        if epoch == 0:
            tr.reset()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()

    sys.exit(0)
