import os, json
import matplotlib.pyplot as plt

import logging
import sys
from tqdm import tqdm
import mpi4py
from mpi4py import MPI
from itertools import chain
import argparse
import time

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.preprocess.load_data import dataset_loading_and_splitting

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

from torch_geometric.data import download_url, extract_tar

## For create_configurations
import shutil
from sympy.utilities.iterables import multiset_permutations
import scipy.special
import math

from create_configurations import create_dataset


class AdioGGO:
    def __init__(self, filename, comm):
        self.filename = filename
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dataset = dict()
        self.adios = ad2.ADIOS()
        self.io = self.adios.DeclareIO(self.filename)

    def add(self, label, data: torch_geometric.data.Data):
        if label not in self.dataset:
            self.dataset[label] = list()
        if isinstance(data, list):
            self.dataset[label].extend(data)
        elif isinstance(data, torch_geometric.data.Data):
            self.dataset[label].append(data)
        else:
            raise Exception("Unsuppored data type yet.")

    def save(self):
        t0 = time.time()
        info("Adios saving:", self.filename)
        self.writer = self.io.Open(self.filename, ad2.Mode.Write, self.comm)
        for label in self.dataset:
            if len(self.dataset[label]) < 1:
                continue
            ns = self.comm.allgather(len(self.dataset[label]))
            ns_offset = sum(ns[: self.rank])

            self.io.DefineAttribute("%s/ndata" % label, np.array(sum(ns)))
            if len(self.dataset[label]) > 0:
                data = self.dataset[label][0]
                self.io.DefineAttribute("%s/keys" % label, data.keys)
                keys = sorted(data.keys)

            for k in keys:
                arr_list = [data[k].cpu().numpy() for data in self.dataset[label]]
                m0 = np.min([x.shape for x in arr_list], axis=0)
                m1 = np.max([x.shape for x in arr_list], axis=0)
                wh = np.where(m0 != m1)[0]
                assert len(wh) < 2
                vdim = wh[0] if len(wh) == 1 else 1
                val = np.concatenate(arr_list, axis=vdim)
                assert val.data.contiguous
                shape_list = self.comm.allgather(list(val.shape))
                offset = [0,] * len(val.shape)
                for i in range(rank):
                    offset[vdim] += shape_list[i][vdim]
                global_shape = shape_list[0]
                for i in range(1, self.size):
                    global_shape[vdim] += shape_list[i][vdim]
                # info ("k,val shape", k, global_shape, offset, val.shape)
                var = self.io.DefineVariable(
                    "%s/%s" % (label, k),
                    val,
                    global_shape,
                    offset,
                    val.shape,
                    ad2.ConstantDims,
                )
                self.writer.Put(var, val, ad2.Mode.Sync)

                self.io.DefineAttribute(
                    "%s/%s/variable_dim" % (label, k), np.array(vdim)
                )

                vcount = np.array([x.shape[vdim] for x in arr_list])
                assert len(vcount) == len(self.dataset[label])

                offset_arr = np.zeros_like(vcount)
                offset_arr[1:] = np.cumsum(vcount)[:-1]
                offset_arr += offset[vdim]

                var = self.io.DefineVariable(
                    "%s/%s/variable_count" % (label, k),
                    vcount,
                    [sum(ns),],
                    [ns_offset,],
                    [len(vcount),],
                    ad2.ConstantDims,
                )
                self.writer.Put(var, vcount, ad2.Mode.Sync)

                var = self.io.DefineVariable(
                    "%s/%s/variable_offset" % (label, k),
                    offset_arr,
                    [sum(ns),],
                    [ns_offset,],
                    [len(vcount),],
                    ad2.ConstantDims,
                )
                self.writer.Put(var, offset_arr, ad2.Mode.Sync)

        self.writer.Close()
        t1 = time.time()
        info("Adios saving time (sec): ", (t1 - t0))


class IsingDataset(torch.utils.data.Dataset):
    def __init__(self, filename, label, comm):
        self.url = (
            "https://dl.dropboxusercontent.com/s/7qe3zppbicw9vxj/ogb_gap.bp.tar.gz"
        )
        t0 = time.time()
        self.filename = filename
        self.label = label
        self.comm = comm
        self.rank = comm.Get_rank()

        self.data_object = dict()
        info("Adios reading:", self.filename)

        if self.rank == 0:
            if not os.path.exists(filename):
                self.prefix = os.path.dirname(self.filename)
                self.download()
        comm.Barrier()
        with ad2.open(self.filename, "r", MPI.COMM_SELF) as f:
            self.vars = f.available_variables()
            self.keys = f.read_attribute_string("%s/keys" % label)
            self.ndata = f.read_attribute("%s/ndata" % label).item()

            self.variable_count = dict()
            self.variable_offset = dict()
            self.variable_dim = dict()
            self.data = dict()
            for k in self.keys:
                self.variable_count[k] = f.read("%s/%s/variable_count" % (label, k))
                self.variable_offset[k] = f.read("%s/%s/variable_offset" % (label, k))
                self.variable_dim[k] = f.read_attribute(
                    "%s/%s/variable_dim" % (label, k)
                ).item()
                ## load full data first
                self.data[k] = f.read("%s/%s" % (label, k))
            t2 = time.time()
            info("Adios reading time (sec): ", (t2 - t0))
        t1 = time.time()
        info("Data loading time (sec): ", (t1 - t0))

    def download(self):
        path = download_url(self.url, self.prefix)
        extract_tar(path, self.prefix)
        # os.unlink(path)

    def __len__(self):
        return self.ndata

    def __getitem__(self, idx):
        if idx in self.data_object:
            data_object = self.data_object[idx]
        else:
            data_object = torch_geometric.data.Data()
            for k in self.keys:
                shape = self.vars["%s/%s" % (self.label, k)]["Shape"]
                ishape = [int(x.strip(","), 16) for x in shape.strip().split()]
                start = [0,] * len(ishape)
                count = ishape
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][idx]
                count[vdim] = self.variable_count[k][idx]
                slice_list = list()
                for n0, n1 in zip(start, count):
                    slice_list.append(slice(n0, n0 + n1))
                val = self.data[k][tuple(slice_list)]

                _ = torch.tensor(val)
                exec("data_object.%s = _" % (k))
                self.data_object[idx] = data_object
        return data_object


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios saving and no train",
    )
    parser.add_argument(
        "--natoms", type=int, default=3, help="number_atoms_per_dimension",
    )
    parser.add_argument(
        "--cutoff", type=int, default=1000, help="configurational_histogram_cutoff",
    )
    parser.add_argument(
        "--bpfile", default=1000, help="adios file to read",
    )
    args = parser.parse_args()

    dirpwd = os.path.dirname(__file__)
    input_filename = os.path.join(dirpwd, "ising_model.json")
    with open(input_filename, "r") as f:
        config = json.load(f)

    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    if args.preonly:
        dir = os.path.join(os.path.dirname(__file__), "../../dataset/ising_model")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        number_atoms_per_dimension = args.natoms
        configurational_histogram_cutoff = args.cutoff

        info("Generating ... ")
        info("number_atoms_per_dimension", number_atoms_per_dimension)
        info("configurational_histogram_cutoff", configurational_histogram_cutoff)

        # Use sine function as non-linear extension of Ising model
        # Use randomized scaling of the spin magnitudes
        spin_func = lambda x: math.sin(math.pi * x / 2)
        create_dataset(
            number_atoms_per_dimension,
            configurational_histogram_cutoff,
            dir,
            spin_function=spin_func,
            scale_spin=True,
        )

        hydragnn.preprocess.transform_raw_data_to_serialized(config["Dataset"])
        hydragnn.preprocess.total_to_train_val_test_pkls(config)
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config)

        fname = "ising_%d_%d.bp" % (
            number_atoms_per_dimension,
            configurational_histogram_cutoff,
        )
        adwriter = AdioGGO("examples/ising_model/dataset/%s.bp" % fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        adwriter.save()
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()

    info("Adios load")
    fname = "examples/ising_model/dataset/%s" % (args.bpfile)
    trainset = IsingDataset(fname, "trainset", comm)
    valset = IsingDataset(fname, "valset", comm)
    testset = IsingDataset(fname, "testset", comm)
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (
        train_loader,
        val_loader,
        test_loader,
        sampler_list,
    ) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    timer.stop()

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"], verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
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
        sampler_list,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
