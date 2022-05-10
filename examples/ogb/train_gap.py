import os, json
import matplotlib.pyplot as plt
from ogb_utils import *

import logging
import sys
from tqdm import tqdm
import mpi4py

from mpi4py import MPI
from itertools import chain
import argparse
import time

from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch

import warnings

from torch_geometric.data import download_url, extract_tar

warnings.filterwarnings("error")


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
                offset = [
                    0,
                ] * len(val.shape)
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
                    [
                        sum(ns),
                    ],
                    [
                        ns_offset,
                    ],
                    [
                        len(vcount),
                    ],
                    ad2.ConstantDims,
                )
                self.writer.Put(var, vcount, ad2.Mode.Sync)

                var = self.io.DefineVariable(
                    "%s/%s/variable_offset" % (label, k),
                    offset_arr,
                    [
                        sum(ns),
                    ],
                    [
                        ns_offset,
                    ],
                    [
                        len(vcount),
                    ],
                    ad2.ConstantDims,
                )
                self.writer.Put(var, offset_arr, ad2.Mode.Sync)

        self.writer.Close()
        t1 = time.time()
        info("Adios saving time (sec): ", (t1 - t0))


class OGBDataset(torch.utils.data.Dataset):
    def __init__(self, filename, label, comm, fullmemcache=False):
        self.url = (
            "https://dl.dropboxusercontent.com/s/7qe3zppbicw9vxj/ogb_gap.bp.tar.gz"
        )
        t0 = time.time()
        self.filename = filename
        self.label = label
        self.comm = comm
        self.rank = comm.Get_rank()
        self.fullmemcache = fullmemcache

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
                if self.fullmemcache:
                    ## load full data first
                    self.data[k] = f.read("%s/%s" % (label, k))
            t2 = time.time()
            info("Adios reading time (sec): ", (t2 - t0))
        t1 = time.time()
        info("Data loading time (sec): ", (t1 - t0))

        if not self.fullmemcache:
            self.f = ad2.open(self.filename, "r", MPI.COMM_SELF)

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
                ishape = [int(x.strip(",")) for x in shape.strip().split()]
                start = [
                    0,
                ] * len(ishape)
                count = ishape
                vdim = self.variable_dim[k]
                start[vdim] = self.variable_offset[k][idx]
                count[vdim] = self.variable_count[k][idx]
                if self.fullmemcache:
                    slice_list = list()
                    for n0, n1 in zip(start, count):
                        slice_list.append(slice(n0, n0 + n1))
                    val = self.data[k][tuple(slice_list)]
                else:
                    val = self.f.read("%s/%s" % (self.label, k), start, count)

                v = torch.tensor(val)
                exec("data_object.%s = v" % (k))
                self.data_object[idx] = data_object
        return data_object

    def __del__(self):
        if not self.fullmemcache:
            self.f.close()


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfilesubstr", help="input file substr")
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios saving and no train",
    )
    args = parser.parse_args()

    graph_feature_names = ["GAP"]
    dirpwd = os.path.dirname(__file__)
    datafile = os.path.join(dirpwd, "dataset/pcqm4m_gap.csv")
    trainset_statistics = os.path.join(dirpwd, "dataset/statistics.pkl")
    ##################################################################################################################
    inputfilesubstr = args.inputfilesubstr
    input_filename = os.path.join(dirpwd, "ogb_" + inputfilesubstr + ".json")
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
    var_config["input_node_feature_names"] = node_attribute_names
    ymax_feature, ymin_feature, ymean_feature, ystd_feature = get_trainset_stat(
        trainset_statistics
    )
    var_config["ymean"] = ymean_feature.tolist()
    var_config["ystd"] = ystd_feature.tolist()
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

    if args.preonly:
        norm_yflag = False  # True
        smiles_sets, values_sets = datasets_load(
            datafile, sampling=args.sampling, seed=43
        )
        import pdb

        pdb.set_trace()
        info([len(x) for x in values_sets])
        dataset_lists = [[] for dataset in values_sets]
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (
                    valueset - torch.tensor(var_config["ymean"])
                ) / torch.tensor(var_config["ystd"])
                print(valueset[:, 0].mean(), valueset[:, 0].std())
                print(valueset[:, 1].mean(), valueset[:, 1].std())
                print(valueset[:, 2].mean(), valueset[:, 2].std())

            rx = list(nsplit(range(len(smileset)), comm_size))[rank]
            info("subset range:", idataset, len(smileset), rx.start, rx.stop)
            ## local portion
            _smileset = smileset[rx.start : rx.stop]
            _valueset = valueset[rx.start : rx.stop]
            info("local smileset size:", len(_smileset))

            for smilestr, ytarget in iterate_tqdm(
                zip(_smileset, _valueset), verbosity, total=len(_smileset)
            ):
                data = generate_graphdata(smilestr, ytarget, var_config)
                dataset_lists[idataset].append(data)

        ## local data
        _trainset = dataset_lists[0]
        _valset = dataset_lists[1]
        _testset = dataset_lists[2]

        adwriter = AdioGGO("examples/ogb/dataset/ogb_gap.bp", comm)
        adwriter.add("trainset", _trainset)
        adwriter.add("valset", _valset)
        adwriter.add("testset", _testset)
        adwriter.save()

        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    trainset = OGBDataset("examples/ogb/dataset/ogb_gap.bp", "trainset", comm)
    valset = OGBDataset("examples/ogb/dataset/ogb_gap.bp", "valset", comm)
    testset = OGBDataset("examples/ogb/dataset/ogb_gap.bp", "testset", comm)

    info("Adios load")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = "ogb_" + inputfilesubstr + "_eV_fullx"
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)
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
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)

    ##################################################################################################################
    for ifeat in range(len(var_config["output_index"])):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        plt.subplots_adjust(
            left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
        )
        ax = axs[0]
        ax.scatter(
            range(len(trainset)),
            [trainset[i].y[ifeat].item() for i in range(len(trainset))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("train, " + str(len(trainset)))
        ax = axs[1]
        ax.scatter(
            range(len(valset)),
            [valset[i].y[ifeat].item() for i in range(len(valset))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("validate, " + str(len(valset)))
        ax = axs[2]
        ax.scatter(
            range(len(testset)),
            [testset[i].y[ifeat].item() for i in range(len(testset))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("test, " + str(len(testset)))
        fig.savefig(
            "./logs/"
            + log_name
            + "/ogb_train_val_test_"
            + var_config["output_names"][ifeat]
            + ".png"
        )
        plt.close()

    for ifeat in range(len(var_config["input_node_features"])):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        plt.subplots_adjust(
            left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
        )
        ax = axs[0]
        ax.plot(
            [
                item
                for i in range(len(trainset))
                for item in trainset[i].x[:, ifeat].tolist()
            ],
            "bo",
        )
        ax.set_title("train, " + str(len(trainset)))
        ax = axs[1]
        ax.plot(
            [
                item
                for i in range(len(valset))
                for item in valset[i].x[:, ifeat].tolist()
            ],
            "bo",
        )
        ax.set_title("validate, " + str(len(valset)))
        ax = axs[2]
        ax.plot(
            [
                item
                for i in range(len(testset))
                for item in testset[i].x[:, ifeat].tolist()
            ],
            "bo",
        )
        ax.set_title("test, " + str(len(testset)))
        fig.savefig(
            "./logs/"
            + log_name
            + "/ogb_train_val_test_"
            + var_config["input_node_feature_names"][ifeat]
            + ".png"
        )
        plt.close()
