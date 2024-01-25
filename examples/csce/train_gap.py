import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random
import pickle, csv

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
)
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist


csce_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def csce_datasets_load(datafile, sampling=None, seed=None, frac=[0.94, 0.02, 0.04]):
    if seed is not None:
        random.seed(seed)
    smiles_all = []
    values_all = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            if (sampling is not None) and (random.random() > sampling):
                continue
            smiles_all.append(row[1])
            values_all.append([float(row[-2])])
    print("Total:", len(smiles_all), len(values_all))

    a = list(range(len(smiles_all)))
    a = random.sample(a, len(a))
    ix0, ix1, ix2 = np.split(
        a, [int(frac[0] * len(a)), int((frac[0] + frac[1]) * len(a))]
    )

    trainsmiles = []
    valsmiles = []
    testsmiles = []
    trainset = []
    valset = []
    testset = []

    for i in ix0:
        trainsmiles.append(smiles_all[i])
        trainset.append(values_all[i])

    for i in ix1:
        valsmiles.append(smiles_all[i])
        valset.append(values_all[i])

    for i in ix2:
        testsmiles.append(smiles_all[i])
        testset.append(values_all[i])

    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
        np.mean(values_all),
        np.std(values_all),
    )


## Torch Dataset for CSCE CSV format
class CSCEDatasetFactory:
    def __init__(
        self, datafile, sampling=1.0, seed=43, var_config=None, norm_yflag=False
    ):
        self.var_config = var_config

        ## Read full data
        (
            smiles_sets,
            values_sets,
            ymean_feature,
            ystd_feature,
        ) = csce_datasets_load(datafile, sampling=sampling, seed=seed)
        ymean = ymean_feature.tolist()
        ystd = ystd_feature.tolist()

        info([len(x) for x in values_sets])
        self.dataset_lists = list()
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (valueset - torch.tensor(ymean)) / torch.tensor(ystd)
            self.dataset_lists.append((smileset, valueset))

    def get(self, label):
        ## Set only assigned label data
        labelnames = ["trainset", "valset", "testset"]
        index = labelnames.index(label)

        smileset, valueset = self.dataset_lists[index]
        return (smileset, valueset)


class CSCEDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.smileset, self.valueset = datasetfactory.get(label)
        self.var_config = datasetfactory.var_config

    def __len__(self):
        return len(self.smileset)

    def __getitem__(self, idx):
        smilestr = self.smileset[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata_from_smilestr(
            smilestr, ytarget, csce_node_types, self.var_config
        )
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--inputfilesubstr", help="input file substr", default="gap")
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--log", help="log name")

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
        "--csv", help="CSV dataset", action="store_const", dest="format", const="csv"
    )
    parser.set_defaults(format="adios")
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        "--preload",
        help="preload dataset",
        action="store_const",
        dest="dataset",
        const="preload",
    )
    group1.add_argument(
        "--shmem",
        help="shmem dataset",
        action="store_const",
        dest="dataset",
        const="shmem",
    )
    group1.add_argument(
        "--ddstore",
        help="ddstore dataset",
        action="store_const",
        dest="dataset",
        const="ddstore",
    )
    group1.add_argument(
        "--simple",
        help="no special dataset",
        action="store_const",
        dest="dataset",
        const="simple",
    )
    parser.set_defaults(dataset="simple")
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    args = parser.parse_args()

    graph_feature_names = ["GAP"]
    graph_feature_dim = [1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(dirpwd, "dataset/csce_gap_synth.csv")
    ##################################################################################################################
    inputfilesubstr = args.inputfilesubstr
    input_filename = os.path.join(dirpwd, "csce_" + inputfilesubstr + ".json")
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
    ) = get_node_attribute_name(csce_node_types)
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

    log_name = "csce_" + inputfilesubstr + "_eV_fullx"
    if args.log is not None:
        log_name = args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    if args.preonly:
        norm_yflag = False  # True

        (
            smiles_sets,
            values_sets,
            ymean_feature,
            ystd_feature,
        ) = csce_datasets_load(datafile, sampling=args.sampling, seed=43)
        var_config["ymean"] = ymean_feature.tolist()
        var_config["ystd"] = ystd_feature.tolist()

        info([len(x) for x in values_sets])
        dataset_lists = [[] for dataset in values_sets]
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (
                    valueset - torch.tensor(var_config["ymean"])
                ) / torch.tensor(var_config["ystd"])

            rx = list(nsplit(range(len(smileset)), comm_size))[rank]
            info("subset range:", idataset, len(smileset), rx.start, rx.stop)
            ## local portion
            _smileset = smileset[rx.start : rx.stop]
            _valueset = valueset[rx.start : rx.stop]
            info("local smileset size:", len(_smileset))

            setname = ["trainset", "valset", "testset"]

            for i, (smilestr, ytarget) in iterate_tqdm(
                enumerate(zip(_smileset, _valueset)), verbosity, total=len(_smileset)
            ):
                data = generate_graphdata_from_smilestr(
                    smilestr, ytarget, csce_node_types, var_config
                )
                dataset_lists[idataset].append(data)

        trainset = dataset_lists[0]
        valset = dataset_lists[1]
        testset = dataset_lists[2]

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        ## pickle
        basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
        attrs = dict()
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

        fname = os.path.join(os.path.dirname(__file__), "dataset", "csce_gap.bp")
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        preload = shmem = ddstore = False
        if args.dataset == "preload":
            preload = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "torch"
            os.environ["HYDRAGNN_USE_ddstore"] = "0"
        elif args.dataset == "shmem":
            shmem = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "torch"
            os.environ["HYDRAGNN_USE_ddstore"] = "0"
        elif args.dataset == "ddstore":
            ddstore = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
            os.environ["HYDRAGNN_USE_ddstore"] = "1"

        opt = {"preload": preload, "shmem": shmem, "ddstore": ddstore}
        print ("opt:", opt)
        fname = fname = os.path.join(
            os.path.dirname(__file__), "dataset", "csce_gap.bp"
        )
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm)
        testset = AdiosDataset(fname, "testset", comm)
        comm.Barrier()
    elif args.format == "csv":
        fname = os.path.join(os.path.dirname(__file__), "dataset", "csce_gap_synth.csv")
        fact = CSCEDatasetFactory(fname, args.sampling, var_config=var_config)
        trainset = CSCEDataset(fact, "trainset")
        valset = CSCEDataset(fact, "valset")
        testset = CSCEDataset(fact, "testset")
    elif args.format == "pickle":
        basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
        trainset = SimplePickleDataset(basedir, "trainset")
        valset = SimplePickleDataset(basedir, "valset")
        testset = SimplePickleDataset(basedir, "testset")
        pna_deg = trainset.pna_deg
        if args.dataset == "ddstore":
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.dataset == "ddstore":
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    comm.Barrier()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)
    comm.Barrier()

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

    if args.mae:
        import matplotlib.pyplot as plt

        ##################################################################################################################
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for isub, (loader, setname) in enumerate(
            zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )
            ihead = 0
            head_true = np.asarray(true_values[ihead].cpu()).squeeze()
            head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = graph_feature_names[ifeat]

            ax = axs[isub]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            print(varname, ": ev, mae=", error_mae, ", rmse= ", error_rmse)

            ax.scatter(
                head_true,
                head_pred,
                s=7,
                linewidth=0.5,
                edgecolor="b",
                facecolor="none",
            )
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2f}".format(error_mae),
            )
        if rank == 0:
            fig.savefig(os.path.join("logs", log_name, varname + "_all.png"))
        plt.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
