import os, json
import matplotlib.pyplot as plt
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
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset, SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata,
)

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

warnings.filterwarnings("error")


csce_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


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
        data = generate_graphdata(smilestr, ytarget, csce_node_types, self.var_config)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfilesubstr", help="input file substr")
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--mae", action="store_true", help="do mae calculation")

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
    args = parser.parse_args()

    graph_feature_names = ["GAP"]
    graph_feature_dim = [1]
    dirpwd = os.path.dirname(__file__)
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
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

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
            if args.format == "pickle":
                if rank == 0:
                    with open(
                        "examples/csce/dataset/pickle/%s.meta" % (setname[idataset]),
                        "w",
                    ) as f:
                        f.write(str(len(smileset)))

            for i, (smilestr, ytarget) in iterate_tqdm(
                enumerate(zip(_smileset, _valueset)), verbosity, total=len(_smileset)
            ):
                data = generate_graphdata(
                    smilestr, ytarget, csce_node_types, var_config
                )
                dataset_lists[idataset].append(data)

                ## (2022/07) This is for testing to compare with Adios
                ## pickle write
                if args.format == "pickle":
                    fname = "examples/csce/dataset/pickle/csce_gap-%s-%d.pk" % (
                        setname[idataset],
                        rx.start + i,
                    )
                    with open(fname, "wb") as f:
                        pickle.dump(data, f)

        ## local data
        if args.format == "adios":
            _trainset = dataset_lists[0]
            _valset = dataset_lists[1]
            _testset = dataset_lists[2]

            adwriter = AdiosWriter("examples/csce/dataset/csce_gap.bp", comm)
            adwriter.add("trainset", _trainset)
            adwriter.add("valset", _valset)
            adwriter.add("testset", _testset)
            adwriter.save()

        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        trainset = AdiosDataset(
            "examples/csce/dataset/csce_gap.bp",
            "trainset",
            comm,
            preload=False,
            shmem=True,
        )
        valset = AdiosDataset("examples/csce/dataset/csce_gap.bp", "valset", comm)
        testset = AdiosDataset("examples/csce/dataset/csce_gap.bp", "testset", comm)
    elif args.format == "csv":
        fact = CSCEDatasetFactory(
            "examples/csce/dataset/csce_gap_synth.csv",
            args.sampling,
            var_config=var_config,
        )
        trainset = CSCEDataset(fact, "trainset")
        valset = CSCEDataset(fact, "valset")
        testset = CSCEDataset(fact, "testset")
    elif args.format == "pickle":
        trainset = SimplePickleDataset(
            "examples/csce/dataset/pickle", "csce_gap", "trainset"
        )
        valset = SimplePickleDataset(
            "examples/csce/dataset/pickle", "csce_gap", "valset"
        )
        testset = SimplePickleDataset(
            "examples/csce/dataset/pickle", "csce_gap", "testset"
        )
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info("Adios load")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

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
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    if args.mae:
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
            fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
        plt.close()

    if args.format == "adios":
        trainset.unlink()

    sys.exit(0)
