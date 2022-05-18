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
from hydragnn.utils.ogbdataset import AdiosOGB, OGBDataset

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch

try:
    import gptl4py as gp
except ImportError:
    import hydragnn.utils.gptl4py_dummy as gp

import warnings

warnings.filterwarnings("error")


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
        smiles_sets, values_sets, ymean_feature, ystd_feature = csce_datasets_load(
            datafile, sampling=sampling, seed=seed
        )
        ymean = ymean_feature.tolist()
        ystd = ystd_feature.tolist()

        info([len(x) for x in values_sets])
        self.dataset_lists = list()
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (valueset - torch.tensor(ymean)) / torch.tensor(ystd)
                # print(valueset[:, 0].mean(), valueset[:, 0].std())
                # print(valueset[:, 1].mean(), valueset[:, 1].std())
                # print(valueset[:, 2].mean(), valueset[:, 2].std())
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

    @gp.profile
    def __getitem__(self, idx):
        smilestr = self.smileset[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata(smilestr, ytarget, self.var_config)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfilesubstr", help="input file substr")
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios saving and no train",
    )
    parser.add_argument("--noadios", action="store_true", help="no adios dataset")
    args = parser.parse_args()

    graph_feature_names = ["GAP"]
    dirpwd = os.path.dirname(__file__)
    datafile = os.path.join(dirpwd, "dataset/csce_gap_synth.csv")
    # trainset_statistics = os.path.join(dirpwd, "dataset/statistics.pkl")
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
    var_config["input_node_feature_names"] = node_attribute_names
    # ymax_feature, ymin_feature, ymean_feature, ystd_feature = get_trainset_stat(
    #     trainset_statistics
    # )
    # var_config["ymean"] = ymean_feature.tolist()
    # var_config["ystd"] = ystd_feature.tolist()
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

        smiles_sets, values_sets, ymean_feature, ystd_feature = csce_datasets_load(
            datafile, sampling=args.sampling, seed=43
        )
        var_config["ymean"] = ymean_feature.tolist()
        var_config["ystd"] = ystd_feature.tolist()

        info([len(x) for x in values_sets])
        dataset_lists = [[] for dataset in values_sets]
        # import pdb; pdb.set_trace()
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

        adwriter = AdiosOGB("examples/csce/dataset/csce_gap.bp", comm)
        adwriter.add("trainset", _trainset)
        adwriter.add("valset", _valset)
        adwriter.add("testset", _testset)
        adwriter.save()

        sys.exit(0)

    gp.initialize()
    timer = Timer("load_data")
    timer.start()
    if not args.noadios:
        trainset = OGBDataset(
            "examples/csce/dataset/csce_gap.bp",
            "trainset",
            comm,
            preload=False,
            shmem=True,
        )
        valset = OGBDataset("examples/csce/dataset/csce_gap.bp", "valset", comm)
        testset = OGBDataset("examples/csce/dataset/csce_gap.bp", "testset", comm)
    else:
        fact = CSCEDatasetFactory(
            "examples/csce/dataset/csce_gap_synth.csv",
            args.sampling,
            var_config=var_config,
        )
        trainset = CSCEDataset(fact, "trainset")
        valset = CSCEDataset(fact, "valset")
        testset = CSCEDataset(fact, "testset")

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

    gp.pr_file("csce_gp_timing.%d" % rank)
    gp.pr_summary_file("csce_gp_timing.summary")
    gp.finalize()

    if args.shmem:
        trainset.unlink()

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
