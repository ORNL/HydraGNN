import os, json
import matplotlib.pyplot as plt
import random
import pickle

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

from rdkit.Chem.rdmolfiles import MolFromPDBFile

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer

# from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_rdkit_molecule,
)

import numpy as np

# import adios2 as ad2

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

warnings.filterwarnings("error")

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
dftb_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def dftb_datasets_load(dirpath, sampling=None, seed=None, frac=[0.9, 0.05, 0.05]):
    if seed is not None:
        random.seed(seed)
    molecules_all = []
    values_all = []
    for subdir, dirs, files in os.walk(dirpath):
        for dir in dirs:
            # collect information about molecular structure and chemical composition
            try:
                pdb_filename = dirpath + "/" + dir + "/" + "smiles.pdb"
                mol = MolFromPDBFile(
                    pdb_filename, sanitize=False, proximityBonding=True, removeHs=True
                )  # , sanitize=False , removeHs=False)
            # file not found -> exit here
            except IOError:
                print(f"'{pdb_filename}'" + " not found")
                sys.exit(1)

            try:
                spectrum_filename = dirpath + "/" + dir + "/" + "EXC-smooth.DAT"
                spectrum_energies = list()
                with open(spectrum_filename, "r") as input_file:
                    count_line = 0
                    for line in input_file:
                        spectrum_energies.append(float(line.strip().split()[1]))

            # file not found -> exit here
            except IOError:
                print(f"'{spectrum_filename}'" + " not found")
                sys.exit(1)

            molecules_all.append(mol)
            values_all.append(spectrum_energies)
    print("Total:", len(molecules_all), len(values_all))

    a = list(range(len(molecules_all)))
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
        trainsmiles.append(molecules_all[i])
        trainset.append(values_all[i])

    for i in ix1:
        valsmiles.append(molecules_all[i])
        valset.append(values_all[i])

    for i in ix2:
        testsmiles.append(molecules_all[i])
        testset.append(values_all[i])

    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
        np.mean(values_all),
        np.std(values_all),
    )


## Torch Dataset for DFTB data with subdirectories
class DFTBDatasetFactory:
    def __init__(
        self, datafile, sampling=1.0, seed=43, var_config=None, norm_yflag=False
    ):
        self.var_config = var_config

        ## Read full data
        (
            molecule_sets,
            values_sets,
            ymean_feature,
            ystd_feature,
        ) = dftb_datasets_load(datafile, sampling=sampling, seed=seed)
        ymean = ymean_feature.tolist()
        ystd = ystd_feature.tolist()

        info([len(x) for x in values_sets])
        self.dataset_lists = list()
        for idataset, (molset, valueset) in enumerate(zip(molecule_sets, values_sets)):
            self.dataset_lists.append((molset, valueset))

    def get(self, label):
        ## Set only assigned label data
        labelnames = ["trainset", "valset", "testset"]
        index = labelnames.index(label)

        molset, valueset = self.dataset_lists[index]
        return (molset, valueset)


class DFTBDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.molecule_set, self.valueset = datasetfactory.get(label)
        self.var_config = datasetfactory.var_config

    def __len__(self):
        return len(self.smileset)

    def __getitem__(self, idx):
        mol = self.molecule_set[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata_from_rdkit_molecule(
            mol, ytarget, dftb_node_types, self.var_config
        )
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.set_defaults(format="pickle")
    args = parser.parse_args()

    graph_feature_names = ["spectrum"]
    graph_feature_dim = [351]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(dirpwd, "dataset/dftb_gdb9_electronic_excitation_spectrum")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, "dftb_smooth_uv_spectrum.json")
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

    log_name = "dftb_eV_fullx"
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if args.preonly:

        (
            molecule_sets,
            values_sets,
            ymean_feature,
            ystd_feature,
        ) = dftb_datasets_load(datafile, sampling=args.sampling, seed=43)
        var_config["ymean"] = ymean_feature.tolist()
        var_config["ystd"] = ystd_feature.tolist()

        os.makedirs("dataset/pickle/", exist_ok=True)

        info([len(x) for x in values_sets])
        dataset_lists = [[] for dataset in values_sets]
        for idataset, (molset, valueset) in enumerate(zip(molecule_sets, values_sets)):

            rx = list(nsplit(range(len(molset)), comm_size))[rank]
            info("subset range:", idataset, len(molset), rx.start, rx.stop)
            ## local portion
            _molset = molset[rx.start : rx.stop]
            _valueset = valueset[rx.start : rx.stop]
            info("local molecule set size:", len(_molset))

            setname = ["trainset", "valset", "testset"]
            if args.format == "pickle":
                if rank == 0:
                    with open(
                        "dataset/pickle/%s.meta" % (setname[idataset]),
                        "w+",
                    ) as f:
                        f.write(str(len(molset)))

            for i, (molset, ytarget) in iterate_tqdm(
                enumerate(zip(_molset, _valueset)), verbosity, total=len(_molset)
            ):
                data = generate_graphdata_from_rdkit_molecule(
                    molset, ytarget, dftb_node_types, var_config
                )
                dataset_lists[idataset].append(data)

                ## (2022/07) This is for testing to compare with Adios
                ## pickle write
                if args.format == "pickle":
                    fname = "dataset/pickle/dftb_uv_spectrum-%s-%d.pk" % (
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

            adwriter = AdiosWriter("dataset/dftb_uv_spectrum.bp", comm)
            adwriter.add("trainset", _trainset)
            adwriter.add("valset", _valset)
            adwriter.add("testset", _testset)
            adwriter.save()

        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        trainset = AdiosDataset(
            "dataset/dftb_uv_spectrum.bp",
            "trainset",
            comm,
            preload=False,
            shmem=True,
        )
        valset = AdiosDataset("dataset/dftb_uv_spectrum.bp", "valset", comm)
        testset = AdiosDataset("dataset/dftb_uv_spectrum.bp", "testset", comm)
    elif args.format == "pickle":
        trainset = SimplePickleDataset("dataset/pickle", "dftb_uv_spectrum", "trainset")
        valset = SimplePickleDataset("dataset/pickle", "dftb_uv_spectrum", "valset")
        testset = SimplePickleDataset("dataset/pickle", "dftb_uv_spectrum", "testset")
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

            num_test_samples = len(test_loader.dataset)
            error_mae = 0.0
            error_mse = 0.0

            for sample_id in range(0, num_test_samples):
                error_mae += np.sum(
                    np.abs(
                        head_pred[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                        - head_true[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                    )
                )
                error_mse += np.sum(
                    (
                        head_pred[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                        - head_true[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                    )
                    ** 2
                )

                fig, ax = plt.subplots()
                true_sample = head_true[
                    (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                    * graph_feature_dim[0]
                ]
                pred_sample = head_pred[
                    (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                    * graph_feature_dim[0]
                ]
                ax.plot(true_sample)
                ax.plot(pred_sample)
                plt.draw()
                plt.tight_layout()
                plt.ylim([-0.2, max(true_sample) + 0.2])
                plt.savefig(f"logs/sample_{sample_id}.png")
                plt.close(fig)

            error_mae /= num_test_samples
            error_mse /= num_test_samples
            error_rmse = np.sqrt(error_mse)

            print(varname, ": ev/cm, mae=", error_mae, ", rmse= ", error_rmse)

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
            ax.set_title(setname + "; " + varname + " (eV/cm)", fontsize=16)
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
