import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random

import logging
import sys
from mpi4py import MPI
import argparse

from rdkit.Chem.rdmolfiles import MolFromPDBFile

import hydragnn
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.descriptors_and_embeddings.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_rdkit_molecule,
)
from hydragnn.utils.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

from hydragnn.utils.distributed import nsplit
import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
dftb_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def dftb_to_graph(moldir, dftb_node_types, var_config):
    pdb_filename = os.path.join(moldir, "smiles.pdb")
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
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir, "trainset")
        valset = SimplePickleDataset(basedir, "valset")
        testset = SimplePickleDataset(basedir, "testset")
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

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
    if writer is not None:
        writer.close()

    if args.mae:
        import matplotlib.pyplot as plt

        ##################################################################################################################
        for isub, (loader, setname) in enumerate(zip([test_loader], ["test"])):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )

            num_test_samples = len(loader.dataset)

            for head_idx in range(len(graph_feature_dim)):
                ihead = head_idx
                head_true = np.asarray(true_values[ihead].cpu()).squeeze()
                head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = graph_feature_names[ifeat]

                error_mae = 0.0
                error_mse = 0.0

                for sample_id in range(0, num_test_samples):
                    true_sample = head_true[
                        (sample_id * graph_feature_dim[ihead]) : (sample_id + 1)
                        * graph_feature_dim[ihead]
                    ]
                    pred_sample = head_pred[
                        (sample_id * graph_feature_dim[ihead]) : (sample_id + 1)
                        * graph_feature_dim[ihead]
                    ]

                    error_mae += np.sum(np.abs(pred_sample - true_sample))
                    error_mse += np.sum((pred_sample - true_sample) ** 2)

                error_mae /= num_test_samples
                error_mse /= num_test_samples
                error_rmse = np.sqrt(error_mse)

                print(varname, ": ev/cm, mae=", error_mae, ", rmse= ", error_rmse)

                fig, ax = plt.subplots()
                ax.scatter(
                    head_true,
                    head_pred,
                    s=7,
                    linewidth=0.5,
                    edgecolor="b",
                    facecolor="none",
                )
                plt.draw()
                plt.tight_layout()
                plt.savefig(f"logs/{setname}_scatterplot_{varname}.png")
                plt.close(fig)

        for sample_id in range(0, num_test_samples):

            pred_location_peaks = predicted_values[0][
                (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                * graph_feature_dim[0]
            ]
            pred_intensity_peaks = predicted_values[1][
                (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                * graph_feature_dim[0]
            ]

            true_location_peaks = true_values[0][
                (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                * graph_feature_dim[0]
            ]
            true_intensity_peaks = true_values[1][
                (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                * graph_feature_dim[0]
            ]

            fig, ax = plt.subplots()
            ax.vlines(
                x=true_location_peaks,
                ymin=[0] * len(pred_location_peaks),
                ymax=true_intensity_peaks,
                colors="blue",
                ls="-",
                lw=4,
                label="TD-DFTB+",
            )
            ax.vlines(
                x=pred_location_peaks,
                ymin=[0] * len(pred_location_peaks),
                ymax=pred_intensity_peaks,
                colors="orange",
                ls="--",
                lw=4,
                label="HydraGNN",
            )
            plt.draw()
            plt.tight_layout()
            # plt.ylim([-0.2, max(true_sample) + 0.2])
            plt.savefig(f"logs/sample_{sample_id}.png")
            plt.close(fig)

        if rank == 0:
            fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
        plt.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()

    sys.exit(0)
