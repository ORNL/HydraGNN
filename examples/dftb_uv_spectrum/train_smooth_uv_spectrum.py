##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import torch
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import random

import logging
import sys

from rdkit.Chem.rdmolfiles import MolFromPDBFile

import hydragnn
from hydragnn import cli
from hydragnn.utils.print.print_utils import iterate_tqdm, log
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.data.descriptors_and_embeddings.smiles_utils import (
    generate_graphdata_from_rdkit_molecule,
)
from hydragnn.distributed import get_device, nsplit
from hydragnn.preprocess.load_data import split_dataset

import numpy as np

import torch.distributed as dist

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
dftb_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def dftb_to_graph(moldir, dftb_node_types, var_config):
    pdb_filename = os.path.join(moldir, "smiles.pdb")
    mol = MolFromPDBFile(
        pdb_filename, sanitize=False, proximityBonding=True, removeHs=True
    )  # , sanitize=False , removeHs=False)
    spectrum_filename = os.path.join(moldir, "EXC-smooth.DAT")
    ytarget = np.loadtxt(spectrum_filename, usecols=1, dtype=np.float32)
    ytarget = torch.tensor(ytarget)
    data = generate_graphdata_from_rdkit_molecule(
        mol, ytarget, dftb_node_types, var_config=var_config
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
    parser = cli.build_parser(default_inputfile="dftb_smooth_uv_spectrum.json")
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    args = parser.parse_args()

    graph_feature_names = ["spectrum"]
    graph_feature_dim = [37500]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(dirpwd, "dataset/dftb_aisd_electronic_excitation_spectrum")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    verbosity = config["Verbosity"]["level"]
    var_config = config["Variables"]
    rank, log_name, writer = cli.setup_run(args, "dftb_eV_fullx")
    path = cli.dataset_path(
        dirpwd, args.modelname or "dftb_smooth_uv_spectrum", args.format
    )

    if args.preonly:
        ## local data
        total = DFTBDataset(
            os.path.join(datafile, "mollist.txt"),
            dftb_node_types,
            var_config,
            dist=True,
            sampling=args.sampling,
        )
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(len(total), len(trainset), len(valset), len(testset))
        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    loaders = []
    model, config = cli.run_training(
        config,
        *cli.load_split_datasets(
            path,
            args.format,
            shmem=args.shmem,
            ddstore=args.ddstore,
            ddstore_width=args.ddstore_width,
        ),
        log_name,
        writer,
        precision=args.precision,
        everyone=args.everyone,
        variables=False,
        before_train=lambda model, optimizer, split_loaders, config: loaders.extend(
            split_loaders
        ),
    )
    train_loader, val_loader, test_loader = loaders

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
            ifeat = ihead
            outtype = var_config["outputs"][ihead]["level"]
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

    if dist.is_initialized():
        dist.destroy_process_group()
