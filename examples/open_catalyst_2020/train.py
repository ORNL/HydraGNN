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
import os
import logging
import sys
import torch
from mpi4py import MPI

import numpy as np

import random

import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess.load_data import split_dataset


from utils.atoms_to_graphs import AtomsToGraphs
from utils.preprocess import write_images_to_adios

import subprocess
from hydragnn.distributed import nsplit

## FIMME
torch.backends.cudnn.enabled = False


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


class OpenCatalystDataset(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        data_type,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.data_path = os.path.join(dirpath, data_type)
        self.energy_per_atom = energy_per_atom

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        mx = None
        if self.rank == 0:
            ## Let rank 0 check the number of files and share
            # cmd = f"ls {os.path.join(self.data_path, '*.txt')} | wc -l"
            cmd = f"find {self.data_path} -maxdepth 1 -type f -name '*.txt' | wc -l"
            print("Check the number of files:", cmd)
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            mx = int(out.stdout)
            print("Total number of files:", mx)
        mx = MPI.COMM_WORLD.bcast(mx, root=0)
        if mx == 0:
            raise RuntimeError("No *.txt files found. Did you uncompress?")

        ## We assume file names are "%d.txt"
        rx = list(nsplit(range(mx), self.world_size))[self.rank]
        chunked_txt_files = list()
        for n in rx:
            fname = os.path.join(self.data_path, "%d.txt" % n)
            chunked_txt_files.append(fname)

        if len(chunked_txt_files) == 0:
            print(self.rank, "WARN: No files to process. Continue ...")

        # Initialize feature extractor.
        a2g = AtomsToGraphs(max_neigh=self.max_neighbours, radius=self.radius)

        list_atomistic_structures = write_images_to_adios(
            a2g,
            chunked_txt_files,
            self.data_path,
            energy_per_atom=self.energy_per_atom,
        )

        for item in list_atomistic_structures:
            assert item is not None, item

            # Calculate chemical composition
            atomic_number_list = item.atomic_numbers.tolist()
            assert len(atomic_number_list) == item.natoms
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

            item.chemical_composition = chemical_composition
            item.smiles_string = None

            if self.graphgps_transform is not None:
                item = self.graphgps_transform(item)

            if self.check_forces_values(item.forces):
                self.dataset.append(item)
            else:
                print(
                    f"L2-norm of force tensor exceeds threshold {self.forces_norm_threshold} - atomistic structure: {item}",
                    flush=True,
                )

        random.shuffle(self.dataset)

    def check_forces_values(self, forces):
        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="open_catalyst_energy.json")
    parser.add_argument(
        "--train_path",
        default="s2ef_train_200K_uncompressed",
        help="training data path",
    )
    parser.add_argument(
        "--test_path", default="s2ef_val_id_uncompressed", help="test data path"
    )
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "OC2020")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "OC2020"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        trainset = OpenCatalystDataset(
            os.path.join(datadir),
            config,
            data_type=args.train_path,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset1, valset2 = split_dataset(
            dataset=trainset,
            perc_train=0.9,
            stratify_splitting=False,
        )
        valset = [*valset1, *valset2]
        testset = OpenCatalystDataset(
            os.path.join(datadir),
            config,
            data_type=args.test_path,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## Need as a list
        testset = testset[:]

        comm.Barrier()
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
