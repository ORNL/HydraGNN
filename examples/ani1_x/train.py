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

import numpy as np

import random
import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
)
from hydragnn.preprocess.load_data import split_dataset

from hydragnn.distributed import nsplit

import h5py


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# Conversion constant from Hartree to electron volt (eV).
# Source: NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?hrjtoeV
# Value: 1 Hartree = 27.2114079527 eV (use at least 10 significant digits for scientific accuracy)
conversion_constant_from_hartree_to_eV = 27.2114079527

# charge and spin are constant across ANI1-x dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


class ANI1xDataset(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.data_path = os.path.join(dirpath, "ani1x-release.h5")
        self.data_keys = ["wb97x_dz.energy", "wb97x_dz.forces"]
        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.convert_trajectories_to_graphs()

    def convert_trajectories_to_graphs(self):

        # Example for extracting DFT/DZ energies and forces
        for data_trj in iterate_tqdm(
            self.iter_data_buckets(self.data_path, keys=self.data_keys),
            verbosity_level=2,
        ):

            X = data_trj["coordinates"]
            Z = data_trj["atomic_numbers"]
            E = data_trj["wb97x_dz.energy"]
            F = data_trj["wb97x_dz.forces"]

            # atomic numbers
            atomic_numbers = torch.from_numpy(Z).unsqueeze(1).to(torch.float32)

            natoms = torch.IntTensor([X.shape[1]])

            global_trajectories_id = range(X.shape[0])
            if self.dist:
                local_trajectories_id = list(
                    nsplit(global_trajectories_id, self.world_size)
                )[self.rank]
            else:
                local_trajectories_id = global_trajectories_id

            # extract positions, energies, and forces for each step
            for frame_id in local_trajectories_id:

                pos = torch.from_numpy(X[frame_id]).to(torch.float32)
                cell = torch.eye(3, dtype=torch.float32)
                pbc = torch.tensor([False, False, False], dtype=torch.bool)
                energy = (
                    torch.tensor(E[frame_id])
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .to(torch.float32)
                ) * conversion_constant_from_hartree_to_eV

                energy_per_atom = energy.detach().clone() / natoms
                forces = (
                    torch.from_numpy(F[frame_id]).to(torch.float32)
                    * conversion_constant_from_hartree_to_eV
                )
                # Calculate chemical composition
                atomic_number_list = atomic_numbers.tolist()
                assert len(atomic_number_list) == natoms
                ## 118: number of atoms in the periodic table
                hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
                chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)
                pos_list = pos.tolist()
                atomic_number_list_int = [int(item[0]) for item in atomic_number_list]
                """
                try:
                    mol = xyz2mol(
                        atomic_number_list_int,
                        pos_list,
                        charge=0,
                        allow_charged_fragments=True,
                        use_graph=False,
                        use_huckel=False,
                        embed_chiral=True,
                        use_atom_maps=False,
                    )

                    assert (
                        len(mol) == 1
                    ), f"molecule with atomic numbers {atomic_number_list_int}  and positions {pos_list} does not produce RDKit.mol object"
                    smiles_string = Chem.MolToSmiles(mol[0])
                except:
                    smiles_string = None
                """

                data_object = Data(
                    dataset_name="ani1x",
                    natoms=natoms,
                    pos=pos,
                    cell=cell,  # even if not needed, cell needs to be defined because ADIOS requires consistency across datasets
                    pbc=pbc,  # even if not needed, pbc needs to be defined because ADIOS requires consistency across datasets
                    # edge_index=None,
                    # edge_attr=None,
                    atomic_numbers=atomic_numbers.long().view(-1),
                    chemical_composition=chemical_composition,
                    # smiles_string=smiles_string,
                    energy=energy,
                    energy_per_atom=energy_per_atom,
                    forces=forces,
                )

                data_object = self.radius_graph(data_object)

                # Build edge attributes
                data_object = transform_coordinates(data_object)

                # Default edge_shifts for when radius_graph_pbc is not activated
                data_object.edge_shifts = torch.zeros(
                    (data_object.edge_index.size(1), 3), dtype=torch.float32
                )

                # FIXME: PBC from bool --> int32 to be accepted by ADIOS
                data_object.pbc = data_object.pbc.int()

                # LPE
                if self.graphgps_transform is not None:
                    data_object = self.graphgps_transform(data_object)

                if self.check_forces_values(data_object.forces):
                    self.dataset.append(data_object)
                else:
                    print(
                        f"L2-norm of force tensor exceeds threshold {self.forces_norm_threshold} - atomistic structure: {data_object}",
                        flush=True,
                    )

        random.shuffle(self.dataset)

    def iter_data_buckets(self, h5filename, keys=["wb97x_dz.energy"]):
        """Iterate over buckets of data in ANI HDF5 file.
        Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
        and other available properties specified by `keys` list, w/o NaN values.
        """
        keys = set(keys)
        keys.discard("atomic_numbers")
        keys.discard("coordinates")
        with h5py.File(h5filename, "r") as f:
            for grp in f.values():
                Nc = grp["coordinates"].shape[0]
                mask = np.ones(Nc, dtype=np.bool_)
                data = dict((k, grp[k][()]) for k in keys)
                for k in keys:
                    v = data[k].reshape(Nc, -1)
                    mask = mask & ~np.isnan(v).any(axis=1)
                if not np.sum(mask):
                    continue
                d = dict((k, data[k][mask]) for k in keys)
                d["atomic_numbers"] = grp["atomic_numbers"][()]
                d["coordinates"] = grp["coordinates"][()][mask]
                yield d

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
    parser = cli.build_parser(default_inputfile="ani1x_energy.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "ANI1x")
    modelname = args.modelname or "ANI1x"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        total = ANI1xDataset(
            os.path.join(datadir),
            config,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
