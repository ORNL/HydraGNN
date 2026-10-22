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
from mpi4py import MPI

import numpy as np

import random

import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.data.materials import normalize_stress, validate_materials_sample
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
)
from hydragnn.preprocess.load_data import split_dataset


from hydragnn.utils.print.print_utils import iterate_tqdm

from jarvis.db.jsonutils import loadjson
from pymatgen.core.structure import Structure
from jarvis.core.atoms import pmg_to_atoms
from utils.generate_dictionary import generate_dictionary_elements

inverted_dict = generate_dictionary_elements()

from hydragnn.distributed import nsplit


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)

# charge and spin are constant across MPTrj dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


class MPTrjDataset(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
        tmpfs=None,
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )
        self.radius_graph_pbc = RadiusGraphPBC(
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

        d = None
        if tmpfs is None:
            d = loadjson(os.path.join(dirpath, "MPtrj_2022.9_full.json"))
        else:
            d = loadjson(os.path.join(tmpfs, "MPtrj_2022.9_full.json"))

        mpids = list(d.keys())

        dataset = []

        if not self.dist:
            mpids_loc = mpids
        else:
            mpids_loc = list(nsplit(mpids, self.world_size))[self.rank]

        for i in iterate_tqdm(mpids_loc, verbosity_level=2, desc="Load"):

            tmp = d[i]

            for j, k in tmp.items():

                info = {}

                info["jid"] = j

                if self.energy_per_atom:
                    info["total_energy"] = k["energy_per_atom"]
                else:
                    info["total_energy"] = k["corrected_total_energy"]

                info["forces"] = k["force"]

                info["stresses"] = k["stress"]

                info["atoms"] = pmg_to_atoms(
                    Structure.from_dict(k["structure"])
                ).to_dict()

                info["magmom"] = k["magmom"]

                # Convert lists to PyTorch tensors
                lattice_mat = None
                pbc = None
                # MPTrj does not define pbc in its samples because they are all implicitly 3D-periodic
                # Therefore, we apply pbc if we can read the cell and default otherwise
                try:
                    lattice_mat = torch.tensor(
                        info["atoms"]["lattice_mat"], dtype=torch.float32
                    ).view(3, 3)
                    pbc = torch.tensor([True, True, True], dtype=torch.bool)
                except:
                    print(f"Structure does not have lattice_mat", flush=True)
                    lattice_mat = torch.eye(3, dtype=torch.float32)
                    pbc = torch.tensor([False, False, False], dtype=torch.bool)

                coords = torch.tensor(info["atoms"]["coords"], dtype=torch.float32)

                # Multiply 'coords' by 'lattice_mat'
                pos = torch.matmul(coords, lattice_mat)

                natoms = torch.IntTensor([pos.shape[0]])

                # Extracting data from info dictionary
                total_energy = info["total_energy"]
                forces = info["forces"]
                stresses = info["stresses"]
                magmom = info["magmom"]
                atoms_dict = info["atoms"]

                # Converting positions and atomic numbers to torch tensors
                atomic_numbers = torch.tensor(
                    [inverted_dict[element] for element in atoms_dict["elements"]],
                    dtype=torch.float32,
                ).view(-1, 1)
                energy = torch.tensor(total_energy, dtype=torch.float32).reshape(1, 1)
                energy_per_atom = energy.detach().clone() / natoms
                forces = torch.tensor(forces, dtype=torch.float32)
                stress = normalize_stress(
                    stresses,
                    source_unit="kbar",
                    source_sign="compression_positive",
                )
                # Calculate chemical composition
                atomic_number_list = atomic_numbers.tolist()
                assert len(atomic_number_list) == natoms
                ## 118: number of atoms in the periodic table
                hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
                chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

                # Creating the Data object
                data_object = Data(
                    dataset_name="mptrj",
                    natoms=natoms,
                    pos=pos,
                    cell=lattice_mat,
                    pbc=pbc,
                    edge_index=None,
                    edge_attr=None,
                    atomic_numbers=atomic_numbers.long().view(-1),
                    chemical_composition=chemical_composition,
                    smiles_string=None,
                    energy=energy,
                    energy_per_atom=energy_per_atom,
                    stress=stress,
                    # magmom=torch.tensor(magmom, dtype=torch.float32),
                    forces=forces,
                )

                if data_object.pbc.any():
                    try:
                        data_object = self.radius_graph_pbc(data_object)
                        data_object = transform_coordinates_pbc(data_object)
                    except:
                        print(
                            f"Structure could not successfully apply one or both of the pbc radius graph and positional transform",
                            flush=True,
                        )
                        data_object = self.radius_graph(data_object)
                        data_object = transform_coordinates(data_object)
                else:
                    data_object = self.radius_graph(data_object)
                    data_object = transform_coordinates(data_object)

                try:
                    validate_materials_sample(data_object, require_stress=True)
                except ValueError as error:
                    print(f"Skipping sample: {error}", flush=True)
                    continue

                # Default edge_shifts for when radius_graph_pbc is not activated
                if not hasattr(data_object, "edge_shifts"):
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
    parser = cli.build_parser(default_inputfile="mptrj_energy.json")
    parser.add_argument("--tmpfs", default=None, help="tmpfs directory")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "MPTrj")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "MPTrj"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        total = MPTrjDataset(
            os.path.join(datadir),
            config,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
            tmpfs=args.tmpfs,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        print("Before COMM.Barrier()", flush=True)
        comm.Barrier()
        print("After COMM.Barrier()", flush=True)

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
