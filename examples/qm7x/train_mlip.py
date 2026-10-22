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
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os
import random

import h5py

import logging
import sys

from hydragnn import cli
from hydragnn.utils.print.print_utils import iterate_tqdm, log

from hydragnn.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset

import numpy as np

import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance

from hydragnn.distributed import nsplit

# FIXME: this works fine for now because we train on QM7-X molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
qm7x_node_types = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "Cl": 5}

torch.set_default_dtype(torch.float32)

EPBE0_atom = {
    6: -1027.592489146,
    17: -12516.444619523,
    1: -13.641404161,
    7: -1484.274819088,
    8: -2039.734879322,
    16: -10828.707468187,
}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# charge and spin are constant across QM7-X dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


from hydragnn.data.abstractbasedataset import AbstractBaseDataset


class QM7XDataset(AbstractBaseDataset):
    """QM7-XDataset datasets class"""

    def __init__(
        self,
        dirpath,
        config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
    ):
        super().__init__()

        self.qm7x_node_types = qm7x_node_types

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        if os.path.isdir(dirpath):
            dirfiles = sorted(os.listdir(dirpath))
        else:
            filelist = dirpath
            info("Reading filelist:", filelist)
            dirpath = os.path.dirname(dirpath)
            dirlist = list()
            with open(filelist, "r") as f:
                lines = f.readlines()
                for line in lines:
                    dirlist.append(line.rstrip())

        setids_files = [x for x in dirfiles if x.endswith("hdf5")]

        self.read_setids(dirpath, setids_files)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def read_setids(self, dirpath, setids_files):

        for setid in setids_files:
            ## load HDF5 file
            fMOL = h5py.File(dirpath + "/" + setid, "r")

            ## get IDs of HDF5 files and loop through
            mol_ids = list(fMOL.keys())

            if self.dist:
                random.shuffle(mol_ids)

                x = torch.tensor(len(mol_ids), requires_grad=False).to(get_device())
                y = x.clone().detach().requires_grad_(False)
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
                assert x == y
                log("molecule dirlist", len(mol_ids))

            for mol_id in iterate_tqdm(mol_ids, verbosity_level=2, desc="Load"):
                self.dataset.extend(self.hdf5_to_graph(fMOL, mol_id))

    def hdf5_to_graph(self, fMOL, molid):

        subset = []

        ## get IDs of individual configurations/conformations of molecule
        conf_ids = list(fMOL[molid].keys())

        rx = list(nsplit(range(len(conf_ids)), self.world_size))[self.rank]

        for confid in conf_ids[rx.start : rx.stop]:
            ## get atomic positions and numbers
            pos = torch.from_numpy(np.array(fMOL[molid][confid]["atXYZ"])).to(
                torch.float32
            )
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)
            atomic_numbers = (
                torch.Tensor(fMOL[molid][confid]["atNUM"])
                .unsqueeze(1)
                .to(torch.float32)
            )

            natoms = torch.IntTensor([pos.shape[0]])

            ## get quantum mechanical properties and add them to properties buffer
            forces = torch.from_numpy(np.array(fMOL[molid][confid]["pbe0FOR"])).to(
                torch.float32
            )
            Eatoms = (
                torch.tensor(sum([EPBE0_atom[zi.item()] for zi in atomic_numbers]))
                .unsqueeze(0)
                .to(torch.float32)
            )  # eatoms
            EPBE0 = (
                torch.tensor(float(list(fMOL[molid][confid]["ePBE0"])[0]))
                .unsqueeze(0)
                .to(torch.float32)
            )  # energy
            EMBD = (
                torch.tensor(float(list(fMOL[molid][confid]["eMBD"])[0]))
                .unsqueeze(0)
                .to(torch.float32)
            )  # embd
            hCHG = torch.from_numpy(np.array(fMOL[molid][confid]["hCHG"])).to(
                torch.float32
            )  # charge
            POL = float(list(fMOL[molid][confid]["mPOL"])[0])
            hVDIP = torch.from_numpy(np.array(fMOL[molid][confid]["hVDIP"])).to(
                torch.float32
            )  # dipole moment
            HLGAP = (
                torch.tensor(fMOL[molid][confid]["HLgap"])
                .unsqueeze(1)
                .to(torch.float32)
            )  # HL gap
            hRAT = torch.from_numpy(np.array(fMOL[molid][confid]["hRAT"])).to(
                torch.float32
            )  # hirshfeld ratios

            try:
                # check forces values
                assert self.check_forces_values(
                    forces
                ), f"qm7x dataset - molid:{molid} - confid:{confid} - L2-norm of atomic forces exceeds {self.forces_norm_threshold}"

                energy = torch.tensor(EPBE0, dtype=torch.float32).unsqueeze(0)
                energy_per_atom = energy.detach().clone() / natoms

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
                    dataset_name="qm7x",
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
                        f"L2-norm of force tensor is {data_object.forces.norm()} and exceeds threshold {self.forces_norm_threshold} - molecule ID: {molid}",
                        flush=True,
                    )

                subset.append(data_object)
            except AssertionError as e:
                print(f"Assertion error occurred: {e}")

        return subset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="qm7x_mlip.json")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset", "QM7-X")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "qm7x")
    modelname = args.modelname or "qm7x"
    path = cli.dataset_path(dirpwd, modelname + "-v2", args.format)

    if args.preonly:
        ## local data
        total = QM7XDataset(
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
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
