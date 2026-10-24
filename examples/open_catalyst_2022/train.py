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
import time
import glob
import logging
import sys
import torch
from mpi4py import MPI
import numpy as np
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn import cli
from hydragnn.data.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess.load_data import split_dataset


from hydragnn.utils.print.print_utils import log

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    pbc_as_tensor,
    should_skip_self_loops,
)
from ase.io import read

from hydragnn.distributed import nsplit


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def bump(g):
    return Data.from_dict(g.__dict__)


# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)

# charge and spin are constant across Open Catalyst 2022 dataset
charge = 0.0  # neutral
spin = 1.0  # singlet


def ase_to_torch_geom(
    atoms_chunk, energy_per_atom, radius_graph, radius_graph_pbc, graphgps_transform
):
    # Require energies and forces to be present
    data_objects = []
    for atoms in atoms_chunk:
        t1 = time.time()
        if (
            atoms.calc is None
            or "energy" not in atoms.calc.results
            or "forces" not in atoms.calc.results
        ):
            continue

        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.float32
        ).unsqueeze(1)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        natoms = torch.tensor([positions.shape[0]], dtype=torch.int32)
        tags = torch.tensor(atoms.get_tags(), dtype=torch.float32)

        try:
            cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float32).view(
                3, 3
            )
        except Exception:
            cell = torch.eye(3, dtype=torch.float32)

        try:
            pbc = pbc_as_tensor(atoms.get_pbc())
        except Exception:
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

        if cell is None or pbc is None:
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

        # Calculate chemical composition
        atomic_number_list = atomic_numbers.tolist()
        assert len(atomic_number_list) == natoms
        ## 118: number of atoms in the periodic table
        hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        energy = atoms.get_potential_energy(apply_constraint=False)
        energy_tensor = torch.tensor(energy, dtype=torch.float32).reshape(1, 1)
        energy_per_atom_tensor = energy_tensor.detach().clone() / natoms

        forces = torch.tensor(
            atoms.get_forces(apply_constraint=False), dtype=torch.float32
        )

        data_object = Data(
            dataset_name="oc2022",
            natoms=natoms,
            pos=positions,
            cell=cell,
            pbc=pbc,
            atomic_numbers=atomic_numbers.long().view(-1),
            chemical_composition=chemical_composition,
            tags=tags,
            energy=energy_tensor,
            energy_per_atom=energy_per_atom_tensor,
            forces=forces,
        )

        if data_object.pbc.any():
            try:
                data_object = radius_graph_pbc(data_object)
                data_object = transform_coordinates_pbc(data_object)
            except Exception:
                data_object = radius_graph(data_object)
                data_object = transform_coordinates(data_object)
        else:
            data_object = radius_graph(data_object)
            data_object = transform_coordinates(data_object)

        # Skip samples that still contain self-loops
        if should_skip_self_loops(data_object, context="oc2022"):
            continue

        if not hasattr(data_object, "edge_shifts"):
            data_object.edge_shifts = torch.zeros(
                (data_object.edge_index.size(1), 3), dtype=torch.float32
            )

        data_object.pbc = data_object.pbc.int()

        if graphgps_transform is not None:
            data_object = graphgps_transform(data_object)

        data_objects.append(data_object)
        time_taken = round(time.time() - t1, 3)
        # print(f"Worker process {os.getpid()} created data object in {time_taken} seconds")
    return data_objects


def _distribute_files(dirpath, data_type, rank, world_size):
    extxyz_files = None
    if rank == 0:
        data_root = os.path.join(dirpath, data_type)
        if not os.path.isdir(data_root):
            raise RuntimeError(f"Data folder not found: {data_root}")
        search_pattern = os.path.join(data_root, "**", "*.extxyz")
        extxyz_files = sorted(glob.glob(search_pattern, recursive=True))
        if len(extxyz_files) == 0:
            raise RuntimeError(f"No EXTXYZ shards found under: {data_root}")
    extxyz_files = MPI.COMM_WORLD.bcast(extxyz_files, root=0)

    local_files_list = list(nsplit(extxyz_files, world_size))[rank]
    log("local shard count", f"rank {rank} -> {len(local_files_list)} shards")
    return local_files_list


def _chunk_iterable(iterable, chunk_size):
    """Split iterable into chunks"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:  # remaining items
        yield chunk


class OpenCatalystDataset(AbstractBaseDataset):
    """Dataset loader for OC22 S2EF shards stored as EXTXYZ."""

    def __init__(
        self,
        dirpath,
        config,
        data_type,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
        sampling_ratio=None,
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.data_path = dirpath
        self.data_type = data_type
        self.energy_per_atom = energy_per_atom
        self.sampling_ratio = sampling_ratio

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )
        self.radius_graph_pbc = RadiusGraphPBC(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        self.dist = dist
        if self.dist and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # distribute files amongst ranks
        local_files_list = _distribute_files(
            dirpath, data_type, self.rank, self.world_size
        )

        data_objects = self._process_local_files(local_files_list)
        for obj in data_objects:
            if self.check_forces_values(obj.forces):
                self.dataset.append(obj)
            else:
                print(
                    f"L2-norm of force tensor exceeds threshold {self.forces_norm_threshold}"
                    f" - atomistic structure: {obj}"
                )

        if self.sampling_ratio is not None:
            if not (0 < self.sampling_ratio <= 1):
                raise ValueError("sampling_ratio must be in (0, 1].")
            target = max(1, int(len(self.dataset) * self.sampling_ratio))
            self.dataset = random.sample(self.dataset, target)

        random.shuffle(self.dataset)

    def _process_local_files(self, local_file_list):
        data_list = []
        os.environ["MPI4PY_RC_INITIALIZE"] = "0"
        ctx = multiprocessing.get_context("spawn")
        convertor = partial(
            ase_to_torch_geom,
            energy_per_atom=self.energy_per_atom,
            radius_graph=self.radius_graph,
            radius_graph_pbc=self.radius_graph_pbc,
            graphgps_transform=self.graphgps_transform,
        )

        nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1
        # with ThreadPoolExecutor(mp_context=ctx, max_workers=nw) as executor:
        with ThreadPoolExecutor(max_workers=nw) as executor:
            for filename in local_file_list:
                try:
                    traj = read(filename, ":", parallel=False)
                except Exception as exc:
                    print(f"Failed to read {filename}: {exc}. Skipping ..", flush=True)
                    continue

                futures = [
                    executor.submit(convertor, chunk)
                    for chunk in _chunk_iterable(traj, 1000)
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {filename}",
                    disable=(self.rank != 0),
                ):
                    data_objects = future.result()
                    if data_objects is not None:
                        data_list.extend(data_objects)

        return data_list

    def check_forces_values(self, forces):
        norms = torch.norm(forces, p=2, dim=1)
        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = cli.build_parser(default_inputfile="open_catalyst_energy.json")
    parser.add_argument("--train_path", default="train_s2ef", help="training data path")
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    config = cli.load_config(os.path.join(dirpwd, args.inputfile), args)
    rank, log_name, writer = cli.setup_run(args, "OC2022")
    comm = MPI.COMM_WORLD
    modelname = args.modelname or "OC2022"
    path = cli.dataset_path(dirpwd, modelname, args.format)

    if args.preonly:
        ## local data
        dataset = OpenCatalystDataset(
            os.path.join(datadir),
            config,
            data_type=args.train_path,
            graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
            sampling_ratio=args.sampling,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=dataset,
            perc_train=0.9,
            stratify_splitting=False,
        )
        ## Need as a list
        trainset = trainset[:]
        valset = valset[:]
        testset = testset[:]

        comm.Barrier()

        print(
            rank,
            "Local splitting: ",
            len(trainset),
            len(valset),
            len(testset),
            flush=True,
        )

        cli.write_split_datasets(path, args.format, trainset, valset, testset)
        sys.exit(0)

    cli.train_from_args(args, config, path, log_name, writer)
    dist.destroy_process_group()
