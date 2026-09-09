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
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

from mpi4py import MPI
import torch
import torch.distributed as dist
from torch_geometric.data import Data
from torch_geometric.transforms import Distance, RadiusGraph

import hydragnn
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleDataset,
    SimplePickleWriter,
)
from hydragnn.utils.print.print_utils import log


torch.set_default_dtype(torch.float32)

EXAMPLE_DIR = Path(__file__).resolve().parent
RAW_ARCHIVE_DIR = EXAMPLE_DIR / "dataset" / "raw" / "data"
EXTRACTED_DIR = EXAMPLE_DIR / "dataset" / "raw" / "extracted"
PICKLE_DIR = EXAMPLE_DIR / "dataset" / "pubchem_gaussian.pickle"
CACHE_VERSION = "autograd-force-hessian-v1"
ANGSTROM_TO_BOHR = 1.8897261254578281


def parse_structure(path):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    atomic_numbers = torch.tensor([[int(row[0])] for row in rows], dtype=torch.float32)
    positions = torch.tensor(
        [[float(value) for value in row[1:4]] for row in rows], dtype=torch.float32
    )
    return atomic_numbers, positions


def parse_optimization_forces(
    path, atomic_numbers, optimized_positions_angstrom, coordinate_tolerance=5.0e-4
):
    """Return forces for the trajectory geometry matching ``Structure.txt``.

    VIBRANT trajectory atom rows contain atomic number, Cartesian coordinates,
    and three Gaussian force components.  The matching geometry is selected by
    coordinates rather than assuming the last textual block is complete.
    """
    lines = path.read_text().splitlines()
    rows = []
    energies = []
    energy_pattern = re.compile(
        r"(?:SCF\s+Done:.*?=|\bEnergy\b\s*[=:])\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
        re.IGNORECASE,
    )
    for line_number, line in enumerate(lines):
        match = energy_pattern.search(line)
        if match:
            energies.append(
                (line_number, float(match.group(1).replace("D", "E").replace("d", "e")))
            )
        fields = line.replace("D", "E").split()
        if len(fields) < 7:
            rows.append(None)
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            rows.append(None)
            continue
        atomic_number = int(values[0])
        if values[0] != atomic_number or atomic_number <= 0:
            rows.append(None)
            continue
        rows.append((atomic_number, values[1:4], values[-3:]))

    expected_numbers = atomic_numbers.reshape(-1).to(torch.int64).tolist()
    candidates = []
    for start in range(len(rows) - len(expected_numbers) + 1):
        block = rows[start : start + len(expected_numbers)]
        if any(row is None for row in block):
            continue
        if [row[0] for row in block] != expected_numbers:
            continue
        coordinates = torch.tensor([row[1] for row in block], dtype=torch.float32)
        mismatch = torch.max(torch.abs(coordinates - optimized_positions_angstrom))
        candidates.append((float(mismatch), start, block))

    if not candidates:
        raise ValueError(
            f"No complete {len(expected_numbers)}-atom force block in {path}"
        )
    mismatch, block_start, matching_block = min(candidates, key=lambda item: item[0])
    if mismatch > coordinate_tolerance:
        raise ValueError(
            f"Closest Opt_Trj geometry differs from Structure.txt by {mismatch:.3g} Angstrom"
        )
    if not energies:
        # Some XYZ-style trajectories store the energy alone on the comment
        # line instead of prefixing it with ``Energy =``.
        for line_number in range(block_start - 1, -1, -1):
            fields = lines[line_number].replace("D", "E").split()
            if len(fields) != 1:
                continue
            try:
                energies.append((line_number, float(fields[0])))
                break
            except ValueError:
                continue
    if not energies:
        raise ValueError(f"No energy record found for matching geometry in {path}")
    preceding = [item for item in energies if item[0] <= block_start]
    if preceding:
        energy = max(preceding, key=lambda item: item[0])[1]
    else:
        energy = min(energies, key=lambda item: abs(item[0] - block_start))[1]
    forces = torch.tensor([row[2] for row in matching_block], dtype=torch.float32)
    return torch.tensor([[energy]], dtype=torch.float32), forces


def parse_hessian(path, num_atoms):
    dimension = 3 * num_atoms
    hessian = torch.zeros((dimension, dimension), dtype=torch.float32)
    columns = []
    values_read = 0

    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if all(field.isdigit() for field in fields):
            columns = [int(field) - 1 for field in fields]
            continue
        if not columns or not fields[0].isdigit():
            continue

        row = int(fields[0]) - 1
        for column, value in zip(columns, fields[1:]):
            if column <= row:
                hessian[row, column] = float(value.replace("D", "E"))
                values_read += 1

    expected_values = dimension * (dimension + 1) // 2
    if values_read != expected_values:
        raise ValueError(
            f"Expected {expected_values} Hessian values in {path}, found {values_read}"
        )

    hessian = hessian + torch.tril(hessian, diagonal=-1).T
    return hessian


def extract_records(archive_dir, output_dir, limit):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_dir()}
    remaining = max(0, limit - len(existing))
    if remaining == 0:
        return

    archives = sorted(
        archive_dir.glob("*.tar.zst"),
        key=lambda path: int(path.stem.split(".")[0]),
    )
    if not archives:
        raise FileNotFoundError(f"No .tar.zst archives found in {archive_dir}")

    for archive in archives:
        if remaining == 0:
            break
        listing = subprocess.run(
            ["tar", "--zstd", "-tf", str(archive)],
            check=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.splitlines()
        members = [name for name in listing if name.endswith(".tar")]
        members = [name for name in members if Path(name).stem not in existing][:remaining]
        if not members:
            continue

        with tempfile.TemporaryDirectory(dir=output_dir) as temporary_dir:
            subprocess.run(
                ["tar", "--zstd", "-xf", str(archive), "-C", temporary_dir, *members],
                check=True,
            )
            for member in members:
                nested_archive = Path(temporary_dir) / member
                with tarfile.open(nested_archive) as nested:
                    nested.extractall(output_dir)
                existing.add(nested_archive.stem)
                remaining -= 1

    if remaining:
        raise RuntimeError(f"Only extracted {limit - remaining} of {limit} requested records")


class PubChemGaussianDataset(AbstractBaseDataset):
    def __init__(self, root, config, rank=0, world_size=1):
        super().__init__()
        architecture = config["NeuralNetwork"]["Architecture"]
        radius_graph = RadiusGraph(
            architecture["radius"],
            loop=False,
            max_num_neighbors=architecture["max_neighbours"],
        )
        distance = Distance(norm=False, cat=False)
        molecule_dirs = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        )[rank::world_size]

        for molecule_dir in molecule_dirs:
            structure_path = molecule_dir / "Structure.txt"
            trajectory_path = molecule_dir / "Opt_Trj.txt"
            hessian_path = molecule_dir / "Hessian.txt"
            if not all(
                path.exists() for path in (structure_path, trajectory_path, hessian_path)
            ):
                logging.warning(
                    "Skipping CID %s: required files are missing", molecule_dir.name
                )
                continue
            try:
                atomic_numbers, positions_angstrom = parse_structure(structure_path)
                energy, forces = parse_optimization_forces(
                    trajectory_path, atomic_numbers, positions_angstrom
                )
                full_hessian = parse_hessian(hessian_path, atomic_numbers.shape[0])
                positions = positions_angstrom * ANGSTROM_TO_BOHR
                if forces.shape != positions.shape:
                    raise ValueError(
                        f"Force shape {tuple(forces.shape)} does not match "
                        f"position shape {tuple(positions.shape)}"
                    )
                data = Data(
                    dataset_name="pubchem_gaussian",
                    molecule_id=molecule_dir.name,
                    natoms=torch.tensor([atomic_numbers.shape[0]], dtype=torch.int32),
                    atomic_numbers=atomic_numbers,
                    pos=positions,
                    forces=forces,
                    hessian=full_hessian,
                    energy=energy,
                    cell=torch.eye(3, dtype=torch.float32),
                    pbc=torch.zeros(3, dtype=torch.int32),
                )
                data = radius_graph(data)
                data = distance(data)
                data.edge_shifts = torch.zeros((data.num_edges, 3), dtype=torch.float32)
                self.dataset.append(data)
            except (OSError, ValueError) as error:
                logging.warning("Skipping CID %s: %s", molecule_dir.name, error)

    def len(self):
        return len(self.dataset)

    def get(self, index):
        return self.dataset[index]


def preprocess(config, num_molecules, comm, rank, world_size):
    if rank == 0:
        extract_records(RAW_ARCHIVE_DIR, EXTRACTED_DIR, num_molecules)
    comm.Barrier()

    dataset = PubChemGaussianDataset(EXTRACTED_DIR, config, rank, world_size)
    trainset, valset, testset = split_dataset(
        dataset=dataset, perc_train=0.8, stratify_splitting=False
    )
    degree = gather_deg(trainset)

    if rank == 0 and PICKLE_DIR.exists():
        shutil.rmtree(PICKLE_DIR)
    comm.Barrier()
    attributes = {"pna_deg": degree, "cache_version": CACHE_VERSION}
    SimplePickleWriter(trainset, PICKLE_DIR, "trainset", use_subdir=True, attrs=attributes)
    SimplePickleWriter(valset, PICKLE_DIR, "valset", use_subdir=True)
    SimplePickleWriter(testset, PICKLE_DIR, "testset", use_subdir=True)
    log(
        f"Preprocessed {sum(comm.allgather(len(dataset)))} molecules into {PICKLE_DIR}",
        rank=0,
    )


def load_datasets(var_config):
    datasets = tuple(
        SimplePickleDataset(PICKLE_DIR, label, var_config=var_config)
        for label in ("trainset", "valset", "testset")
    )
    if datasets[0].attrs.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("Processed data is stale; rerun with --preonly")
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess VIBRANT PubChem records and train an autograd "
            "force/Hessian model."
        )
    )
    parser.add_argument("--inputfile", default="pubchem_gaussian.json")
    parser.add_argument("--preonly", action="store_true")
    parser.add_argument("--num-molecules", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--log", default="pubchem_gaussian")
    args = parser.parse_args()

    with open(EXAMPLE_DIR / args.inputfile) as config_file:
        config = json.load(config_file)
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if config["NeuralNetwork"]["Training"]["batch_size"] != 1:
        raise ValueError("PubChem Hessian training currently requires --batch-size 1")
    if bool(int(os.getenv("HYDRAGNN_USE_FSDP", "0"))):
        raise ValueError("PubChem Hessian training does not currently support FSDP")
    if int(os.getenv("HYDRAGNN_GRAPH_PARALLEL_GROUP_SIZE", "1")) > 1:
        raise ValueError("PubChem Hessian training does not support graph parallelism")

    world_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % rank,
    )
    hydragnn.utils.print.setup_log(args.log)

    if args.preonly:
        preprocess(config, args.num_molecules, comm, rank, world_size)
        dist.destroy_process_group()
        return

    trainset, valset, testset = load_datasets(config["Variables"])
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
        variables=config["Variables"],
    )
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    config["pna_deg"] = trainset.pna_deg
    hydragnn.utils.input_config_parsing.save_config(config, args.log)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(config["NeuralNetwork"], verbosity)
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1.0e-5
    )
    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity, config=config
    )
    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )
    writer = hydragnn.utils.model.get_summary_writer(args.log)
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config,
        args.log,
        verbosity,
        create_plots=False,
        compute_grad_energy=True,
    )
    hydragnn.utils.model.save_model(model, optimizer, args.log)
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
