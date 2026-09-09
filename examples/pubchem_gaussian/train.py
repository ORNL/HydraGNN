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
CACHE_VERSION = "autograd-trajectory-hessian-v2"
SCF_ENERGY_PATTERN = re.compile(
    r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)"
)
# Coordinate matching is performed in the source unit (Angstrom) before conversion.
OPTIMIZED_POSITION_TOLERANCE = 1.0e-5
# CODATA conversion: 1 Angstrom = 1.8897261254578281 Bohr.
ANGSTROM_TO_BOHR = 1.8897261254578281


def parse_structure(path):
    """Parse atomic numbers and optimized Cartesian coordinates in Angstrom."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    atomic_numbers = torch.tensor([[int(row[0])] for row in rows], dtype=torch.float32)
    positions = torch.tensor(
        [[float(value) for value in row[1:4]] for row in rows], dtype=torch.float32
    )
    return atomic_numbers, positions


def parse_hessian(path, num_atoms):
    """Parse the symmetric Cartesian Hessian in Hartree/Bohr^2.

    Gaussian indexes Cartesian degrees of freedom in atom-major order:
    ``(atom 0 x, atom 0 y, atom 0 z, atom 1 x, ...)``. For positions and
    forces shaped ``(N, 3)``, a PyTorch force Jacobian has shape
    ``(output_atom, output_xyz, input_atom, input_xyz)``. Negating that
    Jacobian gives the energy Hessian because ``F = -dE/dR``. A direct
    ``reshape(3 * N, 3 * N)`` then uses the same atom-major ordering as
    Gaussian, so matrix entry ``[3*a + alpha, 3*b + beta]`` is
    ``d2E / (dR[a, alpha] dR[b, beta])``.
    """
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


def _parse_gaussian_table(lines, start, value_columns, header_dividers):
    index = start + 1
    for _ in range(header_dividers):
        while index < len(lines) and "-----" not in lines[index]:
            index += 1
        index += 1

    atomic_numbers = []
    values = []
    while index < len(lines) and "-----" not in lines[index]:
        fields = lines[index].split()
        if len(fields) >= value_columns + 2 and fields[0].isdigit():
            atomic_numbers.append(int(fields[1]))
            values.append(
                [float(value.replace("D", "E")) for value in fields[-value_columns:]]
            )
        index += 1

    if not values:
        raise ValueError(f"No rows found in Gaussian table starting at line {start + 1}")
    return atomic_numbers, torch.tensor(values, dtype=torch.float32)


def parse_gaussian_log(path):
    """Return aligned coordinates (Angstrom), energies (Hartree), and forces (Hartree/Bohr)."""
    lines = path.read_text(errors="replace").splitlines()
    latest_atomic_numbers = None
    latest_positions = None
    latest_energy = None
    records = []

    for index, line in enumerate(lines):
        if "Input orientation:" in line:
            latest_atomic_numbers, latest_positions = _parse_gaussian_table(
                lines, index, 3, header_dividers=2
            )
            continue

        energy_match = SCF_ENERGY_PATTERN.search(line)
        if energy_match:
            latest_energy = float(energy_match.group(1).replace("D", "E"))
            continue

        if "Forces (Hartrees/Bohr)" not in line:
            continue
        if latest_atomic_numbers is None or latest_positions is None:
            raise ValueError(f"Force table in {path} has no preceding input orientation")
        if latest_energy is None:
            raise ValueError(f"Force table in {path} has no preceding SCF energy")

        force_atomic_numbers, forces = _parse_gaussian_table(
            lines, index, 3, header_dividers=1
        )
        if force_atomic_numbers != latest_atomic_numbers:
            raise ValueError(f"Atomic numbers differ between geometry and forces in {path}")
        records.append(
            {
                "atomic_numbers": torch.tensor(
                    latest_atomic_numbers, dtype=torch.float32
                ).unsqueeze(1),
                "pos": latest_positions.clone(),
                "energy": torch.tensor([[latest_energy]], dtype=torch.float32),
                "forces": forces,
            }
        )

    if not records:
        raise ValueError(f"No aligned energy/force records found in {path}")
    return records


def extract_records(archive_dir, output_dir, limit):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_dir()}
    remaining = max(0, limit - len(existing))
    if remaining == 0:
        return

    archives = sorted(archive_dir.glob("*.tar.zst"), key=lambda path: int(path.stem.split(".")[0]))
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
        # data.pos is stored in Bohr, so the configured radius must also be in Bohr.
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
            hessian_path = molecule_dir / "Hessian.txt"
            log_path = molecule_dir / f"{molecule_dir.name}.log"
            if not all(path.exists() for path in (structure_path, hessian_path, log_path)):
                continue
            try:
                optimized_atomic_numbers, optimized_positions = parse_structure(
                    structure_path
                )
                full_hessian = parse_hessian(
                    hessian_path, optimized_atomic_numbers.shape[0]
                )
                records = parse_gaussian_log(log_path)
                molecule_data = []

                for step, record in enumerate(records):
                    if not torch.equal(
                        record["atomic_numbers"], optimized_atomic_numbers
                    ):
                        raise ValueError(
                            f"Atomic numbers in {log_path} do not match {structure_path}"
                        )
                    is_optimized = torch.allclose(
                        record["pos"],
                        optimized_positions,
                        rtol=0.0,
                        atol=OPTIMIZED_POSITION_TOLERANCE,
                    )
                    hessian = (
                        full_hessian.clone()
                        if is_optimized
                        else torch.full_like(full_hessian, torch.nan)
                    )
                    data = Data(
                        dataset_name="pubchem_gaussian",
                        molecule_id=molecule_dir.name,
                        optimization_step=torch.tensor([step], dtype=torch.int32),
                        natoms=torch.tensor(
                            [record["atomic_numbers"].shape[0]], dtype=torch.int32
                        ),
                        atomic_numbers=record["atomic_numbers"],
                        # Using Bohr here makes -dE/dpos and d2E/dpos2 directly
                        # comparable to Gaussian forces and Hessians below.
                        pos=record["pos"] * ANGSTROM_TO_BOHR,
                        energy=record["energy"],
                        forces=record["forces"],
                        hessian=hessian,
                        hessian_available=torch.tensor([is_optimized]),
                        cell=torch.eye(3, dtype=torch.float32),
                        pbc=torch.zeros(3, dtype=torch.int32),
                    )
                    data = radius_graph(data)
                    data = distance(data)
                    data.edge_shifts = torch.zeros(
                        (data.num_edges, 3), dtype=torch.float32
                    )
                    molecule_data.append(data)

                if not any(data.hessian_available.item() for data in molecule_data):
                    raise ValueError(
                        f"No log geometry in {log_path} matches {structure_path}"
                    )
                self.dataset.extend(molecule_data)
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
            "Preprocess VIBRANT PubChem energy, force, and Hessian records and "
            "train an interatomic potential."
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
        compute_grad_energy=config["NeuralNetwork"]["Architecture"].get(
            "enable_interatomic_potential", False
        ),
    )
    hydragnn.utils.model.save_model(model, optimizer, args.log)
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()