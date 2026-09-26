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
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.transforms import Distance
from torch_geometric.utils import degree
from torch.utils.data import Subset

import hydragnn
from hydragnn.preprocess.load_data import SchemaPreparedDataset, split_dataset
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.adiosdataset import AdiosDataset, AdiosWriter
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleDataset,
    SimplePickleWriter,
)
from hydragnn.utils.print.print_utils import log
from hydragnn.utils.input_config_parsing import (
    parse_variable_schema,
    prepare_data_from_schema,
)

torch.set_default_dtype(torch.float32)

EXAMPLE_DIR = Path(__file__).resolve().parent
RAW_ARCHIVE_DIR = EXAMPLE_DIR / "dataset" / "raw" / "data"
EXTRACTED_DIR = EXAMPLE_DIR / "dataset" / "raw" / "extracted"
ATOMIC_REFERENCE_ARCHIVE = EXAMPLE_DIR / "dataset" / "raw" / "atomization.tar.gz"
PICKLE_DIR = EXAMPLE_DIR / "dataset" / "pubchem_gaussian.pickle"
ADIOS_PATH = EXAMPLE_DIR / "dataset" / "pubchem_gaussian.bp"
CACHE_VERSION = "gaussian-multitask-properties-v6"
SCF_ENERGY_PATTERN = re.compile(
    r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)"
)
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}
# Coordinate matching is performed in the source unit (Angstrom) before conversion.
OPTIMIZED_POSITION_TOLERANCE = 1.0e-5
# CODATA conversion: 1 Angstrom = 1.8897261254578281 Bohr.
ANGSTROM_TO_BOHR = 1.8897261254578281
MULTITASK_PROPERTY_NAMES = {
    "mulliken_charges",
    "dipole_magnitude",
    "quadrupole_eigenvalues",
    "polarizability_eigenvalues",
    "frontier_orbital_energies",
    "rotational_constants",
    "thermochemistry",
}
PROCESSED_FIELDS = {"x", "y", "y_loc", "graph_attr"}


def _last_floats(pattern, text, count=None):
    """Return floats from the final regex match, or raise for a missing label."""
    matches = list(
        re.finditer(pattern, text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    )
    if not matches:
        raise ValueError(f"Gaussian property not found: {pattern}")
    values = [float(value.replace("D", "E")) for value in matches[-1].groups()]
    if count is not None and len(values) != count:
        raise ValueError(f"Expected {count} values for {pattern}, found {len(values)}")
    return values


def _last_labeled_floats(label, text, count):
    matches = re.findall(rf"^\s*{re.escape(label)}\s*(.*?)\s*$", text, re.MULTILINE)
    if not matches:
        raise ValueError(f"Gaussian property not found: {label}")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?"
    values = [
        float(value.replace("D", "E")) for value in re.findall(number, matches[-1])
    ]
    if len(values) != count:
        raise ValueError(f"Expected {count} values for {label}, found {len(values)}")
    return values


def _tensor_eigenvalues(xx, yy, zz, xy, xz, yz):
    tensor = torch.tensor(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=torch.float32
    )
    return torch.linalg.eigvalsh(tensor).unsqueeze(0)


def _parse_mulliken_charges(text, num_atoms):
    sections = list(
        re.finditer(
            r"Mulliken charges(?: and spin densities)?:\s*\n"
            r"\s*1(?:\s+2)?\s*\n(?P<table>.*?)(?=\s*Sum of Mulliken charges)",
            text,
            flags=re.DOTALL,
        )
    )
    if not sections:
        raise ValueError("Final Mulliken charges not found")
    charges = []
    for line in sections[-1].group("table").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit():
            charges.append(float(fields[2].replace("D", "E")))
    if len(charges) != num_atoms:
        raise ValueError(f"Expected {num_atoms} Mulliken charges, found {len(charges)}")
    return charges


def parse_gaussian_properties(path, num_atoms, text=None):
    """Parse final, hardware-independent labels from a Gaussian log.

    Vector/tensor observables use rotation-invariant magnitudes or sorted
    eigenvalues because the current HydraGNN graph heads are scalar heads.
    """
    if text is None:
        text = path.read_text(errors="replace")
    number = r"([-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?)"
    charge, multiplicity = _last_floats(
        rf"Charge\s*=\s*{number}\s+Multiplicity\s*=\s*{number}", text, 2
    )

    charges = _parse_mulliken_charges(text, num_atoms)

    dipole = _last_floats(
        rf"X=\s*{number}\s+Y=\s*{number}\s+Z=\s*{number}\s+Tot=\s*{number}",
        text,
        4,
    )
    quadrupole = _last_floats(
        rf"Traceless Quadrupole moment.*?\n\s*XX=\s*{number}\s+YY=\s*{number}\s+ZZ=\s*{number}\s*\n\s*XY=\s*{number}\s+XZ=\s*{number}\s+YZ=\s*{number}",
        text,
        6,
    )
    polarizability = _last_labeled_floats("Exact polarizability:", text, 6)
    pxx, pyx, pyy, pzx, pzy, pzz = polarizability

    homo = None
    lumo = None
    for line in text.splitlines():
        occupied_match = re.search(r"Alpha\s+occ\. eigenvalues --\s*(.*)", line)
        virtual_match = re.search(r"Alpha\s+virt\. eigenvalues --\s*(.*)", line)
        if occupied_match:
            homo = float(occupied_match.group(1).split()[-1].replace("D", "E"))
            lumo = None
        elif virtual_match and lumo is None:
            lumo = float(virtual_match.group(1).split()[0].replace("D", "E"))
    if homo is None or lumo is None:
        raise ValueError("Final alpha occupied/virtual orbital energies not found")
    rotational_constants = _last_floats(
        rf"Rotational constants \(GHZ\):\s*{number}\s+{number}\s+{number}",
        text,
        3,
    )
    thermochemistry = _last_floats(
        rf"Zero-point correction=\s*{number}.*?"
        rf"Thermal correction to Energy=\s*{number}.*?"
        rf"Thermal correction to Enthalpy=\s*{number}.*?"
        rf"Thermal correction to Gibbs Free Energy=\s*{number}.*?"
        rf"Sum of electronic and zero-point Energies=\s*{number}.*?"
        rf"Sum of electronic and thermal Energies=\s*{number}.*?"
        rf"Sum of electronic and thermal Enthalpies=\s*{number}.*?"
        rf"Sum of electronic and thermal Free Energies=\s*{number}",
        text,
        8,
    )
    return {
        "total_charge": torch.tensor([[charge]], dtype=torch.float32),
        "spin_multiplicity": torch.tensor([[multiplicity]], dtype=torch.float32),
        "mulliken_charges": torch.tensor(charges, dtype=torch.float32).unsqueeze(1),
        "dipole_magnitude": torch.tensor([[dipole[3]]], dtype=torch.float32),
        "quadrupole_eigenvalues": _tensor_eigenvalues(*quadrupole),
        "polarizability_eigenvalues": _tensor_eigenvalues(pxx, pyy, pzz, pyx, pzx, pzy),
        "frontier_orbital_energies": torch.tensor(
            [[homo, lumo, lumo - homo]], dtype=torch.float32
        ),
        "rotational_constants": torch.tensor(
            [sorted(rotational_constants, reverse=True)], dtype=torch.float32
        ),
        "thermochemistry": torch.tensor([thermochemistry], dtype=torch.float32),
    }


def parse_structure(path):
    """Parse atomic numbers and optimized Cartesian coordinates in Angstrom."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    atomic_numbers = torch.tensor([[int(row[0])] for row in rows], dtype=torch.float32)
    positions = torch.tensor(
        [[float(value) for value in row[1:4]] for row in rows], dtype=torch.float32
    )
    return atomic_numbers, positions


def parse_atomic_reference_energies(path):
    """Read final isolated-atom SCF energies from the atomization archive."""
    reference_energies = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            match = re.fullmatch(r"atomization/([A-Z][a-z]?)/elem\.log", member.name)
            if match is None:
                continue
            symbol = match.group(1)
            if symbol not in ATOMIC_NUMBERS:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode(errors="replace")
            energies = SCF_ENERGY_PATTERN.findall(text)
            if not energies:
                raise ValueError(f"No SCF energy found in {member.name}")
            reference_energies[ATOMIC_NUMBERS[symbol]] = float(
                energies[-1].replace("D", "E")
            )

    if not reference_energies:
        raise ValueError(f"No isolated-atom reference energies found in {path}")
    return reference_energies


def compute_formation_energy(total_energy, atomic_numbers, reference_energies):
    """Return E_molecule - sum(n_element * E_isolated_atom), in Hartree."""
    atomic_reference_energy = 0.0
    for atomic_number in atomic_numbers.reshape(-1).to(torch.int64).tolist():
        if atomic_number not in reference_energies:
            raise ValueError(
                f"No isolated-atom reference energy for atomic number {atomic_number}"
            )
        atomic_reference_energy += reference_energies[atomic_number]
    reference = torch.full_like(total_energy, atomic_reference_energy)
    return total_energy - reference, reference


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
        raise ValueError(
            f"No rows found in Gaussian table starting at line {start + 1}"
        )
    return atomic_numbers, torch.tensor(values, dtype=torch.float32)


def parse_gaussian_log(path, text=None):
    """Return aligned coordinates (Angstrom), energies (Hartree), and forces (Hartree/Bohr)."""
    if text is None:
        text = path.read_text(errors="replace")
    lines = text.splitlines()
    latest_atomic_numbers = None
    latest_positions = None
    latest_energy = None
    records = []

    for index, line in enumerate(lines):
        if "Input orientation:" in line:
            latest_atomic_numbers, latest_positions = _parse_gaussian_table(
                lines, index, 3, header_dividers=2
            )
            latest_energy = None
            continue

        energy_match = SCF_ENERGY_PATTERN.search(line)
        if energy_match:
            latest_energy = float(energy_match.group(1).replace("D", "E"))
            continue

        if "Forces (Hartrees/Bohr)" not in line:
            continue
        if latest_atomic_numbers is None or latest_positions is None:
            raise ValueError(
                f"Force table in {path} has no preceding input orientation"
            )
        if latest_energy is None:
            raise ValueError(f"Force table in {path} has no preceding SCF energy")

        force_atomic_numbers, forces = _parse_gaussian_table(
            lines, index, 3, header_dividers=1
        )
        if force_atomic_numbers != latest_atomic_numbers:
            raise ValueError(
                f"Atomic numbers differ between geometry and forces in {path}"
            )
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


def _allocate_archive_limits(available_counts, limit):
    archive_limits = {}
    remaining = limit
    for archive_index in sorted(available_counts):
        archive_limits[archive_index] = min(available_counts[archive_index], remaining)
        remaining -= archive_limits[archive_index]
    return archive_limits, remaining


def extract_records(archive_dir, output_dir, limit, comm, rank, world_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_directories = None
    if rank == 0:
        stale_directories = sorted(output_dir.parent.glob(".extract-rank-*"))
    stale_directories = comm.bcast(stale_directories, root=0)
    for stale_directory in stale_directories[rank::world_size]:
        shutil.rmtree(stale_directory)
    comm.Barrier()

    existing = None
    if rank == 0:
        existing = {
            path.name
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        }
    existing = comm.bcast(existing, root=0)
    remaining = max(0, limit - len(existing))
    if remaining == 0:
        return

    archives = sorted(
        archive_dir.glob("*.tar.zst"), key=lambda path: int(path.stem.split(".")[0])
    )
    if not archives:
        raise FileNotFoundError(f"No .tar.zst archives found in {archive_dir}")

    local_archives = list(enumerate(archives))[rank::world_size]
    local_members = {}
    for archive_index, archive in local_archives:
        listing = subprocess.run(
            ["tar", "--zstd", "-tf", str(archive)],
            check=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.splitlines()
        local_members[archive_index] = [
            name
            for name in listing
            if name.endswith(".tar") and Path(name).stem not in existing
        ]

    available_counts = {}
    for rank_counts in comm.allgather(
        {index: len(members) for index, members in local_members.items()}
    ):
        available_counts.update(rank_counts)
    archive_limits, unavailable = _allocate_archive_limits(available_counts, remaining)
    if unavailable:
        raise RuntimeError(
            f"Only found {limit - unavailable} of {limit} requested records"
        )

    local_target = sum(archive_limits[index] for index, _ in local_archives)
    log(
        f"Extracting {local_target} records from {len(local_archives)} archives",
        rank=None,
    )
    extracted = 0
    for archive_index, archive in local_archives:
        members = local_members[archive_index][: archive_limits[archive_index]]
        if not members:
            continue

        with tempfile.TemporaryDirectory(
            dir=output_dir.parent, prefix=f".extract-rank-{rank}-"
        ) as temporary_dir:
            subprocess.run(
                ["tar", "--zstd", "-xf", str(archive), "-C", temporary_dir, *members],
                check=True,
            )
            for member in members:
                nested_archive = Path(temporary_dir) / member
                target = output_dir / nested_archive.stem
                if target.exists():
                    continue
                unpack_dir = Path(temporary_dir) / f"unpack-{nested_archive.stem}"
                unpack_dir.mkdir()
                with tarfile.open(nested_archive) as nested:
                    nested.extractall(unpack_dir)
                source = unpack_dir / nested_archive.stem
                if not source.is_dir():
                    raise RuntimeError(
                        f"{nested_archive} did not contain {nested_archive.stem}/"
                    )
                source.rename(target)
                extracted += 1
        log(
            f"Extracted {extracted} of {local_target} records on rank {rank}",
            rank=None,
        )


def gather_degree_mpi(dataset, comm):
    local_max_degree = 0
    for data in dataset:
        node_degree = degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        if node_degree.numel():
            local_max_degree = max(local_max_degree, int(node_degree.max()))
    max_degree = comm.allreduce(local_max_degree, op=MPI.MAX)
    local_degree = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        node_degree = degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        local_degree += torch.bincount(node_degree, minlength=local_degree.numel())
    return comm.allreduce(local_degree.numpy(), op=MPI.SUM)


def select_molecule_dirs(root, rank=0, world_size=1, limit=None):
    molecule_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if limit is not None:
        molecule_dirs = molecule_dirs[:limit]
    return molecule_dirs[rank::world_size]


class PubChemGaussianDataset(AbstractBaseDataset):
    def __init__(
        self,
        root,
        config,
        rank=0,
        world_size=1,
        atomic_reference_energies=None,
        limit=None,
    ):
        super().__init__()
        architecture = config["NeuralNetwork"]["Architecture"]
        variable_schema = (
            parse_variable_schema(config["Variables"])
            if "Variables" in config
            else None
        )
        configured_outputs = {
            output["name"] for output in config.get("Variables", {}).get("outputs", [])
        }
        parse_multitask_properties = bool(configured_outputs & MULTITASK_PROPERTY_NAMES)
        if atomic_reference_energies is None:
            atomic_reference_energies = parse_atomic_reference_energies(
                ATOMIC_REFERENCE_ARCHIVE
            )
        distance = Distance(norm=False, cat=False)
        molecule_dirs = select_molecule_dirs(root, rank, world_size, limit)

        for molecule_dir in molecule_dirs:
            structure_path = molecule_dir / "Structure.txt"
            hessian_path = molecule_dir / "Hessian.txt"
            log_path = molecule_dir / f"{molecule_dir.name}.log"
            if not all(
                path.exists() for path in (structure_path, hessian_path, log_path)
            ):
                continue
            try:
                optimized_atomic_numbers, optimized_positions = parse_structure(
                    structure_path
                )
                full_hessian = parse_hessian(
                    hessian_path, optimized_atomic_numbers.shape[0]
                )
                log_text = log_path.read_text(errors="replace")
                records = parse_gaussian_log(log_path, text=log_text)
                properties = (
                    parse_gaussian_properties(
                        log_path, optimized_atomic_numbers.shape[0], text=log_text
                    )
                    if parse_multitask_properties
                    else {}
                )
                matching_records = []
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
                    if is_optimized:
                        matching_records.append((step, record))

                if not matching_records:
                    raise ValueError(
                        f"No log geometry in {log_path} matches {structure_path}"
                    )
                step, record = matching_records[-1]
                formation_energy, atomic_reference_energy = compute_formation_energy(
                    record["energy"],
                    record["atomic_numbers"],
                    atomic_reference_energies,
                )
                data = Data(
                    molecule_id=molecule_dir.name,
                    optimization_step=torch.tensor([step], dtype=torch.int32),
                    natoms=torch.tensor(
                        [record["atomic_numbers"].shape[0]], dtype=torch.int32
                    ),
                    atomic_numbers=record["atomic_numbers"],
                    # Using Bohr here makes -dE/dpos and d2E/dpos2 directly
                    # comparable to Gaussian forces and Hessians below.
                    pos=record["pos"] * ANGSTROM_TO_BOHR,
                    total_energy=record["energy"],
                    atomic_reference_energy=atomic_reference_energy,
                    formation_energy=formation_energy,
                    atomization_energy=-formation_energy,
                    # The potential loss expects data.energy. Subtracting the
                    # composition-only reference leaves force/Hessian derivatives unchanged.
                    energy=formation_energy,
                    forces=record["forces"],
                    hessian=full_hessian,
                    **properties,
                    cell=torch.eye(3, dtype=torch.float32),
                    pbc=torch.zeros(3, dtype=torch.int32),
                )
                # data.pos is stored in Bohr, so the configured radius is in Bohr.
                data.edge_index = radius_graph(
                    data.pos,
                    r=architecture["radius"],
                    loop=False,
                    max_num_neighbors=architecture["max_neighbours"],
                )
                data = distance(data)
                data.edge_shifts = torch.zeros((data.num_edges, 3), dtype=torch.float32)
                if variable_schema is not None:
                    data = prepare_data_from_schema(data, variable_schema)
                self.dataset.append(data)
            except (OSError, ValueError) as error:
                logging.warning("Skipping CID %s: %s", molecule_dir.name, error)

    def len(self):
        return len(self.dataset)

    def get(self, index):
        return self.dataset[index]


def preprocess(
    config,
    num_molecules,
    comm,
    rank,
    world_size,
    dataset_format,
    extracted_dir=EXTRACTED_DIR,
    adios_path=ADIOS_PATH,
    pickle_dir=PICKLE_DIR,
):
    extract_records(
        RAW_ARCHIVE_DIR, extracted_dir, num_molecules, comm, rank, world_size
    )
    comm.Barrier()
    completed = None
    if rank == 0:
        completed = sum(
            path.is_dir() and path.name.isdigit() for path in extracted_dir.iterdir()
        )
    completed = comm.bcast(completed, root=0)
    if completed < num_molecules:
        raise RuntimeError(
            f"Only extracted {completed} of {num_molecules} requested records"
        )

    dataset = PubChemGaussianDataset(
        extracted_dir, config, rank, world_size, limit=num_molecules
    )
    trainset, valset, testset = split_dataset(
        dataset=dataset, perc_train=0.8, stratify_splitting=False
    )
    degree = gather_degree_mpi(trainset, comm)

    if dataset_format == "adios":
        if rank == 0 and adios_path.exists():
            shutil.rmtree(adios_path)
        comm.Barrier()
        writer = AdiosWriter(str(adios_path), comm)
        writer.add("trainset", trainset)
        writer.add("valset", valset)
        writer.add("testset", testset)
        writer.add_global("pna_deg", degree)
        writer.add_global("cache_version", CACHE_VERSION)
        writer.save()
        output_path = adios_path
    else:
        if rank == 0 and pickle_dir.exists():
            shutil.rmtree(pickle_dir)
        comm.Barrier()
        attributes = {"pna_deg": degree, "cache_version": CACHE_VERSION}
        SimplePickleWriter(
            trainset, pickle_dir, "trainset", use_subdir=True, attrs=attributes
        )
        SimplePickleWriter(valset, pickle_dir, "valset", use_subdir=True)
        SimplePickleWriter(testset, pickle_dir, "testset", use_subdir=True)
        output_path = pickle_dir
    log(
        f"Preprocessed {sum(comm.allgather(len(dataset)))} molecules into {output_path}",
        rank=0,
    )


def load_datasets(
    var_config,
    dataset_format,
    comm,
    ddstore,
    ddstore_width,
    shmem,
    dataset_path=None,
):
    if dataset_format == "adios":
        if shmem and ddstore:
            raise ValueError("Cannot use both --shmem and --ddstore")
        options = {
            "preload": False,
            "shmem": shmem,
            "ddstore": ddstore,
            "ddstore_width": ddstore_width,
        }
        path = dataset_path if dataset_path is not None else ADIOS_PATH
        datasets = tuple(
            AdiosDataset(str(path), label, comm, **options)
            for label in ("trainset", "valset", "testset")
        )
        return tuple(
            model_ready_dataset(dataset, var_config) for dataset in datasets
        )

    path = dataset_path if dataset_path is not None else PICKLE_DIR
    datasets = tuple(
        SimplePickleDataset(path, label, var_config=var_config)
        for label in ("trainset", "valset", "testset")
    )
    if datasets[0].attrs.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("Processed data is stale; rerun with --preonly")
    return tuple(model_ready_dataset(dataset, var_config) for dataset in datasets)


def model_ready_dataset(dataset, var_config):
    """Use persisted model tensors, adapting caches from before they were stored."""
    keys = set(dataset.keys) if hasattr(dataset, "keys") else set(dataset[0].keys())
    if PROCESSED_FIELDS.issubset(keys):
        return dataset
    logging.warning(
        "Processed dataset lacks internal model tensors; compiling its named fields "
        "while reading. Regenerate the cache to persist data.x/data.y directly."
    )
    return SchemaPreparedDataset(dataset, parse_variable_schema(var_config))


def deterministic_subset(dataset, num_samples, seed):
    """Select a reproducible subset without copying graph records."""
    if num_samples is None or num_samples >= len(dataset):
        return dataset
    if num_samples <= 0:
        raise ValueError("subset size must be positive")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    return Subset(dataset, indices.tolist())


def validate_fsdp_mode(allow_experimental_fsdp2):
    """Keep FSDP restricted to the explicit FSDP2 Hessian diagnostic."""
    if not bool(int(os.getenv("HYDRAGNN_USE_FSDP", "0"))):
        return
    if not allow_experimental_fsdp2:
        raise ValueError(
            "PubChem Hessian training does not support FSDP; use "
            "--allow-experimental-fsdp2 only for controlled diagnostics"
        )
    if os.getenv("HYDRAGNN_FSDP_VERSION", "1") != "2":
        raise ValueError("The PubChem Hessian diagnostic requires FSDP2")
    if os.getenv("HYDRAGNN_FSDP_STRATEGY", "FULL_SHARD") != "FULL_SHARD":
        raise ValueError("The PubChem Hessian diagnostic requires FULL_SHARD")


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
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-epoch", type=int)
    parser.add_argument("--num-train-samples", type=int)
    parser.add_argument("--num-val-samples", type=int)
    parser.add_argument("--num-test-samples", type=int)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--log", default="pubchem_gaussian")
    parser.add_argument("--ddstore", action="store_true", help="use DDStore")
    parser.add_argument("--ddstore-width", type=int)
    parser.add_argument("--shmem", action="store_true", help="use shared memory")
    parser.add_argument(
        "--allow-experimental-fsdp2",
        action="store_true",
        help="allow the unsupported FSDP2 FULL_SHARD Hessian memory diagnostic",
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--adios", action="store_const", dest="dataset_format", const="adios"
    )
    format_group.add_argument(
        "--pickle", action="store_const", dest="dataset_format", const="pickle"
    )
    parser.set_defaults(dataset_format="adios")
    args = parser.parse_args()

    with open(EXAMPLE_DIR / args.inputfile) as config_file:
        config = json.load(config_file)
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch
    if config["NeuralNetwork"]["Training"]["batch_size"] != 1:
        raise ValueError("PubChem Hessian training currently requires --batch-size 1")
    validate_fsdp_mode(args.allow_experimental_fsdp2)
    if int(os.getenv("HYDRAGNN_GRAPH_PARALLEL_GROUP_SIZE", "1")) > 1:
        raise ValueError("PubChem Hessian training does not support graph parallelism")

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % rank,
    )
    hydragnn.utils.print.setup_log(args.log)

    if args.preonly:
        adios_path = args.output_path if args.output_path else ADIOS_PATH
        pickle_dir = args.output_path if args.output_path else PICKLE_DIR
        preprocess(
            config,
            args.num_molecules,
            comm,
            rank,
            world_size,
            args.dataset_format,
            extracted_dir=args.extracted_dir,
            adios_path=adios_path,
            pickle_dir=pickle_dir,
        )
        return

    ddp_world_size, ddp_rank = hydragnn.utils.distributed.setup_ddp()
    if (ddp_world_size, ddp_rank) != (world_size, rank):
        raise RuntimeError("MPI and torch.distributed rank assignments differ")

    trainset, valset, testset = load_datasets(
        config["Variables"],
        args.dataset_format,
        comm,
        args.ddstore,
        args.ddstore_width,
        args.shmem,
        args.dataset_path,
    )
    pna_deg = trainset.pna_deg
    trainset = deterministic_subset(trainset, args.num_train_samples, args.subset_seed)
    valset = deterministic_subset(valset, args.num_val_samples, args.subset_seed + 1)
    testset = deterministic_subset(testset, args.num_test_samples, args.subset_seed + 2)
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset,
        valset,
        testset,
        config["NeuralNetwork"]["Training"]["batch_size"],
    )
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    config["pna_deg"] = pna_deg.tolist()
    hydragnn.utils.input_config_parsing.save_config(config, args.log)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(config["NeuralNetwork"], verbosity)
    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    use_fsdp2 = bool(int(os.getenv("HYDRAGNN_USE_FSDP", "0"))) and os.getenv(
        "HYDRAGNN_FSDP_VERSION", "1"
    ) == "2"
    optimizer = None if use_fsdp2 else torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )
    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity, config=config
    )
    if use_fsdp2:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1.0e-5
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
