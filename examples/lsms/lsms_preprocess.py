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
"""Example-owned conversion of raw LSMS records into prepared PyG graphs."""

from pathlib import Path

import torch
from torch_geometric.data import Data

from hydragnn.preprocess import get_radius_graph, split_dataset
from hydragnn.utils.datasets.serializeddataset import SerializedWriter
from hydragnn.utils.input_config_parsing.variable_schema import (
    parse_variable_schema,
    prepare_data_from_schema,
)


def parse_lsms_file(filepath):
    """Parse one LSMS text record into semantic, named PyG attributes.

    The example's LSMS layout stores total free energy on the first line. Atom
    rows store atomic number in column 0, Cartesian position in columns 2--4,
    electronic charge in column 5, and magnetic moment in column 6. Charge
    density is the electronic charge minus the number of protons.
    """
    path = Path(filepath)
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 2 or not lines[0]:
        raise ValueError(f"LSMS record has no atom rows: {path}")

    atom_rows = lines[1:]
    if any(len(row) < 7 for row in atom_rows):
        raise ValueError(f"LSMS atom rows require at least seven columns: {path}")

    atomic_numbers = torch.tensor(
        [[float(row[0])] for row in atom_rows], dtype=torch.float32
    )
    electronic_charge = torch.tensor(
        [[float(row[5])] for row in atom_rows], dtype=torch.float32
    )
    magnetic_moment = torch.tensor(
        [[float(row[6])] for row in atom_rows], dtype=torch.float32
    )
    pos = torch.tensor(
        [[float(row[index]) for index in (2, 3, 4)] for row in atom_rows],
        dtype=torch.float32,
    )
    total_energy = float(lines[0][0])

    return Data(
        num_of_protons=atomic_numbers,
        free_energy_per_atom=torch.tensor(
            [[total_energy / len(atom_rows)]], dtype=torch.float32
        ),
        charge_density=electronic_charge - atomic_numbers,
        magnetic_moment=magnetic_moment,
        pos=pos,
    )


def load_lsms_directory(raw_directory):
    """Recursively parse every non-hidden file in an LSMS directory."""
    root = Path(raw_directory)
    if not root.is_dir():
        raise ValueError(f"LSMS raw-data directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    paths = [path for path in paths if not path.name.startswith(".")]
    if not paths:
        raise ValueError(f"No LSMS records found in: {root}")
    return [parse_lsms_file(path) for path in paths]


def prepare_lsms_dataset(raw_directory, config):
    """Build edges and compile named variables for every raw LSMS record."""
    architecture = config["NeuralNetwork"]["Architecture"]
    compute_edges = get_radius_graph(
        radius=architecture["radius"],
        max_neighbours=architecture["max_neighbours"],
        loop=False,
    )
    schema = parse_variable_schema(config["Variables"])
    prepared = []
    for sample in load_lsms_directory(raw_directory):
        sample = compute_edges(sample)
        prepared.append(prepare_data_from_schema(sample, schema))
    return prepared


def split_and_write_lsms_dataset(dataset, config, output_directory):
    """Split prepared graphs and write HydraGNN pickle containers."""
    trainset, valset, testset = split_dataset(
        dataset,
        config["NeuralNetwork"]["Training"]["perc_train"],
        config["Dataset"].get("compositional_stratified_splitting", False),
    )
    output = Path(output_directory)
    dataset_name = config["Dataset"]["name"]
    for label, split in (
        ("trainset", trainset),
        ("valset", valset),
        ("testset", testset),
    ):
        SerializedWriter(split, str(output), dataset_name, label)
    return trainset, valset, testset
