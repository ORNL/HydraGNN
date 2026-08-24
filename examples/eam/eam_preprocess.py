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
"""Example-owned conversion of EAM CFG records to named PyG graphs."""

from pathlib import Path

import torch
from ase.io.cfg import read_cfg
from torch_geometric.data import Data

from hydragnn.preprocess import get_radius_graph_pbc, split_dataset
from hydragnn.utils.datasets.serializeddataset import SerializedWriter
from hydragnn.utils.input_config_parsing.variable_schema import (
    parse_variable_schema,
    prepare_data_from_schema,
)


def parse_eam_cfg(filepath, config):
    """Read one CFG file and expose attributes named by the EAM schemas."""
    path = Path(filepath)
    atoms = read_cfg(path)
    arrays = atoms.arrays
    schema = parse_variable_schema(config["Variables"])
    configured = {spec.name for spec in (*schema.inputs, *schema.outputs)}
    required = {"numbers"}
    if "atomic_energy" in configured:
        required.add("c_peratom")
    if "atomic_forces" in configured:
        required.update(("fx", "fy", "fz"))
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"CFG record {path} is missing arrays: {', '.join(missing)}")

    size = len(atoms)
    values = dict(
        node_features=torch.as_tensor(arrays["numbers"]).float().reshape(size, 1),
        pos=torch.as_tensor(atoms.positions).float(),
        cell=torch.as_tensor(atoms.cell.array).float(),
        pbc=torch.tensor([True, True, True]),
    )
    if "atomic_energy" in configured:
        values["atomic_energy"] = (
            torch.as_tensor(arrays["c_peratom"]).float().reshape(size, 1)
        )
    if "atomic_forces" in configured:
        values["atomic_forces"] = torch.stack(
            [torch.as_tensor(arrays[name]).float() for name in ("fx", "fy", "fz")],
            dim=1,
        )
    sample = Data(**values)

    bulk_path = path.with_suffix(".bulk")
    if bulk_path.is_file():
        values = bulk_path.read_text(encoding="utf-8").split()
        if not values:
            raise ValueError(f"Bulk-modulus file is empty: {bulk_path}")
        sample.bulk_modulus = torch.tensor([[float(values[0])]])

    return sample, schema


def prepare_eam_dataset(raw_directory, config):
    """Read all CFG records, construct periodic edges, and compile the schema."""
    root = Path(raw_directory)
    paths = sorted(root.rglob("*.cfg")) if root.is_dir() else []
    if not paths:
        raise ValueError(f"No CFG records found in: {root}")
    architecture = config["NeuralNetwork"]["Architecture"]
    add_edges = get_radius_graph_pbc(
        radius=architecture["radius"],
        max_neighbours=architecture["max_neighbours"],
        loop=False,
    )
    prepared = []
    for path in paths:
        sample, schema = parse_eam_cfg(path, config)
        prepared.append(prepare_data_from_schema(add_edges(sample), schema))
    return prepared


def split_and_write_eam_dataset(dataset, config, output_directory):
    """Split prepared EAM graphs and write HydraGNN pickle containers."""
    splits = split_dataset(
        dataset,
        config["NeuralNetwork"]["Training"]["perc_train"],
        config["Dataset"].get("compositional_stratified_splitting", False),
    )
    for label, split in zip(("trainset", "valset", "testset"), splits):
        SerializedWriter(split, output_directory, config["Dataset"]["name"], label)
    return splits
