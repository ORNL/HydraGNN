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
"""Load and prepare explicitly supplied graph datasets."""

import pickle

from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    Distance,
    NormalizeRotation,
    PointPairFeatures,
    Spherical,
)

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    get_radius_graph,
    get_radius_graph_pbc,
    update_atom_features,
    update_predicted_values,
)
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print.print_utils import iterate_tqdm, print_distributed


def load_pickled_graphs(dataset_path):
    """Load the historical three-object HydraGNN pickle container."""
    with open(dataset_path, "rb") as stream:
        pickle.load(stream)
        pickle.load(stream)
        return pickle.load(stream)


def _build_edges(data, radius, max_neighbours, periodic):
    if periodic:
        data.pbc = [True, True, True]
        if not hasattr(data, "cell") or data.cell is None:
            raise ValueError("Periodic graph samples require data.cell")
        return get_radius_graph_pbc(radius, loop=False, max_neighbours=max_neighbours)(
            data
        )
    data = get_radius_graph(radius, loop=False, max_neighbours=max_neighbours)(data)
    return Distance(norm=False, cat=True)(data)


def _stratified_sample(dataset, percentage, verbosity):
    categories = []
    print_distributed(verbosity, "Computing the categories for the whole datasets.")
    for data in iterate_tqdm(dataset, verbosity):
        frequencies = sorted(
            value for value in torch.bincount(data.x[:, 0].int()).tolist() if value
        )
        categories.append(
            sum(value * (100**index) for index, value in enumerate(frequencies))
        )
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=percentage, random_state=0)
    indices, _ = next(splitter.split(dataset, categories))
    return [dataset[index] for index in indices]


def prepare_graph_dataset(dataset, config, dist=False):
    """Prepare an explicit graph collection according to per-sample geometry."""
    architecture = config["NeuralNetwork"]["Architecture"]
    dataset_config = config["Dataset"]
    variables = config["NeuralNetwork"]["Variables_of_interest"]
    verbosity = config["Verbosity"]["level"]

    node_features = dataset_config["node_features"]
    graph_features = dataset_config["graph_features"]
    if not (
        len(node_features["name"])
        == len(node_features["dim"])
        == len(node_features["column_index"])
    ):
        raise ValueError("Node feature names, dimensions, and columns must align")
    if not (
        len(graph_features["name"])
        == len(graph_features["dim"])
        == len(graph_features["column_index"])
    ):
        raise ValueError("Graph feature names, dimensions, and columns must align")

    if dataset_config["rotational_invariance"]:
        transform = NormalizeRotation(max_points=-1, sort=False)
        dataset = [transform(data) for data in dataset]

    dataset = [
        _build_edges(
            data,
            architecture["radius"],
            architecture["max_neighbours"],
            architecture["periodic_boundary_conditions"],
        )
        for data in dataset
    ]

    max_edge_length = max(torch.max(data.edge_attr) for data in dataset)
    if dist:
        device = max_edge_length.device
        max_edge_length = max_edge_length.to(get_device())
        torch.distributed.all_reduce(max_edge_length, op=torch.distributed.ReduceOp.MAX)
        max_edge_length = max_edge_length.to(device)
    for data in dataset:
        data.edge_attr = data.edge_attr / max_edge_length

    descriptors = dataset_config.get("Descriptors", {})
    if descriptors.get("SphericalCoordinates"):
        dataset = [Spherical(data) for data in dataset]
    if descriptors.get("PointPairFeatures"):
        dataset = [PointPairFeatures(data) for data in dataset]

    laplacian = AddLaplacianEigenvectorPE(
        k=architecture["pe_dim"], attr_name="pe", is_undirected=True
    )
    dataset = [laplacian(data) for data in dataset]
    for data in dataset:
        data.rel_pe = torch.abs(
            data.pe[data.edge_index[0]] - data.pe[data.edge_index[1]]
        )
        update_predicted_values(
            variables["type"],
            variables["output_index"],
            dataset_config["graph_features"]["dim"],
            dataset_config["node_features"]["dim"],
            data,
        )
        update_atom_features(variables["input_node_features"], data)

    percentage = variables.get("subsample_percentage")
    if percentage is not None:
        dataset = _stratified_sample(dataset, percentage, verbosity)
    return dataset


def load_and_prepare_graph_dataset(dataset_path, config, dist=False):
    """Load an explicit pickle path and prepare its graph samples."""
    return prepare_graph_dataset(load_pickled_graphs(dataset_path), config, dist=dist)
