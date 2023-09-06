##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import pickle
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import (
    Distance,
    NormalizeRotation,
    Spherical,
    PointPairFeatures,
)

from hydragnn.preprocess import update_predicted_values, update_atom_features
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.preprocess.utils import (
    get_radius_graph,
    get_radius_graph_pbc,
)


class SerializedDataLoader:
    """A class used for loading existing structures from files that are lists of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_serialized_data method.
    """

    """
    Constructor
    """

    def __init__(self, config, dist=False):
        self.verbosity = config["Verbosity"]["level"]
        self.node_feature_name = config["Dataset"]["node_features"]["name"]
        self.node_feature_dim = config["Dataset"]["node_features"]["dim"]
        self.node_feature_col = config["Dataset"]["node_features"]["column_index"]
        self.graph_feature_name = config["Dataset"]["graph_features"]["name"]
        self.graph_feature_dim = config["Dataset"]["graph_features"]["dim"]
        self.graph_feature_col = config["Dataset"]["graph_features"]["column_index"]
        self.rotational_invariance = config["Dataset"]["rotational_invariance"]
        self.periodic_boundary_conditions = config["NeuralNetwork"]["Architecture"][
            "periodic_boundary_conditions"
        ]
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        self.variables = config["NeuralNetwork"]["Variables_of_interest"]
        self.variables_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
        self.output_index = config["NeuralNetwork"]["Variables_of_interest"][
            "output_index"
        ]
        self.input_node_features = config["NeuralNetwork"]["Variables_of_interest"][
            "input_node_features"
        ]

        self.spherical_coordinates = False
        self.point_pair_features = False

        if "Descriptors" in config["Dataset"]:
            if "SphericalCoordinates" in config["Dataset"]["Descriptors"]:
                self.spherical_coordinates = config["Dataset"]["Descriptors"][
                    "SphericalCoordinates"
                ]
            if "PointPairFeatures" in config["Dataset"]["Descriptors"]:
                self.point_pair_features = config["Dataset"]["Descriptors"][
                    "PointPairFeatures"
                ]

        self.subsample_percentage = None

        # In situations where someone already provides the .pkl filed with data
        # the asserts from raw_dataset_loader are not performed
        # Therefore, we need to re-check consistency
        assert len(self.node_feature_name) == len(self.node_feature_dim)
        assert len(self.node_feature_name) == len(self.node_feature_col)
        assert len(self.graph_feature_name) == len(self.graph_feature_dim)
        assert len(self.graph_feature_name) == len(self.graph_feature_col)

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

    """
    Methods
    -------
    load_serialized_data(dataset_path: str, config: dict)
        Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.
    """

    def load_serialized_data(self, dataset_path: str):
        """Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.

        Parameters
        ----------
        dataset_path: str
            Directory path where files containing serialized structures are stored.
        config: dict
            Dictionary containing information needed to load the data and transform it, respectively: atom_features, radius, max_num_node_neighbours and predicted_value_option.
        Returns
        ----------
        [Data]
            List of Data objects representing atom structures.
        """
        with open(dataset_path, "rb") as f:
            _ = pickle.load(f)
            _ = pickle.load(f)
            dataset = pickle.load(f)

        rotational_invariance = NormalizeRotation(max_points=-1, sort=False)
        if self.rotational_invariance:
            dataset[:] = [rotational_invariance(data) for data in dataset]

        if self.periodic_boundary_conditions:
            # edge lengths already added manually if using PBC, so no need to call Distance.
            compute_edges = get_radius_graph_pbc(
                radius=self.radius,
                loop=False,
                max_neighbours=self.max_neighbours,
            )
        else:
            compute_edges = get_radius_graph(
                radius=self.radius,
                loop=False,
                max_neighbours=self.max_neighbours,
            )

        dataset[:] = [compute_edges(data) for data in dataset]

        # edge lengths already added manually if using PBC.
        if not self.periodic_boundary_conditions:
            compute_edge_lengths = Distance(norm=False, cat=True)
            dataset[:] = [compute_edge_lengths(data) for data in dataset]

        max_edge_length = torch.Tensor([float("-inf")])

        for data in dataset:
            max_edge_length = torch.max(max_edge_length, torch.max(data.edge_attr))

        if self.dist:
            ## Gather max in parallel
            device = max_edge_length.device
            max_edge_length = max_edge_length.to(get_device())
            torch.distributed.all_reduce(
                max_edge_length, op=torch.distributed.ReduceOp.MAX
            )
            max_edge_length = max_edge_length.to(device)

        # Normalization of the edges
        for data in dataset:
            data.edge_attr = data.edge_attr / max_edge_length

        # Descriptors about topology of the local environment
        if self.spherical_coordinates:
            self.dataset[:] = [Spherical(data) for data in self.dataset]

        if self.point_pair_features:
            self.dataset[:] = [PointPairFeatures(data) for data in self.dataset]

        # Move data to the device, if used. # FIXME: this does not respect the choice set by use_gpu
        device = get_device(verbosity_level=self.verbosity)
        for data in dataset:
            ## (2022/04) jyc: no need for parallel loading
            # data.to(device)
            update_predicted_values(
                self.variables_type,
                self.output_index,
                self.graph_feature_dim,
                self.node_feature_dim,
                data,
            )

            update_atom_features(self.input_node_features, data)

        if "subsample_percentage" in self.variables.keys():
            self.subsample_percentage = self.variables["subsample_percentage"]
            return self.__stratified_sampling(
                dataset=dataset, subsample_percentage=self.subsample_percentage
            )

        return dataset

    def __stratified_sampling(self, dataset: [Data], subsample_percentage: float):
        """Given the dataset and the percentage of data you want to extract from it, method will
        apply stratified sampling where X is the dataset and Y is are the category values for each datapoint.
        In the case of the structures dataset where each structure contains 2 types of atoms, the category will
        be constructed in a way: number of atoms of type 1 + number of protons of type 2 * 100.

        Parameters
        ----------
        dataset: [Data]
            A list of Data objects representing a structure that has atoms.
        subsample_percentage: float
            Percentage of the dataset.

        Returns
        ----------
        [Data]
            Subsample of the original dataset constructed using stratified sampling.
        """
        dataset_categories = []
        print_distributed(
            self.verbosity, "Computing the categories for the whole dataset."
        )
        for data in iterate_tqdm(dataset, self.verbosity):
            frequencies = torch.bincount(data.x[:, 0].int())
            frequencies = sorted(frequencies[frequencies > 0].tolist())
            category = 0
            for index, frequency in enumerate(frequencies):
                category += frequency * (100 ** index)
            dataset_categories.append(category)

        subsample_indices = []
        subsample = []

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=subsample_percentage, random_state=0
        )

        for subsample_index, rest_of_data_index in sss.split(
            dataset, dataset_categories
        ):
            subsample_indices = subsample_index.tolist()

        for index in subsample_indices:
            subsample.append(dataset[index])

        return subsample
