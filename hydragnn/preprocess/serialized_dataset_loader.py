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

import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance

from .dataset_descriptors import AtomFeatures
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm


class SerializedDataLoader:

    """
    Constructor
    """

    def __init__(self, verbosity: int):
        self.verbosity = verbosity

    """A class used for loading existing structures from files that are lists of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_serialized_data method.

    Methods
    -------
    load_serialized_data(dataset_path: str, config: dict)
        Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.
    """

    def load_serialized_data(self, dataset_path: str, config):
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
        dataset = []
        with open(dataset_path, "rb") as f:
            _ = pickle.load(f)
            _ = pickle.load(f)
            dataset = pickle.load(f)

        compute_edges = RadiusGraph(
            r=config["Architecture"]["radius"],
            loop=False,
            max_num_neighbors=config["Architecture"]["max_neighbours"],
        )
        compute_edge_lengths = Distance(norm=False, cat=True)

        dataset[:] = [compute_edges(data) for data in dataset]
        dataset[:] = [compute_edge_lengths(data) for data in dataset]

        max_edge_length = torch.Tensor([float("-inf")])

        for data in dataset:
            max_edge_length = torch.max(max_edge_length, torch.max(data.edge_attr))

        # Normalization of the edges
        for data in dataset:
            data.edge_attr = data.edge_attr / max_edge_length

        # Move data to the device, if used. # FIXME: this does not respect the choice set by use_gpu
        device = get_device(verbosity_level=self.verbosity)
        for data in dataset:
            data.to(device)
            self.__update_predicted_values(
                config["Variables_of_interest"]["type"],
                config["Variables_of_interest"]["output_index"],
                data,
            )
            self.__update_atom_features(
                config["Variables_of_interest"]["input_node_features"], data
            )

        if "subsample_percentage" in config["Variables_of_interest"].keys():
            return self.__stratified_sampling(
                dataset=dataset,
                subsample_percentage=config["Variables_of_interest"][
                    "subsample_percentage"
                ],
            )

        return dataset

    def __update_atom_features(self, atom_features: [AtomFeatures], data: Data):
        """Updates atom features of a structure. An atom is represented with x,y,z coordinates and associated features.

        Parameters
        ----------
        atom_features: [AtomFeatures]
            List of features to update. Each feature is instance of Enum AtomFeatures.
        data: Data
            A Data object representing a structure that has atoms.
        """
        feature_indices = [i for i in atom_features]
        data.x = data.x[:, feature_indices]

    def __update_predicted_values(self, type: list, index: list, data: Data):
        """Updates values of the structure we want to predict. Predicted value is represented by integer value.
        Parameters
        ----------
        type: "graph" level or "node" level
        index: index/location in data.y for graph level and in data.x for node level
        data: Data
            A Data object representing a structure that has atoms.
        """
        output_feature = []
        data.y_loc = torch.zeros(1, len(type) + 1, dtype=torch.int64)
        for item in range(len(type)):
            if type[item] == "graph":
                feat_ = torch.reshape(data.y[index[item]], (1, 1))
            elif type[item] == "node":
                feat_ = torch.reshape(data.x[:, index[item]], (-1, 1))
            else:
                raise ValueError("Unknown output type", type[item])
            output_feature.append(feat_)
            data.y_loc[0, item + 1] = (
                data.y_loc[0, item] + feat_.shape[0] * feat_.shape[1]
            )
        data.y = torch.cat(output_feature, 0)

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
        unique_values = torch.unique(dataset[0].x[:, 0]).tolist()
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
