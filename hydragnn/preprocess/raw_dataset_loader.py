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

import os
import numpy as np
import pickle
import pathlib

import torch
from torch_geometric.data import Data
from torch import tensor

# WARNING: DO NOT use collective communication calls here because only rank 0 uses this routines


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))


class RawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, config):
        """
        config:
          shows the dataset path the target variables information, e.g, location and dimension, in data file
        ###########
        dataset_list:
          list of datasets read from self.path_dictionary
        serial_data_name_list:
          list of pkl file names
        node_feature_dim:
          list of dimensions of node features
        node_feature_col:
          list of column location/index (start location if dim>1) of node features
        graph_feature_dim:
          list of dimensions of graph features
        graph_feature_col: list,
          list of column location/index (start location if dim>1) of graph features
        """
        self.dataset_list = []
        self.serial_data_name_list = []
        self.node_feature_name = config["node_features"]["name"]
        self.node_feature_dim = config["node_features"]["dim"]
        self.node_feature_col = config["node_features"]["column_index"]
        self.graph_feature_name = config["graph_features"]["name"]
        self.graph_feature_dim = config["graph_features"]["dim"]
        self.graph_feature_col = config["graph_features"]["column_index"]
        self.raw_dataset_name = config["name"]
        self.data_format = config["format"]
        self.path_dictionary = config["path"]

    def load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        if not os.path.exists(serialized_dir):
            os.mkdir(serialized_dir)

        for dataset_type, raw_data_path in self.path_dictionary.items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(os.getcwd(), raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)

            dataset = []
            assert (
                len(os.listdir(raw_data_path)) > 0
            ), "No data files provided in {}!".format(raw_data_path)

            for filename in os.listdir(raw_data_path):
                if filename == ".DS_Store":
                    continue
                f = open(os.path.join(raw_data_path, filename), "r", encoding="utf-8")
                all_lines = f.readlines()
                data_object = self.__transform_input_to_data_object_base(
                    lines=all_lines
                )
                dataset.append(data_object)
                f.close()

            if self.data_format == "LSMS":
                for idx, data_object in enumerate(dataset):
                    dataset[idx] = self.__charge_density_update_for_LSMS(data_object)

            # scaled features by number of nodes
            dataset = self.__scale_features_by_num_nodes(dataset)

            if dataset_type == "total":
                serial_data_name = self.raw_dataset_name + ".pkl"
            else:
                # append for train; test; validation
                serial_data_name = self.raw_dataset_name + "_" + dataset_type + ".pkl"

            self.dataset_list.append(dataset)
            self.serial_data_name_list.append(serial_data_name)

        self.__normalize_dataset()

        for serial_data_name, dataset_normalized in zip(
            self.serial_data_name_list, self.dataset_list
        ):
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(dataset_normalized, f)

    def __transform_input_to_data_object_base(self, lines: [str]):
        """Transforms lines of strings read from the raw data file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        data_object = Data()

        graph_feat = lines[0].split(None, 2)
        g_feature = []
        # collect graph features
        for item in range(len(self.graph_feature_dim)):
            for icomp in range(self.graph_feature_dim[item]):
                it_comp = self.graph_feature_col[item] + icomp
                g_feature.append(float(graph_feat[it_comp].strip()))
        data_object.y = tensor(g_feature)

        node_feature_matrix = []
        node_position_matrix = []
        for line in lines[1:]:
            node_feat = line.split(None, 11)

            x_pos = float(node_feat[2].strip())
            y_pos = float(node_feat[3].strip())
            z_pos = float(node_feat[4].strip())
            node_position_matrix.append([x_pos, y_pos, z_pos])

            node_feature = []
            for item in range(len(self.node_feature_dim)):
                for icomp in range(self.node_feature_dim[item]):
                    it_comp = self.node_feature_col[item] + icomp
                    node_feature.append(float(node_feat[it_comp].strip()))
            node_feature_matrix.append(node_feature)

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)
        data_object.num_nodes_list = data_object.pos.shape[0]
        return data_object

    def __charge_density_update_for_LSMS(self, data_object: Data):
        """Calculate charge density for LSMS format
        Parameters
        ----------
        data_object: Data
            Data object representing structure of a graph sample.

        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        num_of_protons = data_object.x[:, 0]
        charge_density = data_object.x[:, 1]
        charge_density -= num_of_protons
        data_object.x[:, 1] = charge_density
        return data_object

    def __scale_features_by_num_nodes(self, dataset):
        """Calculate [**]_scaled_num_nodes"""
        scaled_graph_feature_index = [
            i
            for i in range(len(self.graph_feature_name))
            if "_scaled_num_nodes" in self.graph_feature_name[i]
        ]
        scaled_node_feature_index = [
            i
            for i in range(len(self.node_feature_name))
            if "_scaled_num_nodes" in self.node_feature_name[i]
        ]

        for idx, data_object in enumerate(dataset):
            dataset[idx].y[scaled_graph_feature_index] = (
                dataset[idx].y[scaled_graph_feature_index] / data_object.num_nodes
            )
            dataset[idx].x[:, scaled_node_feature_index] = (
                dataset[idx].x[:, scaled_node_feature_index] / data_object.num_nodes
            )

        return dataset

    def __normalize_dataset(self):

        """Performs the normalization on Data objects and returns the normalized dataset."""
        num_node_features = self.dataset_list[0][0].x.shape[1]
        num_graph_features = len(self.dataset_list[0][0].y)

        self.minmax_graph_feature = np.full((2, num_graph_features), np.inf)
        # [0,...]:minimum values; [1,...]: maximum values
        self.minmax_node_feature = np.full((2, num_node_features), np.inf)
        self.minmax_graph_feature[1, :] *= -1
        self.minmax_node_feature[1, :] *= -1
        for dataset in self.dataset_list:
            for data in dataset:
                # find maximum and minimum values for graph level features
                for ifeat in range(num_graph_features):
                    self.minmax_graph_feature[0, ifeat] = min(
                        data.y[ifeat], self.minmax_graph_feature[0, ifeat]
                    )
                    self.minmax_graph_feature[1, ifeat] = max(
                        data.y[ifeat], self.minmax_graph_feature[1, ifeat]
                    )
                # find maximum and minimum values for node level features
                for ifeat in range(num_node_features):
                    self.minmax_node_feature[0, ifeat] = np.minimum(
                        np.amin(data.x[:, ifeat].numpy()),
                        self.minmax_node_feature[0, ifeat],
                    )
                    self.minmax_node_feature[1, ifeat] = np.maximum(
                        np.amax(data.x[:, ifeat].numpy()),
                        self.minmax_node_feature[1, ifeat],
                    )
        for dataset in self.dataset_list:
            for data in dataset:
                for ifeat in range(num_graph_features):
                    data.y[ifeat] = tensor_divide(
                        (data.y[ifeat] - self.minmax_graph_feature[0, ifeat]),
                        (
                            self.minmax_graph_feature[1, ifeat]
                            - self.minmax_graph_feature[0, ifeat]
                        ),
                    )
                for ifeat in range(num_node_features):
                    data.x[:, ifeat] = tensor_divide(
                        (data.x[:, ifeat] - self.minmax_node_feature[0, ifeat]),
                        (
                            self.minmax_node_feature[1, ifeat]
                            - self.minmax_node_feature[0, ifeat]
                        ),
                    )
