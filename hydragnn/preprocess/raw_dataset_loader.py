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

import torch

from hydragnn.utils.distributed import get_device
from hydragnn.utils.print.print_utils import log
from hydragnn.utils.distributed import nsplit, comm_reduce
from hydragnn.utils.model.model import tensor_divide

import random


class AbstractRawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, config, dist=False):
        """
        config:
          shows the datasets path the target variables information, e.g, location and dimension, in data file
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

        dist: True if RawDataLoder is distributed (i.e., each RawDataLoader will read different subset of data)
        """

        self.dataset_list = []
        self.serial_data_name_list = []
        self.node_feature_name = (
            config["node_features"]["name"]
            if config["node_features"]["name"] is not None
            else None
        )
        self.node_feature_dim = config["node_features"]["dim"]
        self.node_feature_col = config["node_features"]["column_index"]
        self.graph_feature_name = (
            config["graph_features"]["name"]
            if config["graph_features"]["name"] is not None
            else None
        )
        self.graph_feature_dim = config["graph_features"]["dim"]
        self.graph_feature_col = config["graph_features"]["column_index"]
        self.raw_dataset_name = config["name"]
        self.data_format = config["format"]
        self.path_dictionary = config["path"]

        assert len(self.node_feature_name) == len(self.node_feature_dim)
        assert len(self.node_feature_name) == len(self.node_feature_col)
        assert len(self.graph_feature_name) == len(self.graph_feature_dim)
        assert len(self.graph_feature_name) == len(self.graph_feature_col)

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

    def load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        if not os.path.exists(serialized_dir):
            os.makedirs(serialized_dir, exist_ok=True)

        for dataset_type, raw_data_path in self.path_dictionary.items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(os.getcwd(), raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)

            dataset = []
            assert (
                len(os.listdir(raw_data_path)) > 0
            ), "No data files provided in {}!".format(raw_data_path)

            filelist = sorted(os.listdir(raw_data_path))
            if self.dist:
                ## Random shuffle filelist to avoid the same test/validation set
                random.seed(43)
                random.shuffle(filelist)

                x = torch.tensor(len(filelist), requires_grad=False).to(get_device())
                y = x.clone().detach().requires_grad_(False)
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
                assert x == y
                filelist = list(nsplit(filelist, self.world_size))[self.rank]
                log("local filelist", len(filelist))

            for name in filelist:
                if name == ".DS_Store":
                    continue
                # if the directory contains file, iterate over them
                if os.path.isfile(os.path.join(raw_data_path, name)):
                    data_object = self.transform_input_to_data_object_base(
                        filepath=os.path.join(raw_data_path, name)
                    )
                    if not isinstance(data_object, type(None)):
                        dataset.append(data_object)
                # if the directory contains subdirectories, explore their content
                elif os.path.isdir(os.path.join(raw_data_path, name)):
                    dir_name = os.path.join(raw_data_path, name)
                    for subname in os.listdir(dir_name):
                        if os.path.isfile(os.path.join(dir_name, subname)):
                            data_object = self.transform_input_to_data_object_base(
                                filepath=os.path.join(dir_name, subname)
                            )
                            if not isinstance(data_object, type(None)):
                                dataset.append(data_object)

            # scaled features by number of nodes
            dataset = self.scale_features_by_num_nodes(dataset)

            if dataset_type == "total":
                serial_data_name = self.raw_dataset_name + ".pkl"
            else:
                # append for train; test; validation
                serial_data_name = self.raw_dataset_name + "_" + dataset_type + ".pkl"

            self.dataset_list.append(dataset)
            self.serial_data_name_list.append(serial_data_name)

        self.normalize_dataset()

        for serial_data_name, dataset_normalized in zip(
            self.serial_data_name_list, self.dataset_list
        ):
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(dataset_normalized, f)

    def transform_input_to_data_object_base(self, filepath):
        pass

    def scale_features_by_num_nodes(self, dataset):
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
            if dataset[idx].y is not None:
                dataset[idx].y[scaled_graph_feature_index] = (
                    dataset[idx].y[scaled_graph_feature_index] / data_object.num_nodes
                )
            if dataset[idx].x is not None:
                dataset[idx].x[:, scaled_node_feature_index] = (
                    dataset[idx].x[:, scaled_node_feature_index] / data_object.num_nodes
                )

        return dataset

    def normalize_dataset(self):

        """Performs the normalization on Data objects and returns the normalized datasets."""
        num_node_features = len(self.node_feature_dim)
        num_graph_features = len(self.graph_feature_dim)

        self.minmax_graph_feature = np.full((2, num_graph_features), np.inf)
        # [0,...]:minimum values; [1,...]: maximum values
        self.minmax_node_feature = np.full((2, num_node_features), np.inf)
        self.minmax_graph_feature[1, :] *= -1
        self.minmax_node_feature[1, :] *= -1
        for dataset in self.dataset_list:
            for data in dataset:
                # find maximum and minimum values for graph level features
                g_index_start = 0
                for ifeat in range(num_graph_features):
                    g_index_end = g_index_start + self.graph_feature_dim[ifeat]
                    self.minmax_graph_feature[0, ifeat] = min(
                        torch.min(data.y[g_index_start:g_index_end]),
                        self.minmax_graph_feature[0, ifeat],
                    )
                    self.minmax_graph_feature[1, ifeat] = max(
                        torch.max(data.y[g_index_start:g_index_end]),
                        self.minmax_graph_feature[1, ifeat],
                    )
                    g_index_start = g_index_end

                # find maximum and minimum values for node level features
                n_index_start = 0
                for ifeat in range(num_node_features):
                    n_index_end = n_index_start + self.node_feature_dim[ifeat]
                    self.minmax_node_feature[0, ifeat] = min(
                        torch.min(data.x[:, n_index_start:n_index_end]),
                        self.minmax_node_feature[0, ifeat],
                    )
                    self.minmax_node_feature[1, ifeat] = max(
                        torch.max(data.x[:, n_index_start:n_index_end]),
                        self.minmax_node_feature[1, ifeat],
                    )
                    n_index_start = n_index_end

        ## Gather minmax in parallel
        if self.dist:
            self.minmax_graph_feature[0, :] = comm_reduce(
                self.minmax_graph_feature[0, :], torch.distributed.ReduceOp.MIN
            )
            self.minmax_graph_feature[1, :] = comm_reduce(
                self.minmax_graph_feature[1, :], torch.distributed.ReduceOp.MAX
            )
            self.minmax_node_feature[0, :] = comm_reduce(
                self.minmax_node_feature[0, :], torch.distributed.ReduceOp.MIN
            )
            self.minmax_node_feature[1, :] = comm_reduce(
                self.minmax_node_feature[1, :], torch.distributed.ReduceOp.MAX
            )

        for dataset in self.dataset_list:
            for data in dataset:
                g_index_start = 0
                for ifeat in range(num_graph_features):
                    g_index_end = g_index_start + self.graph_feature_dim[ifeat]
                    data.y[g_index_start:g_index_end] = tensor_divide(
                        (
                            data.y[g_index_start:g_index_end]
                            - self.minmax_graph_feature[0, ifeat]
                        ),
                        (
                            self.minmax_graph_feature[1, ifeat]
                            - self.minmax_graph_feature[0, ifeat]
                        ),
                    )
                    g_index_start = g_index_end
                n_index_start = 0
                for ifeat in range(num_node_features):
                    n_index_end = n_index_start + self.node_feature_dim[ifeat]
                    data.x[:, n_index_start:n_index_end] = tensor_divide(
                        (
                            data.x[:, n_index_start:n_index_end]
                            - self.minmax_node_feature[0, ifeat]
                        ),
                        (
                            self.minmax_node_feature[1, ifeat]
                            - self.minmax_node_feature[0, ifeat]
                        ),
                    )
                    n_index_start = n_index_end
