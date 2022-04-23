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
from torch_geometric.data import Data
from torch import tensor

from ase.io.cfg import read_cfg

try:
    from mpi4py import MPI
except ImportError:
    pass

from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
import random

# WARNING: DO NOT use collective communication calls here because only rank 0 uses this routines


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def comm_reduce(x, comm, op):
    y = np.zeros_like(x)
    assert x.dtype == np.double
    comm.Allreduce([x, MPI.DOUBLE], [y, MPI.DOUBLE], op)
    return y


class RawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, config, comm=None):
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
        self.comm = None
        self.rank = 0
        self.comm_size = 1
        if comm is not None:
            self.comm = comm
            self.rank = comm.Get_rank()
            self.comm_size = comm.Get_size()

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

            filelist = sorted(os.listdir(raw_data_path))
            ## Random shuffle with reproducibility (both in single and multi processing)
            random.seed(43)
            random.shuffle(filelist)
            if self.comm is not None:
                x = np.array(len(filelist))
                y = self.comm.allreduce(x, op=MPI.MAX)
                assert x == y
                filelist = list(nsplit(filelist, self.comm_size))[self.rank]
                log("local filelist", len(filelist))
            for name in filelist:
                if name == ".DS_Store":
                    continue
                # if the directory contains file, iterate over them
                if os.path.isfile(os.path.join(raw_data_path, name)):
                    data_object = self.__transform_input_to_data_object_base(
                        filepath=os.path.join(raw_data_path, name)
                    )
                    if not isinstance(data_object, type(None)):
                        dataset.append(data_object)
                # if the directory contains subdirectories, explore their content
                elif os.path.isdir(os.path.join(raw_data_path, name)):
                    dir_name = os.path.join(raw_data_path, name)
                    for subname in os.listdir(dir_name):
                        if os.path.isfile(os.path.join(dir_name, subname)):
                            data_object = self.__transform_input_to_data_object_base(
                                filepath=os.path.join(dir_name, subname)
                            )
                            if not isinstance(data_object, type(None)):
                                dataset.append(data_object)

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
            print("serial_data_name", serial_data_name)
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(dataset_normalized, f)

    def __transform_input_to_data_object_base(self, filepath):
        if self.data_format == "LSMS" or self.data_format == "unit_test":
            data_object = self.__transform_LSMS_input_to_data_object_base(
                filepath=filepath
            )
        elif self.data_format == "CFG":
            data_object = self.__transform_CFG_input_to_data_object_base(
                filepath=filepath
            )
        return data_object

    def __transform_CFG_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data CFG file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        if filepath.endswith(".cfg"):

            data_object = self.__transform_ASE_object_to_data_object(filepath)

            return data_object

        else:
            return None

    def __transform_ASE_object_to_data_object(self, filepath):

        # FIXME:
        #  this still assumes bulk modulus is specific to the CFG format.
        #  To deal with multiple files across formats, one should generalize this function
        #  by moving the reading of the .bulk file in a standalone routine.
        #  Morevoer, this approach assumes tha there is only one global feature to look at,
        #  and that this global feature is specicially retrieveable in a file with the string *bulk* inside.

        ase_object = read_cfg(filepath)

        data_object = Data()

        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        masses = np.expand_dims(ase_object.arrays["masses"], axis=1)
        c_peratom = np.expand_dims(ase_object.arrays["c_peratom"], axis=1)
        fx = np.expand_dims(ase_object.arrays["fx"], axis=1)
        fy = np.expand_dims(ase_object.arrays["fy"], axis=1)
        fz = np.expand_dims(ase_object.arrays["fz"], axis=1)
        node_feature_matrix = np.concatenate(
            (proton_numbers, masses, c_peratom, fx, fy, fz), axis=1
        )
        data_object.x = tensor(node_feature_matrix).float()

        filename_without_extension = os.path.splitext(filepath)[0]

        if os.path.exists(os.path.join(filename_without_extension + ".bulk")):
            filename_bulk = os.path.join(filename_without_extension + ".bulk")
            f = open(filename_bulk, "r", encoding="utf-8")
            lines = f.readlines()
            graph_feat = lines[0].split(None, 2)
            g_feature = []
            # collect graph features
            for item in range(len(self.graph_feature_dim)):
                for icomp in range(self.graph_feature_dim[item]):
                    it_comp = self.graph_feature_col[item] + icomp
                    g_feature.append(float(graph_feat[it_comp].strip()))
            data_object.y = tensor(g_feature)

        return data_object

    def __transform_LSMS_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data LSMS file to Data object and returns it.

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

        f = open(filepath, "r", encoding="utf-8")

        lines = f.readlines()
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

        f.close()

        data_object.pos = tensor(node_position_matrix)
        data_object.x = tensor(node_feature_matrix)
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
            if dataset[idx].y is not None:
                dataset[idx].y[scaled_graph_feature_index] = (
                    dataset[idx].y[scaled_graph_feature_index] / data_object.num_nodes
                )
            if dataset[idx].x is not None:
                dataset[idx].x[:, scaled_node_feature_index] = (
                    dataset[idx].x[:, scaled_node_feature_index] / data_object.num_nodes
                )

        return dataset

    def __normalize_dataset(self):

        """Performs the normalization on Data objects and returns the normalized dataset."""
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
        if self.comm is not None:
            self.minmax_graph_feature[0, :] = comm_reduce(
                self.minmax_graph_feature[0, :], self.comm, MPI.MIN
            )
            self.minmax_graph_feature[1, :] = comm_reduce(
                self.minmax_graph_feature[1, :], self.comm, MPI.MAX
            )
            self.minmax_node_feature[0, :] = comm_reduce(
                self.minmax_node_feature[0, :], self.comm, MPI.MIN
            )
            self.minmax_node_feature[1, :] = comm_reduce(
                self.minmax_node_feature[1, :], self.comm, MPI.MAX
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
