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

from torch_geometric.data import Data
from torch import tensor

from hydragnn.preprocess.raw_dataset_loader import AbstractRawDataLoader

# WARNING: DO NOT use collective communication calls here because only rank 0 uses this routines


class LSMS_RawDataLoader(AbstractRawDataLoader):
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, config, dist=False):
        super(LSMS_RawDataLoader, self).__init__(config, dist)

    def transform_input_to_data_object_base(self, filepath):
        data_object = self.__transform_LSMS_input_to_data_object_base(filepath=filepath)

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
        data_object = self.__charge_density_update_for_LSMS(data_object)
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
