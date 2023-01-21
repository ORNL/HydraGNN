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
import shutil
import math
import torch
import numpy
from sklearn.neighbors import KNeighborsRegressor


def deterministic_graph_data(
    path: str,
    number_configurations: int = 500,
    configuration_start: int = 0,
    unit_cell_x_range: list = [1, 3],
    unit_cell_y_range: list = [1, 3],
    unit_cell_z_range: list = [1, 2],
    number_types: int = 3,
    types: list = None,
    number_neighbors: int = 2,
    linear_only=False,
):
    if types == None:
        types = range(number_types)

    # We assume that the unit cell is Body Center Cubic (BCC)
    unit_cell_x = torch.randint(
        unit_cell_x_range[0],
        unit_cell_x_range[1],
        (number_configurations,),
    )
    unit_cell_y = torch.randint(
        unit_cell_y_range[0],
        unit_cell_y_range[1],
        (number_configurations,),
    )
    unit_cell_z = torch.randint(
        unit_cell_z_range[0],
        unit_cell_z_range[1],
        (number_configurations,),
    )

    for configuration in range(number_configurations):
        uc_x = unit_cell_x[configuration]
        uc_y = unit_cell_y[configuration]
        uc_z = unit_cell_z[configuration]
        create_configuration(
            path,
            configuration,
            configuration_start,
            uc_x,
            uc_y,
            uc_z,
            types,
            number_neighbors,
            linear_only,
        )


def create_configuration(
    path,
    configuration,
    configuration_start,
    uc_x,
    uc_y,
    uc_z,
    types,
    number_neighbors,
    linear_only,
):
    ###############################################################################################
    ###################################   STRUCTURE OF THE DATA  ##################################
    ###############################################################################################

    #   GLOCAL_OUTPUT
    #   NODE1_FEATURE   NODE1_INDEX     NODE1_COORDINATE_X  NODE1_COORDINATE_Y  NODE1_COORDINATE_Z  NODAL_OUTPUT1   NODAL_OUTPUT2   NODAL_OUTPUT3
    #   NODE2_FEATURE   NODE2_INDEX     NODE2_COORDINATE_X  NODE2_COORDINATE_Y  NODE2_COORDINATE_Z  NODAL_OUTPUT1   NODAL_OUTPUT2   NODAL_OUTPUT3
    #   ...
    #   NODENn_FEATURE   NODEn_INDEX     NODEn_COORDINATE_X  NODEn_COORDINATE_Y  NODEn_COORDINATE_Z  NODAL_OUTPUT1   NODAL_OUTPUT2   NODAL_OUTPUT3

    ###############################################################################################
    #################################   FORMULAS FOR NODAL FEATURE  ###############################
    ###############################################################################################

    #   NODAL_FEATURE = MODULUS( NODE_INDEX, NUM_CLUSTERS )

    ###############################################################################################
    ##########################   FORMULAS FOR GLOBAL AND NODAL OUTOUTS  ###########################
    ###############################################################################################

    #   GLOBAL_OUTPUT = SUM_OVER_NODES ( NODAL_OUTPUT1 ) + SUM_OVER_NODES ( NODAL_OUTPUT2 ) + SUM_OVER_NODES ( NODAL_OUTPUT3 )
    #   NODAL_OUTPUT1(X) = X
    #   NODAL_OUTPUT2(X) = X^2
    #   NODAL_OUTPUT3(X) = X^3

    ###############################################################################################
    count_pos = 0
    number_nodes = 2 * uc_x * uc_y * uc_z
    positions = torch.zeros(number_nodes, 3)
    for x in range(uc_x):
        for y in range(uc_y):
            for z in range(uc_z):
                positions[count_pos][0] = x
                positions[count_pos][1] = y
                positions[count_pos][2] = z
                positions[count_pos + 1][0] = x + 0.5
                positions[count_pos + 1][1] = y + 0.5
                positions[count_pos + 1][2] = z + 0.5
                count_pos = count_pos + 2

    node_ids = torch.tensor(range(number_nodes), dtype=torch.int64).reshape(
        (number_nodes, 1)
    )
    node_feature = torch.randint(min(types), max(types) + 1, (number_nodes, 1))

    if linear_only:
        node_output_x = node_feature
    else:
        # We use a K nearest neighbor model to average nodal features and simulate a message passing between neighboring nodes
        knn = KNeighborsRegressor(number_neighbors)
        knn.fit(positions, node_feature)
        node_output_x = torch.Tensor(knn.predict(positions))

    node_output_x_square = node_output_x ** 2 + node_feature
    node_output_x_cube = node_output_x ** 3

    updated_table = torch.cat(
        (
            node_feature,
            node_ids,
            positions,
            node_output_x,
            node_output_x_square,
            node_output_x_cube,
        ),
        1,
    )
    updated_table = updated_table.detach().numpy()

    if linear_only:
        total_value = torch.sum(node_output_x)
    else:
        total_value_linear = torch.sum(node_output_x)
        total_value = (
            torch.sum(node_output_x)
            + torch.sum(node_output_x_square)
            + torch.sum(node_output_x_cube)
        )
    filetxt = numpy.array2string(total_value.detach().numpy())
    if not linear_only:
        filetxt += "\t" + numpy.array2string(total_value_linear.detach().numpy())

    for index in range(0, number_nodes):
        numpy_row = updated_table[index, :]
        numpy_string_row = numpy.array2string(
            numpy_row, precision=2, separator="\t", suppress_small=True
        )
        filetxt += "\n" + numpy_string_row.lstrip("[").rstrip("]")

    filename = os.path.join(
        path, "output" + str(configuration + configuration_start) + ".txt"
    )
    with open(filename, "w") as f:
        f.write(filetxt)
