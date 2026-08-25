##############################################################################
# Copyright (c) 2023, Oak Ridge National Laboratory                          #
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
from pathlib import Path
from torch_geometric.data import Data
from sklearn.neighbors import KNeighborsRegressor


def ensure_deterministic_graph_data(path, number_configurations, **kwargs):
    """Reuse a named-data fixture cache or rebuild an incompatible old cache."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    samples = sorted(directory.glob("*.pt"))
    cache_valid = len(samples) == number_configurations
    if cache_valid:
        first = torch.load(samples[0], weights_only=False)
        cache_valid = all(
            hasattr(first, name)
            for name in ("node_features", "sum_x_x2_x3", "edge_index", "edge_lengths")
        )
    if cache_valid:
        return
    for sample in samples:
        sample.unlink()
    deterministic_graph_data(
        str(directory), number_configurations=number_configurations, **kwargs
    )


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
    seed: int = 43,
):
    if types == None:
        types = range(number_types)

    generator = torch.Generator().manual_seed(seed + configuration_start)
    # We assume that the unit cell is Body Center Cubic (BCC)
    unit_cell_x = torch.randint(
        unit_cell_x_range[0],
        unit_cell_x_range[1],
        (number_configurations,),
        generator=generator,
    )
    unit_cell_y = torch.randint(
        unit_cell_y_range[0],
        unit_cell_y_range[1],
        (number_configurations,),
        generator=generator,
    )
    unit_cell_z = torch.randint(
        unit_cell_z_range[0],
        unit_cell_z_range[1],
        (number_configurations,),
        generator=generator,
    )

    samples = []
    for configuration in range(number_configurations):
        uc_x = unit_cell_x[configuration]
        uc_y = unit_cell_y[configuration]
        uc_z = unit_cell_z[configuration]
        samples.append(
            create_configuration(
                configuration,
                configuration_start,
                uc_x,
                uc_y,
                uc_z,
                types,
                number_neighbors,
                linear_only,
                generator,
            )
        )

    # The retired raw test-data loader normalized every declared feature. Keep
    # that fixture behavior here so model-quality thresholds remain meaningful
    # while production loaders stay schema-only and do not reinterpret values.
    for name in (
        "node_features",
        "x_target",
        "x2",
        "x3",
        "xx2_vec",
        "x2x3_vec",
        "sum_x_x2_x3",
        "sum",
        "sums_vec",
        "sum_linear",
    ):
        values = torch.cat([getattr(data, name).reshape(-1) for data in samples])
        minimum, maximum = values.min(), values.max()
        scale = maximum - minimum
        for data in samples:
            value = getattr(data, name)
            setattr(data, name, (value - minimum) / scale if scale else value * 0)

    for configuration, data in enumerate(samples):
        filename = os.path.join(
            path, "output" + str(configuration + configuration_start) + ".pt"
        )
        torch.save(data, filename)


def create_configuration(
    configuration,
    configuration_start,
    uc_x,
    uc_y,
    uc_z,
    types,
    number_neighbors,
    linear_only,
    generator,
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
    node_feature = torch.randint(
        min(types), max(types) + 1, (number_nodes, 1), generator=generator
    )

    if linear_only:
        node_output_x = node_feature
    else:
        # We use a K nearest neighbor model to average nodal features and simulate a message passing between neighboring nodes
        knn = KNeighborsRegressor(number_neighbors)
        knn.fit(positions, node_feature)
        node_output_x = torch.Tensor(knn.predict(positions))

    node_output_x_square = node_output_x**2 + node_feature
    node_output_x_cube = node_output_x**3

    if linear_only:
        total_value = torch.sum(node_output_x)
    else:
        total_value_linear = torch.sum(node_output_x)
        total_value = (
            torch.sum(node_output_x)
            + torch.sum(node_output_x_square)
            + torch.sum(node_output_x_cube)
        )
    total_value = total_value.reshape(1, 1).float()
    total_value_linear = (
        total_value if linear_only else total_value_linear.reshape(1, 1).float()
    )
    data = Data(
        node_features=node_feature.float(),
        x_target=node_feature.float(),
        x2=node_output_x_square.float(),
        x3=node_output_x_cube.float(),
        xx2_vec=torch.cat((node_feature, node_ids), dim=1).float(),
        x2x3_vec=torch.cat((node_output_x_square, node_output_x_cube), dim=1).float(),
        sum_x_x2_x3=total_value,
        sum=total_value,
        sums_vec=torch.cat((total_value, total_value_linear), dim=1),
        sum_linear=total_value_linear,
        global_conditioning=(node_feature == node_feature[0])
        .sum()
        .reshape(1, 1)
        .float(),
        pos=positions.float(),
    )
    distances = torch.cdist(data.pos, data.pos)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.topk(
        min(number_neighbors, number_nodes - 1), largest=False
    ).indices
    targets = torch.arange(number_nodes).repeat_interleave(neighbors.shape[1])
    sources = neighbors.reshape(-1)
    data.edge_index = torch.stack((sources, targets))
    data.edge_lengths = torch.linalg.vector_norm(
        data.pos[sources] - data.pos[targets], dim=1, keepdim=True
    )
    return data
