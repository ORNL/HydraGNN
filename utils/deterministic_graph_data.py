import os
import shutil
import math
import torch
import numpy
from sklearn.neighbors import KNeighborsRegressor


def deterministic_graph_data(
    number_configurations: int = 1000,
    number_unit_cell_x: int = 2,
    number_unit_cell_y: int = 2,
    number_unit_cell_z: int = 1,
    number_clusters: int = 3,
    number_neighbors: int = 2,
):

    original_path = "output_files"
    if os.path.exists(original_path):
        shutil.rmtree(original_path)
    os.mkdir(original_path)

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

    number_nodes = 2 * number_unit_cell_x * number_unit_cell_y * number_unit_cell_z
    positions = torch.zeros(number_nodes, 3)

    assert (
        number_neighbors < number_nodes
    ), "Number of neighbors exceeds total number of nodes in the graph"

    # We assume that the unit cell is Body Center Cubic (BCC)
    count_pos = 0
    for x in range(0, number_unit_cell_x):
        for y in range(0, number_unit_cell_y):
            for z in range(0, number_unit_cell_z):
                positions[count_pos][0] = x
                positions[count_pos][1] = y
                positions[count_pos][2] = z
                positions[count_pos + 1][0] = x + 0.5
                positions[count_pos + 1][1] = y + 0.5
                positions[count_pos + 1][2] = z + 0.5
                count_pos = count_pos + 2

    for configuration in range(0, number_configurations):

        node_ids = torch.tensor([int(i) for i in range(0, number_nodes)]).reshape(
            (number_nodes, 1)
        )
        cluster_ids_x = torch.randint(0, number_clusters, (number_nodes, 1))

        node_feature = cluster_ids_x
        cluster_ids_x_square = node_feature ** 2
        cluster_ids_x_cube = node_feature ** 3

        # We use a K neraest neighbor model to average nodal features and simulate a message passing between neighboring nodes
        knn = KNeighborsRegressor(number_neighbors)
        knn.fit(positions, node_feature)
        node_feature = torch.Tensor(knn.predict(positions))
        cluster_ids_x_square = node_feature ** 2
        cluster_ids_x_cube = node_feature ** 3

        updated_table = torch.cat(
            (
                node_feature,
                node_ids,
                positions,
                cluster_ids_x,
                cluster_ids_x_square,
                cluster_ids_x_cube,
            ),
            1,
        )
        numpy_updated_table = updated_table.detach().numpy()

        total_value = (
            torch.sum(cluster_ids_x)
            + torch.sum(cluster_ids_x_square)
            + torch.sum(cluster_ids_x_cube)
        )
        numpy_total_value = total_value.detach().numpy()
        numpy_string_total_value = numpy.array2string(numpy_total_value)

        file = open(original_path + "/output" + str(configuration) + ".txt", "a")
        file.write(numpy_string_total_value)

        for index in range(0, number_nodes):
            numpy_row = numpy_updated_table[index, :]
            numpy_string_row = numpy.array2string(
                numpy_row, precision=2, separator="\t", suppress_small=True
            )
            file.write("\n")
            file.write(numpy_string_row.lstrip("[").rstrip("]"))

        file.close()

    final_path = "./dataset/unit_test"
    if os.path.isdir(final_path):
        shutil.rmtree(final_path)
    os.makedirs(final_path)
    _ = shutil.move(original_path, final_path, copy_function=shutil.copytree)
