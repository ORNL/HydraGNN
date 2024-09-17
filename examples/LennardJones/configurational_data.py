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
import torch
import numpy

numpy.set_printoptions(threshold=numpy.inf)
numpy.set_printoptions(linewidth=numpy.inf)

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

from torch_geometric.data import Data

from graph_utils import get_radius_graph_pbc
from AtomicStructure import AtomicStructureHandler
from LJpotential import LJpotential

from distributed_utils import nsplit

from mpi4py import MPI

# Angstrom unit
primitive_bravais_lattice_constant_x = 3.8
primitive_bravais_lattice_constant_y = 3.8
primitive_bravais_lattice_constant_z = 3.8


def deterministic_graph_data(
    path: str,
    atom_types: list,
    atomic_structure_handler,
    radius_cutoff=float("inf"),
    max_num_neighbors=float("inf"),
    number_configurations: int = 500,
    configuration_start: int = 0,
    unit_cell_x_range: list = [3, 4],
    unit_cell_y_range: list = [3, 4],
    unit_cell_z_range: list = [3, 4],
    relative_maximum_atomic_displacement: float = 1e-1,
):

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    torch.manual_seed(comm_rank)

    if 0 == comm_rank:
        os.makedirs(path, exist_ok=False)
    comm.Barrier()

    # We assume that the unit cell is Simple Center Cubic (SCC)
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

    configurations_list = range(number_configurations)
    rx = list(nsplit(configurations_list, comm_size))[comm_rank]

    for configuration in configurations_list[rx.start : rx.stop]:
        uc_x = unit_cell_x[configuration]
        uc_y = unit_cell_y[configuration]
        uc_z = unit_cell_z[configuration]
        create_configuration(
            path,
            atomic_structure_handler,
            configuration,
            configuration_start,
            uc_x,
            uc_y,
            uc_z,
            atom_types,
            radius_cutoff,
            max_num_neighbors,
            relative_maximum_atomic_displacement,
        )


def create_configuration(
    path,
    atomic_structure_handler,
    configuration,
    configuration_start,
    uc_x,
    uc_y,
    uc_z,
    types,
    radius_cutoff,
    max_num_neighbors,
    relative_maximum_atomic_displacement,
):
    ###############################################################################################
    ###################################   STRUCTURE OF THE DATA  ##################################
    ###############################################################################################

    #   GLOBAL_OUTPUT1
    #   GLOBAL_OUTPUT2
    #   NODE1_FEATURE   NODE1_INDEX     NODE1_COORDINATE_X  NODE1_COORDINATE_Y  NODE1_COORDINATE_Z  NODAL_OUTPUT1   NODAL_OUTPUT2   NODAL_OUTPUT3
    #   NODE2_FEATURE   NODE2_INDEX     NODE2_COORDINATE_X  NODE2_COORDINATE_Y  NODE2_COORDINATE_Z  NODAL_OUTPUT1   NODAL_OUTPUT2   NODAL_OUTPUT3
    #   ...
    #   NODENn_FEATURE   NODEn_INDEX     NODEn_COORDINATE_X  NODEn_COORDINATE_Y  NODEn_COORDINATE_Z  NODAL_OUTPUT1   NODAL_OUTPUT2   NODAL_OUTPUT3

    ###############################################################################################
    #################################   FORMULAS FOR NODAL FEATURE  ###############################
    ###############################################################################################

    #   NODAL_FEATURE = ATOM SPECIES

    ###############################################################################################
    ##########################   FORMULAS FOR GLOBAL AND NODAL OUTPUTS  ###########################
    ###############################################################################################

    #   GLOBAL_OUTPUT = TOTAL ENERGY
    #   GLOBAL_OUTPUT = TOTAL ENERGY / NUMBER OF NODES
    #   NODAL_OUTPUT1(X) = FORCE ACTING ON ATOM IN X DIRECTION
    #   NODAL_OUTPUT2(X) = FORCE ACTING ON ATOM IN Y DIRECTION
    #   NODAL_OUTPUT3(X) = FORCE ACTING ON ATOM IN Z DIRECTION

    ###############################################################################################
    count_pos = 0
    number_nodes = uc_x * uc_y * uc_z
    positions = torch.zeros(number_nodes, 3)
    for x in range(uc_x):
        for y in range(uc_y):
            for z in range(uc_z):
                positions[count_pos][0] = (
                    x
                    + relative_maximum_atomic_displacement
                    * ((torch.rand(1, 1).item()) - 0.5)
                ) * primitive_bravais_lattice_constant_x
                positions[count_pos][1] = (
                    y
                    + relative_maximum_atomic_displacement
                    * ((torch.rand(1, 1).item()) - 0.5)
                ) * primitive_bravais_lattice_constant_y
                positions[count_pos][2] = (
                    z
                    + relative_maximum_atomic_displacement
                    * ((torch.rand(1, 1).item()) - 0.5)
                ) * primitive_bravais_lattice_constant_z

                count_pos = count_pos + 1

    atom_types = torch.randint(min(types), max(types) + 1, (number_nodes, 1))

    data = Data()

    data.pos = positions
    supercell_size_x = primitive_bravais_lattice_constant_x * uc_x
    supercell_size_y = primitive_bravais_lattice_constant_y * uc_y
    supercell_size_z = primitive_bravais_lattice_constant_z * uc_z
    data.supercell_size = torch.diag(
        torch.tensor([supercell_size_x, supercell_size_y, supercell_size_z])
    )

    create_graph_connectivity_pbc = get_radius_graph_pbc(
        radius_cutoff, max_num_neighbors
    )
    data = create_graph_connectivity_pbc(data)

    atomic_descriptors = torch.cat(
        (
            atom_types,
            positions,
        ),
        1,
    )

    data.x = atomic_descriptors

    data = atomic_structure_handler.compute(data)

    total_energy = torch.sum(data.x[:, 4])
    energy_per_atom = total_energy / number_nodes

    total_energy_str = numpy.array2string(total_energy.detach().numpy())
    energy_per_atom_str = numpy.array2string(energy_per_atom.detach().numpy())
    filetxt = total_energy_str + "\n" + energy_per_atom_str

    for index in range(0, 3):
        numpy_row = data.supercell_size[index, :].detach().numpy()
        numpy_string_row = numpy.array2string(numpy_row, precision=64, separator="\t")
        filetxt += "\n" + numpy_string_row.lstrip("[").rstrip("]")

    for index in range(0, number_nodes):
        numpy_row = data.x[index, :].detach().numpy()
        numpy_string_row = numpy.array2string(numpy_row, precision=64, separator="\t")
        filetxt += "\n" + numpy_string_row.lstrip("[").rstrip("]")

    filename = os.path.join(
        path, "output" + str(configuration + configuration_start) + ".txt"
    )
    with open(filename, "w") as f:
        f.write(filetxt)