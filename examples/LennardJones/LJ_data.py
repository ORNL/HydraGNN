##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

# General
import os
import logging
import numpy

numpy.set_printoptions(threshold=numpy.inf)
numpy.set_printoptions(linewidth=numpy.inf)

# Torch
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_scatter import scatter

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float64)

# Distributed
import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

# HydraGNN
from hydragnn.utils.datasets.abstractrawdataset import AbstractBaseDataset
from hydragnn.utils.distributed import nsplit
from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths

# Angstrom unit
primitive_bravais_lattice_constant_x = 3.8
primitive_bravais_lattice_constant_y = 3.8
primitive_bravais_lattice_constant_z = 3.8


##################################################################################################################


"""High-Level Function"""


def create_dataset(path, config):
    radius_cutoff = config["NeuralNetwork"]["Architecture"]["radius"]
    max_num_neighbors = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
    number_configurations = (
        config["Dataset"]["number_configurations"]
        if "number_configurations" in config["Dataset"]
        else 300
    )
    atom_types = [1]
    formula = LJpotential(1.0, 3.4)
    atomic_structure_handler = AtomicStructureHandler(
        atom_types,
        [
            primitive_bravais_lattice_constant_x,
            primitive_bravais_lattice_constant_y,
            primitive_bravais_lattice_constant_z,
        ],
        radius_cutoff,
        formula,
    )
    deterministic_graph_data(
        path,
        atom_types,
        atomic_structure_handler=atomic_structure_handler,
        radius_cutoff=radius_cutoff,
        max_num_neighbors=max_num_neighbors,
        relative_maximum_atomic_displacement=1e-1,
        number_configurations=number_configurations,
    )


"""Reading/Transforming Data"""


class LJDataset(AbstractBaseDataset):
    """LJDataset dataset class"""

    def __init__(self, dirpath, config, dist=False, sampling=None):
        super().__init__()

        self.dist = dist
        self.world_size = 1
        self.rank = 1
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        dirfiles = sorted(os.listdir(dirpath))

        rx = list(nsplit((dirfiles), self.world_size))[self.rank]

        # LPE
        self.transform = AddLaplacianEigenvectorPE(
            k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
            attr_name="pe",
            is_undirected=True,
        )

        for file in rx:
            filepath = os.path.join(dirpath, file)
            self.dataset.append(self.transform_input_to_data_object_base(filepath))

    def transform_input_to_data_object_base(self, filepath):

        # Using readline()
        file = open(filepath, "r")

        torch_data = torch.empty((0, 8), dtype=torch.float32)
        torch_supercell = torch.zeros((0, 3), dtype=torch.float32)

        count = 0

        while True:
            count += 1

            # Get next line from file
            line = file.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break

            if count == 1:
                total_energy = float(line)
            elif count == 2:
                energy_per_atom = float(line)
            elif 2 < count < 6:
                array_line = numpy.fromstring(line, dtype=float, sep="\t")
                torch_supercell = torch.cat(
                    [torch_supercell, torch.from_numpy(array_line).unsqueeze(0)], axis=0
                )
            elif count > 5:
                array_line = numpy.fromstring(line, dtype=float, sep="\t")
                torch_data = torch.cat(
                    [torch_data, torch.from_numpy(array_line).unsqueeze(0)], axis=0
                )
            # print("Line{}: {}".format(count, line.strip()))

        file.close()

        num_nodes = torch_data.shape[0]

        energy_pre_translation_factor = 0.0
        energy_pre_scaling_factor = 1.0 / num_nodes
        energy_per_atom_pretransformed = (
            energy_per_atom - energy_pre_translation_factor
        ) * energy_pre_scaling_factor
        grad_energy_post_scaling_factor = (
            1.0 / energy_pre_scaling_factor * torch.ones(num_nodes, 1)
        )
        forces = torch_data[:, [5, 6, 7]]
        forces_pre_scaling_factor = 1.0
        forces_pre_scaled = forces * forces_pre_scaling_factor

        data = Data(
            cell=torch_supercell.to(torch.float32),
            num_nodes=num_nodes,
            grad_energy_post_scaling_factor=grad_energy_post_scaling_factor,
            forces_pre_scaling_factor=torch.tensor(forces_pre_scaling_factor).to(
                torch.float32
            ),
            forces=forces,
            forces_pre_scaled=forces_pre_scaled,
            pos=torch_data[:, [1, 2, 3]].to(torch.float32),
            x=torch.cat([torch_data[:, [0, 4]]], axis=1).to(torch.float32),
            y=torch.tensor(total_energy).unsqueeze(0).to(torch.float32),
            energy_per_atom=torch.tensor(energy_per_atom_pretransformed)
            .unsqueeze(0)
            .to(torch.float32),
            energy=torch.tensor(total_energy).unsqueeze(0).to(torch.float32),
            pbc=torch.tensor(
                [
                    True,
                    True,
                    True,
                ],
                dtype=torch.bool,
            ),  # LJ example always has periodic boundary conditions
        )

        # Create pbc edges and lengths
        edge_creation = get_radius_graph_pbc(self.radius, self.max_neighbours)
        data = edge_creation(data)
        data = self.transform(data)
        # gps requires relative edge features, introduced rel_lapPe as edge encodings
        source_pe = data.pe[data.edge_index[0]]
        target_pe = data.pe[data.edge_index[1]]
        data.rel_pe = torch.abs(
            source_pe - target_pe
        )  # Compute feature-wise difference
        return data

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


"""Create Data"""


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
        os.makedirs(path, exist_ok=True)
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
    data.cell = torch.diag(
        torch.tensor([supercell_size_x, supercell_size_y, supercell_size_z])
    )
    data.pbc = torch.tensor([True, True, True], dtype=torch.bool)
    data.x = torch.cat([atom_types, positions], dim=1)

    create_graph_connectivity_pbc = get_radius_graph_pbc(
        radius_cutoff, max_num_neighbors
    )
    data = create_graph_connectivity_pbc(data)

    data = atomic_structure_handler.compute(data)

    total_energy = torch.sum(data.x[:, 4])
    energy_per_atom = total_energy / number_nodes

    total_energy_str = numpy.array2string(total_energy.detach().cpu().numpy())
    energy_per_atom_str = numpy.array2string(energy_per_atom.detach().cpu().numpy())
    filetxt = total_energy_str + "\n" + energy_per_atom_str

    for index in range(0, 3):
        numpy_row = data.cell[index, :].detach().numpy()
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


"""Function Calculation"""


class AtomicStructureHandler:
    def __init__(
        self, list_atom_types, bravais_lattice_constants, radius_cutoff, formula
    ):
        self.bravais_lattice_constants = bravais_lattice_constants
        self.radius_cutoff = radius_cutoff
        self.formula = formula

    # Calculate the potential energy with torch gradient tracking, then simply use autograd to calculate the forces
    def compute(self, data):
        # Instantiate
        assert data.pos.shape[0] == data.x.shape[0]
        node_potential = torch.zeros([data.pos.shape[0], 1])
        node_forces = torch.zeros([data.pos.shape[0], 3])

        # Calculate
        data.pos.requires_grad = True
        edge_vec, edge_dist = get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=data.edge_index,
            shifts=data.edge_shifts,
            normalize=False,
        )

        # Sum potential by edge, node, and total
        edge_potential = self.formula.potential_energy(
            edge_dist
        )  # Shape [num_edges, 1]
        node_potential = scatter(
            edge_potential,
            data.edge_index[0],
            dim=0,
            dim_size=data.pos.shape[0],
            reduce="add",
        )  # Shape [num_nodes, 1]
        total_potential = torch.sum(node_potential, dim=0, keepdim=True)  # Shape [1]

        # Autograd to calculate forces
        node_forces = -torch.autograd.grad(
            total_potential,
            data.pos,
            grad_outputs=torch.ones_like(total_potential),
        )[
            0
        ]  # Shape [num_nodes, 3]

        # Append to data
        data.x = torch.cat((data.x, node_potential, node_forces), dim=1)

        return data


class LJpotential:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def potential_energy(self, pair_distance):
        return (
            4
            * self.epsilon
            * ((self.sigma / pair_distance) ** 12 - (self.sigma / pair_distance) ** 6)
        )


"""Etc"""


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))
