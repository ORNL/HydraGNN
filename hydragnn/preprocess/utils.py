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

import torch
from torch_geometric.transforms import RadiusGraph
from torch_geometric.utils import remove_self_loops

import ase
import ase.neighborlist


def check_if_graph_size_variable(train_loader, val_loader, test_loader):
    graph_size_variable = False
    nodes_num_list = []
    for loader in [train_loader, val_loader, test_loader]:
        for data in loader.dataset:
            nodes_num_list.append(data.num_nodes)
            if len(list(set(nodes_num_list))) > 1:
                graph_size_variable = True
                return graph_size_variable
    return graph_size_variable


def check_data_samples_equivalence(data1, data2, tol):
    x_bool = data1.x.shape == data2.x.shape
    pos_bool = data1.pos.shape == data2.pos.shape
    y_bool = data1.y.shape == data2.y.shape

    found = [False for i in range(data1.edge_index.shape[1])]
    for i in range(data1.edge_index.shape[1]):
        for j in range(data1.edge_index.shape[1]):
            if torch.equal(data1.edge_index[:, i], data2.edge_index[:, j]):
                found[i] = True
                # Due to numerics, the assert check fails for discrepancies below 1e-6
                assert torch.norm(data1.edge_attr[i] - data2.edge_attr[j]) < tol
                break

    edge_bool = all(found)

    return x_bool and pos_bool and y_bool and edge_bool


def get_radius_graph_config(config, loop=False):
    return RadiusGraph(
        r=config["radius"],
        loop=loop,
        max_num_neighbors=config["max_neighbours"],
    )


def get_radius_graph_pbc_config(config, loop=False):
    return RadiusGraphPBC(
        r=config["radius"],
        loop=loop,
        max_num_neighbors=config["max_neighbours"],
    )


class RadiusGraphPBC(RadiusGraph):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance, including periodic images.
    """

    def __call__(self, data):
        data.edge_attr = None
        data.pbc = True
        assert (
            "batch" not in data
        ), "Periodic boundary conditions not currently supported on batches."
        assert hasattr(
            data, "supercell_size"
        ), "The data must contain the size of the supercell to apply periodic boundary conditions."
        assert hasattr(
            data, "atom_types"
        ), "The data must contain information about the atoms types. Can be a chemical symbol (str) or an atomic number (int)."
        ase_atom_object = ase.Atoms(
            symbols=data.atom_types,
            positions=data.pos,
            cell=data.supercell_size,
            pbc=True,
        )
        # ‘i’ : first atom index
        # ‘j’ : second atom index
        # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
        edge_src, edge_dst = ase.neighborlist.neighbor_list(
            "ij", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop
        )
        distance_matrix = ase_atom_object.get_all_distances(mic=True)
        data.edge_index = torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        )

        # remove duplicate edges
        data.coalesce()

        # remove self loops in a graph
        if not self.loop:
            data.edge_index = remove_self_loops(data.edge_index, None)[0]

        data.edge_attr = torch.zeros(data.edge_index.shape[1], 1)
        for index in range(0, data.edge_index.shape[1]):
            data.edge_attr[index, 0] = distance_matrix[
                data.edge_index[0, index], data.edge_index[1, index]
            ]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"
