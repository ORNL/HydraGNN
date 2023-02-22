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
import os

## This function can be slow if dataset is too large. Use with caution.
## Recommend to use check_if_graph_size_variable_dist
def check_if_graph_size_variable(train_loader, val_loader, test_loader):
    backend = os.getenv("HYDRAGNN_AGGR_BACKEND", "torch")
    if backend == "torch":
        return check_if_graph_size_variable_dist(train_loader, val_loader, test_loader)
    elif backend == "mpi":
        return check_if_graph_size_variable_mpi(train_loader, val_loader, test_loader)
    else:
        graph_size_variable = False
        nodes_num_list = []
        for loader in [train_loader, val_loader, test_loader]:
            for data in loader.dataset:
                nodes_num_list.append(data.num_nodes)
            if len(list(set(nodes_num_list))) > 1:
                graph_size_variable = True
                return graph_size_variable
        return graph_size_variable


def check_if_graph_size_variable_dist(train_loader, val_loader, test_loader):
    from hydragnn.utils.distributed import get_device

    assert torch.distributed.is_initialized()
    graph_size_variable = False
    nodes_num_list = []
    for loader in [train_loader, val_loader, test_loader]:
        for data in loader:
            nodes_num_list.append(data.num_nodes)
            if len(list(set(nodes_num_list))) > 1:
                graph_size_variable = True
                break
        if graph_size_variable:
            break
    b = 1 if graph_size_variable else 0
    b = torch.tensor(b).to(get_device())
    torch.distributed.all_reduce(b, op=torch.distributed.ReduceOp.SUM)
    reduced = True if b.item() > 0 else False
    return reduced


def check_if_graph_size_variable_mpi(train_loader, val_loader, test_loader):
    graph_size_variable = False
    nodes_num_list = []
    for loader in [train_loader, val_loader, test_loader]:
        for data in loader:
            nodes_num_list.append(data.num_nodes)
            if len(list(set(nodes_num_list))) > 1:
                graph_size_variable = True
                break
        if graph_size_variable:
            break
    from mpi4py import MPI

    b = 1 if graph_size_variable else 0
    rb = MPI.COMM_WORLD.allreduce(b, op=MPI.SUM)
    reduced = True if rb > 0 else False
    return reduced


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


def get_radius_graph(radius, max_neighbours, loop=False):
    return RadiusGraph(
        r=radius,
        loop=loop,
        max_num_neighbors=max_neighbours,
    )


def get_radius_graph_pbc(radius, max_neighbours, loop=False):
    return RadiusGraphPBC(
        r=radius,
        loop=loop,
        max_num_neighbors=max_neighbours,
    )


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
        assert (
            "batch" not in data
        ), "Periodic boundary conditions not currently supported on batches."
        assert hasattr(
            data, "supercell_size"
        ), "The data must contain the size of the supercell to apply periodic boundary conditions."
        ase_atom_object = ase.Atoms(
            positions=data.pos,
            cell=data.supercell_size,
            pbc=True,
        )
        # ‘i’ : first atom index
        # ‘j’ : second atom index
        # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
        edge_src, edge_dst, edge_length = ase.neighborlist.neighbor_list(
            "ijd", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop
        )
        data.edge_index = torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        )

        # ensure no duplicate edges
        num_edges = data.edge_index.size(1)
        data.coalesce()
        assert num_edges == data.edge_index.size(
            1
        ), "Adding periodic boundary conditions would result in duplicate edges. Cutoff radius must be reduced or system size increased."

        data.edge_attr = torch.tensor(edge_length, dtype=torch.float).unsqueeze(1)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"
