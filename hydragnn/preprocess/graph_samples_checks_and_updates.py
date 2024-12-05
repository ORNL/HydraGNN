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
from torch_geometric.utils import remove_self_loops, degree
from torch_geometric.data import Data

import ase
import ase.neighborlist
import os

from .dataset_descriptors import AtomFeatures


## This function can be slow if datasets is too large. Use with caution.
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
        data.edge_shifts = None
        assert (
            "batch" not in data
        ), "Periodic boundary conditions not currently supported on batches."
        assert hasattr(
            data, "supercell_size"
        ), "The data must contain the size of the supercell to apply periodic boundary conditions."
        assert hasattr(
            data, "pbc"
        ), "The data must contain data.pbc as a bool (True) or list of bools for the dimensions ([True, False, True]) to apply periodic boundary conditions."
        # NOTE Cutoff radius being less than half the smallest supercell dimension is a sufficient, but not necessary condition for no dupe connections.
        #      However, to prevent an issue from being unobserved until long into an experiment, we assert this condition.
        assert (
            self.r < min(torch.diagonal(data.supercell_size)) / 2
        ), "Cutoff radius must be smaller than half the smallest supercell dimension."
        ase_atom_object = ase.Atoms(
            positions=data.pos,
            cell=data.supercell_size,
            pbc=data.pbc,
        )
        # 'i' : first atom index
        # 'j' : second atom index
        # 'd' : absolute distance
        # 'S' : shift vector
        # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
        (
            edge_src,
            edge_dst,
            edge_length,
            edge_cell_shifts,
        ) = ase.neighborlist.neighbor_list(
            "ijdS", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop
        )
        data.edge_index = torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)],
            dim=0,  # Shape: [2, n_edges]
        )

        # ensure no duplicate edges
        unique_edge_index, unique_indices = torch.unique(
            data.edge_index, dim=1, return_inverse=False  # Shape: [n_edges]
        )
        assert unique_edge_index.unsqueeze(0).size(1) == data.edge_index.size(
            1
        ), "Adding periodic boundary conditions would result in duplicate edges. Cutoff radius must be reduced or system size increased."

        data.edge_attr = torch.tensor(edge_length, dtype=torch.float).unsqueeze(
            1
        )  # Shape: [n_edges, 1]
        # ASE returns whether the cell was shifted or not (-1,0,1). Multiply by the cell size to get the actual shift
        data.edge_shifts = torch.matmul(
            torch.tensor(edge_cell_shifts).float(), data.supercell_size.float()
        )  # Shape: [n_edges, 3]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"


def gather_deg(dataset):
    from hydragnn.utils.print.print_utils import iterate_tqdm

    backend = os.getenv("HYDRAGNN_AGGR_BACKEND", "torch")
    if backend == "torch":
        return gather_deg_dist(dataset)
    elif backend == "mpi":
        return gather_deg_mpi(dataset)
    else:
        max_deg = 0
        for data in iterate_tqdm(dataset, 2, desc="Degree max"):
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_deg = max(max_deg, max(d))

        deg = torch.zeros(max_deg + 1, dtype=torch.long)
        for data in iterate_tqdm(dataset, 2, desc="Degree bincount"):
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg


def gather_deg_dist(dataset):
    import torch.distributed as dist
    from hydragnn.utils.print.print_utils import iterate_tqdm
    from hydragnn.utils.distributed import get_device

    max_deg = 0
    for data in iterate_tqdm(dataset, 2, desc="Degree max"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_deg = max(max_deg, int(d.max()))
    max_deg = torch.tensor(max_deg, requires_grad=False).to(get_device())
    dist.all_reduce(max_deg, op=dist.ReduceOp.MAX)

    deg = torch.zeros(max_deg.item() + 1, requires_grad=False, dtype=torch.long)
    for data in iterate_tqdm(dataset, 2, desc="Degree bincount"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    deg = deg.clone().detach().to(get_device())
    dist.all_reduce(deg, op=dist.ReduceOp.SUM)
    return deg.cpu().detach().numpy()


def gather_deg_mpi(dataset):
    from mpi4py import MPI
    from hydragnn.utils.print.print_utils import iterate_tqdm

    max_deg = 0
    for data in iterate_tqdm(dataset, 2, desc="Degree max"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_deg = max(max_deg, max(d))
    max_deg = MPI.COMM_WORLD.allreduce(max_deg, op=MPI.MAX)

    deg = torch.zeros(max_deg + 1, dtype=torch.long)
    for data in iterate_tqdm(dataset, 2, desc="Degree bincount"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    deg = MPI.COMM_WORLD.allreduce(deg.numpy(), op=MPI.SUM)
    return deg


def update_predicted_values(
    type: list, index: list, graph_feature_dim: list, node_feature_dim: list, data: Data
):
    """Updates values of the structure we want to predict. Predicted value is represented by integer value.
    Parameters
    ----------
    type: "graph" level or "node" level
    index: index/location in data.y for graph level and in data.x for node level
    graph_feature_dim: list of integers to trak the dimension of each graph level feature
    data: Data
        A Data object representing a structure that has atoms.
    """
    output_feature = []
    data.y_loc = torch.zeros(1, len(type) + 1, dtype=torch.int64, device=data.x.device)
    for item in range(len(type)):
        if type[item] == "graph":
            index_counter_global_y = sum(graph_feature_dim[: index[item]])
            feat_ = torch.reshape(
                data.y[
                    index_counter_global_y : index_counter_global_y
                    + graph_feature_dim[index[item]]
                ],
                (graph_feature_dim[index[item]], 1),
            )
            # after the global features are spanned, we need to iterate over the nodal features
            # to do so, the counter of the nodal features need to start from the last value of counter for the graph nodel feature
        elif type[item] == "node":
            index_counter_nodal_y = sum(node_feature_dim[: index[item]])
            feat_ = torch.reshape(
                data.x[
                    :,
                    index_counter_nodal_y : (
                        index_counter_nodal_y + node_feature_dim[index[item]]
                    ),
                ],
                (-1, 1),
            )
        elif type[item] == "pos":
            # index_counter_nodal_y = sum(node_feature_dim[: index[item]])
            feat_ = torch.reshape(
                data.pos[:, : node_feature_dim[index[item]]],
                (-1, 1),
            )
        else:
            raise ValueError("Unknown output type", type[item])
        output_feature.append(feat_)
        data.y_loc[0, item + 1] = data.y_loc[0, item] + feat_.shape[0] * feat_.shape[1]
    data.y = torch.cat(output_feature, 0)


def update_atom_features(atom_features: [AtomFeatures], data: Data):
    """Updates atom features of a structure. An atom is represented with x,y,z coordinates and associated features.

    Parameters
    ----------
    atom_features: [AtomFeatures]
        List of features to update. Each feature is instance of Enum AtomFeatures.
    data: Data
        A Data object representing a structure that has atoms.
    """
    feature_indices = [i for i in atom_features]
    data.x = data.x[:, feature_indices]
