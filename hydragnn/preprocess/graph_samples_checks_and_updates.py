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
from torch_scatter import scatter
from torch_geometric.transforms import RadiusGraph
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, degree
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import LocalCartesian, Distance

import ase
import ase.neighborlist
import numpy as np
import os
from typing import Tuple

from .dataset_descriptors import AtomFeatures
from hydragnn.utils.distributed import get_device


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
    given distance, including periodic images, and limits the number of neighbors per node.
    """

    def __call__(self, data):
        # Checks for attributes and ensures data type and device consistency
        data, device, dtype = self._check_and_standardize_data(
            data
        )  # dtype gives us whether to use float32 or float64

        ase_atom_object = ase.Atoms(
            positions=data.pos.cpu().numpy(),
            cell=data.cell.cpu().numpy(),
            pbc=data.pbc.cpu().tolist(),
        )
        # 'i' : first atom index
        # 'j' : second atom index
        # 'd' : absolute distance
        # 'S' : shift vector
        # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
        cutoff = self.r
        cutoff_multiplier = 1.25
        max_attempts = 3
        for attempt_num in range(max_attempts):
            (
                edge_src,
                edge_dst,
                edge_length,
                edge_cell_shifts,
            ) = ase.neighborlist.neighbor_list(
                "ijdS",
                a=ase_atom_object,
                cutoff=cutoff,
                self_interaction=True,  # We want self-interactions across periodic boundaries
            )

            # Eliminate true self-loops if desired
            if not self.loop:
                (
                    edge_src,
                    edge_dst,
                    edge_length,
                    edge_cell_shifts,
                ) = self._remove_true_self_loops(
                    edge_src, edge_dst, edge_length, edge_cell_shifts
                )

            # Limit the in-degree per node
            edge_src, edge_dst, edge_length, edge_cell_shifts = self._limit_neighbors(
                edge_src,
                edge_dst,
                edge_length,
                edge_cell_shifts,
                self.max_num_neighbors,
            )

            # If all nodes appear in edge_dst, break and proceed
            if np.unique(edge_dst).size == data.num_nodes:
                break
            elif attempt_num < max_attempts - 1:
                # Otherwise, expand radius and retry
                print(
                    f"Not all nodes receive an edge, expanding radius from {cutoff} -> {cutoff * cutoff_multiplier}",
                    flush=True,
                )
                cutoff *= cutoff_multiplier
            else:
                # Ensure each node has an in-degree >= 1
                (
                    edge_src,
                    edge_dst,
                    edge_length,
                    edge_cell_shifts,
                ) = self._ensure_connected(
                    data.num_nodes,
                    cutoff,
                    edge_src,
                    edge_dst,
                    edge_length,
                    edge_cell_shifts,
                )

        # Assign to data
        data.edge_index = torch.stack(
            [
                torch.tensor(edge_src, dtype=torch.long, device=device),
                torch.tensor(edge_dst, dtype=torch.long, device=device),
            ],
            dim=0,  # Shape: [2, n_edges]
        )
        # ASE returns the integer number of cell shifts. Multiply by the cell size to get the shift vector.
        data.edge_shifts = torch.matmul(
            torch.tensor(edge_cell_shifts, dtype=dtype, device=device),
            data.cell,
        )  # Shape: [n_edges, 3]

        return data

    def _remove_true_self_loops(
        self, edge_src, edge_dst, edge_length, edge_cell_shifts
    ):
        # Create a mask to remove true self loops (i.e. the same source and destination node in the same cell)
        true_self_edges = edge_src == edge_dst
        true_self_edges &= np.all(edge_cell_shifts == 0, axis=1)
        mask = ~true_self_edges

        # Apply the mask and return
        return (
            edge_src[mask],
            edge_dst[mask],
            edge_length[mask],
            edge_cell_shifts[mask],
        )

    def _limit_neighbors(
        self, edge_src, edge_dst, edge_length, edge_cell_shifts, max_num_neighbors
    ):
        # Lexsort will sort primarily by edge_dst, then by edge_length within each src node
        sorted_indices = np.lexsort((edge_length, edge_dst))
        edge_src, edge_dst, edge_length, edge_cell_shifts = [
            edge_arg[sorted_indices]
            for edge_arg in [edge_src, edge_dst, edge_length, edge_cell_shifts]
        ]

        # Create a mask to keep only `max_num_neighbors` per node
        unique_dst, counts = np.unique(edge_dst, return_counts=True)
        mask = np.zeros_like(edge_dst, dtype=bool)
        start_idx = 0
        for dst, count in zip(unique_dst, counts):
            end_idx = start_idx + count
            # Keep only the first max_num_neighbors for this src
            mask[start_idx : start_idx + min(count, max_num_neighbors)] = True
            start_idx = end_idx

        # Apply the mask and return
        return (
            edge_src[mask],
            edge_dst[mask],
            edge_length[mask],
            edge_cell_shifts[mask],
        )

    def _ensure_connected(
        self, num_nodes, cutoff, edge_src, edge_dst, edge_length, edge_cell_shifts
    ):
        # In worst cases, create a purely artificial connection
        all_nodes = np.arange(num_nodes)
        missing = np.setdiff1d(all_nodes, np.unique(edge_dst))
        if len(missing) > 0:
            print(
                f"WARNING: Some nodes are still missing in 'edge_dst'. They will be constructed artificially.",
                flush=True,
            )
            for mnode in missing:
                # Connect a random node from [0..num_nodes-1] not equal to the src. Use cutoff for dist and 0 for edge shifts
                if num_nodes > 1:
                    src_node = np.random.choice(all_nodes[all_nodes != mnode])
                else:
                    src_node = 0
                edge_src = np.append(edge_src, src_node)
                edge_dst = np.append(edge_dst, mnode)
                edge_length = np.append(edge_length, cutoff - 1e-8)
                edge_cell_shifts = np.vstack(
                    [edge_cell_shifts, np.array([0.0, 0.0, 0.0])]
                )
        return edge_src, edge_dst, edge_length, edge_cell_shifts

    def _check_and_standardize_data(self, data):
        assert (
            "batch" not in data
        ), "Periodic boundary conditions not currently supported on batches."
        assert hasattr(
            data, "cell"
        ), "The data must contain data.cell as a 3x3 matrix to apply periodic boundary conditions."
        assert hasattr(
            data, "pbc"
        ), "The data must contain data.pbc as a bool (True) or list of bools for the dimensions ([True, False, True]) to apply periodic boundary conditions."

        # Ensure data consistency in terms of device and type
        if not isinstance(data.pos, torch.Tensor):
            data.pos = torch.tensor(data.pos)
        if data.pos.dtype not in [torch.float32, torch.float64]:
            data.pos = data.pos.to(torch.get_default_dtype())
        # Canonicalize based off data.pos, similar to PyG's default behavior
        device, dtype = data.pos.device, data.pos.dtype
        if not (
            isinstance(data.cell, torch.Tensor)
            and data.cell.dtype == dtype
            and data.cell.device == device
        ):
            data.cell = torch.tensor(data.cell, dtype=dtype, device=device)
        if not (
            isinstance(data.pbc, torch.Tensor)
            and data.pbc.dtype == torch.bool
            and data.pbc.device == device
        ):
            data.pbc = torch.tensor(data.pbc, dtype=torch.bool, device=device)

        return data, device, dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"


@functional_transform("pbc_distance")
class PBCDistance(Distance):
    """
    A PBC-aware version of the `Distance` transform from
    Pytorch-Geometric that adds the effect of edge_shifts in PBC
    """

    def forward(self, data: Data) -> Data:
        assert data.pos is not None, "'data.pos' is required."
        assert data.edge_index is not None, "'data.edge_index' is required."
        assert hasattr(data, "edge_shifts"), "'data.edge_shifts' is required for PBC."
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        vec = (
            pos[col] - pos[row] + data.edge_shifts
        )  # The key change is adding edge_shifts
        dist = torch.norm(vec, p=2, dim=-1).view(-1, 1)

        if self.norm and dist.numel() > 0:
            max_val = float(dist.max()) if self.max is None else self.max

            length = self.interval[1] - self.interval[0]
            dist = length * (dist / max_val) + self.interval[0]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.to(pseudo.dtype)], dim=-1)
        else:
            data.edge_attr = dist

        return data


@functional_transform("pbc_local_cartesian")
class PBCLocalCartesian(LocalCartesian):
    """
    A PBC-aware version of the `LocalCartesian` transform from
    Pytorch-Geometric that adds the effect of edge_shifts in PBC
    """

    def forward(self, data: Data) -> Data:
        assert data.pos is not None, "'data.pos' is required."
        assert data.edge_index is not None, "'data.edge_index' is required."
        assert hasattr(data, "edge_shifts"), "'data.edge_shifts' is required for PBC."
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = (
            pos[row] - pos[col]
        ) - data.edge_shifts  # The key change is adding edge_shifts
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm:
            max_value = scatter(
                cart.abs(), col, dim=0, dim_size=pos.size(0), reduce="max"
            )
            max_value = max_value.max(dim=-1, keepdim=True)[0]

            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            cart = length * cart / (2 * max_value[col].clamp_min(1e-9)) + center

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data


def pbc_as_tensor(pbc):
    # Function to convert pbc to a tensor of 3 booleans, regardless of input type
    pbc_t = torch.as_tensor(pbc, dtype=torch.bool)
    shape = pbc_t.shape

    if shape == ():  # single bool
        val = pbc_t.item()
        return torch.tensor([val, val, val], dtype=torch.bool)
    elif shape == (1,):  # list or array of one bool
        val = pbc_t.item()
        return torch.tensor([val, val, val], dtype=torch.bool)
    elif shape == (3,):  # list or array of 3 bools
        return pbc_t
    else:
        raise ValueError(f"Invalid PBC shape {shape}. Expect scalar, (1,), or (3,).")


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
