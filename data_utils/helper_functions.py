import numpy as np
from tqdm import tqdm
from utils.print_utils import print_distributed, tqdm_verbosity_check

import torch
import torch.distributed as dist


def distance_3D(p1: [float], p2: [float]):
    """Computes the Euclidean distance between two 3D points.

    Parameters
    ----------
    p1: [float]
        List of x,y,z coordinates of the first point.
    p2: [float]
        List of x,y,z coordinates of the second point.
    """

    p1 = np.array([p1[0], p1[1], p1[2]])
    p2 = np.array([p2[0], p2[1], p2[2]])
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    distance = np.sqrt(squared_dist)
    return distance


def order_candidates(
    candidate_neighbours: dict, distance_matrix: [[float]], verbosity: int
):
    """Function orders the possible neighbour candidates of an atom based on their distance from the atom.

    Parameters
    ----------
    candidate_neighbours: dict
        Dictionary of neighbours for each atom. Key-value pair: index_of_atom: [indexes_of_neighbours].
    distance_matrix: [[float]]
        Matrix containing the distances for each pair of atoms within the structure.
    """

    print_distributed(
        verbosity, "Ordering candidate neighbours based on their distance."
    )
    sorted_candidate_neighbours = {}
    for point_index, candidates in (
        tqdm(candidate_neighbours.items())
        if tqdm_verbosity_check(verbosity)
        else candidate_neighbours.items()
    ):
        distances = distance_matrix[point_index, candidates]
        candidate_distance_dict = {
            candidates[i]: distances[i] for i in range(len(candidates))
        }
        candidate_distance_dict = {
            k: v
            for k, v in sorted(
                candidate_distance_dict.items(), key=lambda item: item[1]
            )
        }
        sorted_candidate_neighbours[point_index] = candidate_distance_dict.keys()
    return sorted_candidate_neighbours


def resolve_neighbour_conflicts(
    point_index: int,
    neighbours: [int],
    adjacency_matrix: [[int]],
    max_num_node_neighbours: int,
):
    """Function resolves which neighbours of an atom are in conflict. Specifically it is determining which neighbours are already referencing an atom and
     solving the conflict of already taken neighbours who have number of connections equal to maximum number of neighbours.
    and determinin

    Parameters
    ----------
    point_index: int
        Index of the atom in the structure.
    neighbours: [int]
        List containing indexes of neighbour atoms.
    adjacency_matrix: [[int]]
        Adjacency matrix of the structure. Each entry is either zero or one depending if there is a connection between two atoms or not.
    max_num_node_neighbours: int
        Maximum number of neighbours an atom can have.
    """
    already_neighbours = np.nonzero(adjacency_matrix[:, point_index])[0].tolist()
    if max_num_node_neighbours == len(already_neighbours):
        return already_neighbours

    neighbours = [x for x in neighbours if x not in already_neighbours]
    taken_neighbours = []
    for n in neighbours:
        if len(np.nonzero(adjacency_matrix[:, n])[0]) == max_num_node_neighbours:
            taken_neighbours.append(n)
    neighbours = [x for x in neighbours if x not in taken_neighbours]

    if len(already_neighbours) == 0:
        return neighbours[:max_num_node_neighbours]

    number_of_allowed = max_num_node_neighbours - len(already_neighbours)
    neighbours = neighbours[:number_of_allowed] + already_neighbours

    return neighbours


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))
