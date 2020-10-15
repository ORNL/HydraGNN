"""
# Tensor slicing

x = torch.randn(5, 3, dtype=torch.double)

print(x.size())

print(x)

z = x[:, [0,2]]

print(z)

print(x.size())
"""
'''
# Trying it out with dataset descriptors

x = torch.randn(5, 3, dtype=torch.double)

a_f_e = [AtomFeatures.NUM_OF_PROTONS, AtomFeatures.MAGNETIC_MOMENT]

a_f_i = [i.value for i in a_f_e]

print(a_f_i)

print(x)
print(x[:, a_f_i])
'''

# Computing adjacency matrix.
# Stopping criterion: maximum distance from our node(radius of a sphere) and maximum number of neighbours
import torch
from dataset_descriptors import AtomFeatures, StructureFeatures
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
import pickle

# Calculate the distance between points
def distance_between_points(p1, p2):
    p1 = np.array([p1[0], p1[1], p1[2]])
    p2 = np.array([p2[0], p2[1], p2[2]])

    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


def remove_points_on_the_same_line(candidate_neighbours, distance_matrix):
    collinear_neigbours = {}
    for point_index, candidates in candidate_neighbours.items():
        candidates = list(candidates)
        collinear_points = []
        for n1_index in range(len(candidates)):
            for n2_index in range(n1_index + 1, len(candidates)):
                AC = distance_matrix[point_index, candidates[n2_index]]
                AB = distance_matrix[point_index, candidates[n1_index]]
                BC = distance_matrix[candidates[n1_index], candidates[n2_index]]
                if  abs(AC- (AB + BC)) < 10**(-3):
                    collinear_points.append(candidates[n2_index])
        collinear_neigbours[point_index] = collinear_points
    return collinear_neigbours


def order_candidates(candidate_neighbours, distance_matrix):
    sorted_candidate_neighbours = {}
    for point_index, candidates in candidate_neighbours.items():
        distances = distance_matrix[point_index, candidates]
        candidate_distance_dict = {candidates[i]: distances[i] for i in range(len(candidates))}
        candidate_distance_dict = {k: v for k, v in sorted(candidate_distance_dict.items(), key=lambda item: item[1])}
        sorted_candidate_neighbours[point_index] = candidate_distance_dict.keys()
    return sorted_candidate_neighbours


points = [[ 0.0000,  0.0000,  0.0000],
        [ 0.5,  0.5,  0.5],
        [ 1,  1,  1],
        [ 7.8270,  2.6090,  3.5290],
        [ 0.0000,  5.2180,  0.0000],
        [ 2.6090,  7.8270,  3.5290],
        [ 5.2180,  5.2180,  0.0000],
        [ 7.8270,  7.8270,  3.5290],
        [ 0.0000,  0.0000,  7.0580],
        [ 2.6090,  2.6090, 10.5870],
        [ 5.2180,  0.0000,  7.0580],
        [ 7.8270,  2.6090, 10.5870],
        [ 0.0000,  5.2180,  7.0580],
        [ 2.6090,  7.8270, 10.5870],
        [ 5.2180,  5.2180,  7.0580],
        [ 7.8270,  7.8270, 10.5870],
        [10.4360,  0.0000,  0.0000],
        [13.0450,  2.6090,  3.5290],
        [15.6540,  0.0000,  0.0000],
        [18.2630,  2.6090,  3.5290],
        [10.4360,  5.2180,  0.0000],
        [13.0450,  7.8270,  3.5290],
        [15.6540,  5.2180,  0.0000],
        [18.2630,  7.8270,  3.5290],
        [10.4360,  0.0000,  7.0580],
        [13.0450,  2.6090, 10.5870],
        [15.6540,  0.0000,  7.0580],
        [18.2630,  2.6090, 10.5870],
        [10.4360,  5.2180,  7.0580],
        [13.0450,  7.8270, 10.5870],
        [15.6540,  5.2180,  7.0580],
        [18.2630,  7.8270, 10.5870]]

radius = 10
size = 32
max_num = 5
distance_matrix = np.zeros((size, size))
candidate_neighbours = {k: [] for k in range(size)}

#Calculate distances between points and possible candidates for neighbours
for i in range(size):
    for j in range(size):
        distance = distance_between_points(points[i], points[j])
        distance_matrix[i, j] = distance
        if distance_matrix[i, j] <= radius and i!=j:
            candidate_neighbours[i].append(j)

candidate_neighbours = order_candidates(candidate_neighbours, distance_matrix)
collinear_neighbours = remove_points_on_the_same_line(candidate_neighbours, distance_matrix)

neighbour_matrix = np.zeros((size, size))
for point, neighbours in candidate_neighbours.items():
    neighbours = list(neighbours)
    if point in collinear_neighbours.keys():
        collinear_points = list(collinear_neighbours[point])
        neighbours = [x for x in neighbours if x not in collinear_points]
    if len(neighbours)>max_num:
        neighbours = neighbours[:max_num]
    neighbour_matrix[point, neighbours] = 1

print(neighbour_matrix[1])

data = Data()
dataset_path = "./SerializedDataset/FePt_32atoms.pkl"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

print(data)
