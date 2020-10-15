import numpy as np


def distance_3D(p1: [float], p2: [float]):
    p1 = np.array([p1[0], p1[1], p1[2]])
    p2 = np.array([p2[0], p2[1], p2[2]])

    squared_dist = np.sum((p1-p2)**2, axis=0)
    distance = np.sqrt(squared_dist)
    return distance


def order_candidates(candidate_neighbours, distance_matrix):
    sorted_candidate_neighbours = {}
    for point_index, candidates in candidate_neighbours.items():
        distances = distance_matrix[point_index, candidates]
        candidate_distance_dict = {candidates[i]: distances[i] for i in range(len(candidates))}
        candidate_distance_dict = {k: v for k, v in sorted(candidate_distance_dict.items(), key=lambda item: item[1])}
        sorted_candidate_neighbours[point_index] = candidate_distance_dict.keys()
    return sorted_candidate_neighbours

def remove_collinear_candidates(candidate_neighbours, distance_matrix):
    collinear_neigbours = {}
    for point_index, candidates in candidate_neighbours.items():
        candidates = list(candidates)
        collinear_points = []
        for candidate1 in range(len(candidates)):
            for candidate2 in range(candidate1 + 1, len(candidates)):
                AC = distance_matrix[point_index, candidates[candidate2]]
                AB = distance_matrix[point_index, candidates[candidate1]]
                BC = distance_matrix[candidates[candidate1], candidates[candidate2]]
                if  abs(AC- (AB + BC)) < 10**(-3):
                    collinear_points.append(candidates[candidate2])
        collinear_neigbours[point_index] = collinear_points
    return collinear_neigbours