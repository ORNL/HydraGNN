from torch_geometric.utils.convert import to_networkx
import networkx as nx
import os
from torch_geometric.data import Data
import torch
from torch import tensor
import numpy as np
import pickle
import pathlib
from dataset_descriptors import AtomFeatures, StructureFeatures
from utils import distance_3D, remove_collinear_candidates, order_candidates

# from serialized_dataset_loader import SerializedDataLoader
# from raw_dataset_loader import RawDataLoader
import matplotlib.pyplot as plt
import igraph as ig
import chart_studio.plotly as py
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D


# testing raw dataset loader
cu = "CuAu_32atoms"
fe = "FePt_32atoms"

files_dir = "./Dataset/" + fe + "/output_files/"
loader = RawDataLoader()
loader.load_raw_data(dataset_path=files_dir)

# testing serialized dataset loader

cu = "CuAu_32atoms.pkl"
fe = "FePt_32atoms.pkl"
files_dir = "./SerializedDataset/" + fe

atom_features = [
    AtomFeatures.NUM_OF_PROTONS,
    AtomFeatures.CHARGE_DENSITY,
    AtomFeatures.MAGNETIC_MOMENT,
]
structure_features = [StructureFeatures.FREE_ENERGY]
radius = 7
max_num_node_neighbours = 10

loader = SerializedDataLoader()
dataset = loader.load_serialized_data(
    dataset_path=files_dir,
    atom_features=atom_features,
    structure_features=structure_features,
    radius=radius,
    max_num_node_neighbours=max_num_node_neighbours,
)

structure = dataset[0]
vertices = structure.pos.numpy().tolist()
edges = structure.edge_index.t().numpy().tolist()
print(torch.tensor(edges))
N = StructureFeatures.SIZE.value


# Showing the points in 3d coordinate space with axes.
labels = [i for i in range(N)]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
Xn = [vertices[i][0] for i in range(N)]
Yn = [vertices[i][1] for i in range(N)]
Zn = [vertices[i][2] for i in range(N)]
ax.scatter(Xn, Yn, Zn)

for i, v in enumerate(vertices):
    ax.text(v[0], v[1], v[2], labels[i])
plt.show()


# Showing the connection graph. This is not equal to a graph in 3D coordinate system.
G = ig.Graph()
G.add_vertices(vertices)
G.add_edges(edges)

layt = G.layout("kk_3d", dim=3)
Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
Yn = [layt[k][1] for k in range(N)]  # y-coordinates
Zn = [layt[k][2] for k in range(N)]  # z-coordinates

Xe = []
Ye = []
Ze = []

for e in edges:
    Xe += [layt[e[0]][0], layt[e[1]][0], None]  # x-coordinates of edge ends
    Ye += [layt[e[0]][1], layt[e[1]][1], None]
    Ze += [layt[e[0]][2], layt[e[1]][2], None]


types = {26: "Fe", 29: "Cu", 78: "Pt", 79: "Au"}

group = structure.x[:, 0].numpy().tolist()
atom_label = [types[int(i)] for i in group]

labels = []
for i in range(N):
    labels.append("Node=" + str(i) + ", AtomType=" + atom_label[i])


trace1 = go.Scatter3d(
    x=Xe,
    y=Ye,
    z=Ze,
    mode="lines",
    line=dict(color="rgb(125,125,125)", width=1),
    hoverinfo="none",
)
trace2 = go.Scatter3d(
    x=Xn,
    y=Yn,
    z=Zn,
    mode="markers",
    name="Nodes",
    marker=dict(
        symbol="circle",
        size=6,
        color=group,
        colorscale="Viridis",
        line=dict(color="rgb(50,50,50)", width=0.5),
    ),
    text=labels,
    hoverinfo="text",
)
axis = dict(
    showbackground=False,
    showline=False,
    zeroline=False,
    showgrid=False,
    showticklabels=False,
    title="",
)
layout = go.Layout(
    title="Structure (3D visualization), radius = "
    + str(radius)
    + ", maximum number of neighbours = "
    + str(max_num_node_neighbours),
    width=1000,
    height=1000,
    showlegend=False,
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis),
    ),
    margin=dict(t=100),
    hovermode="closest",
    annotations=[
        dict(
            showarrow=False,
            x=0,
            y=0.1,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=14),
        )
    ],
)

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)

fig.show()
"""
# This section should be deleted in the future. Was previously content of scrabble.py.
# Tensor slicing

x = torch.randn(5, 3, dtype=torch.double)

print(x.size())

print(x)

z = x[:, 1]

y = x[:, 2]

print(z)

print(y)

w = z - y

print(w)

a = np.arange(5)
b = np.arange(5)
print(np.divide(a, b))

# Trying it out with dataset descriptors

x = torch.randn(5, 3, dtype=torch.double)

a_f_i = [AtomFeatures.NUM_OF_PROTONS.value, AtomFeatures.MAGNETIC_MOMENT.value]

print(a_f_i)

print(x)
print(x[:, a_f_i])

t = x[:, a_f_i]
print(t.shape)
y = torch.randn(10, 1, dtype=torch.double)
print(torch.reshape(y, t.shape))

# Computing adjacency matrix.
# Stopping criterion: maximum distance from our node(radius of a sphere) and maximum number of neighbours
import torch
from dataset_descriptors import AtomFeatures, StructureFeatures
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
import pickle
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


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

points2 = [[0, 0, 0], [1,2,1], [3,3,3], [5,3,5],[2,4,2]]
points = points2
radius = 10
size = 5
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


adjacency_matrix = torch.tensor(np.nonzero(neighbour_matrix))
print(adjacency_matrix)

data = Data()
dataset_path = "./SerializedDataset/FePt_32atoms.pkl"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)[0]

data.edge_index = adjacency_matrix
data.z = torch.tensor([26, 78])


a = np.zeros((3, 2))
a[1, 0] = 1
a[2, 0] = 1

n = np.nonzero(a[:, 0])
print(n[0])

a = [1, 2, 3, 4, 5]
b = [6, 7, 8]
a = a[: len(b)] + b
print(a)
"""
