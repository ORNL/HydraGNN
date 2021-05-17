from torch_geometric.utils.convert import to_networkx
import networkx as nx
import os
from torch_geometric.data import Data
import torch
from torch import tensor
import numpy as np
import json
import pickle
import pathlib
from dataset_descriptors import AtomFeatures, StructureFeatures
from helper_functions import distance_3D, remove_collinear_candidates, order_candidates

from serialized_dataset_loader import SerializedDataLoader
from raw_dataset_loader import RawDataLoader
import matplotlib.pyplot as plt
import igraph as ig
import chart_studio.plotly as py
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

cu = "CuAu.pkl"
fe = "FePt.pkl"
si = "FeSi.pkl"
files_dir = "./serialized_dataset/" + fe

os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

config = {}
with open("./examples/configuration.json", "r") as f:
    config = json.load(f)

loader = SerializedDataLoader()

dataset = loader.load_serialized_data(dataset_path=files_dir, config=config)
breakpoint()
structure = dataset[0]
vertices = structure.pos.numpy().tolist()
edges = structure.edge_index.t().numpy().tolist()
print(torch.tensor(edges))
N = num_of_atoms = len(structure.x)


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
    title=f"Structure (3D visualization), radius = {str(config['radius'])},"
    + f" maximum number of neighbours = {str(config['max_num_node_neighbours'])}",
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
