import pdb
import torch
from torch_geometric.utils import to_networkx
import igraph as ig
import numpy as np
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler


def compute_topo_features(data: Data):
    """
    Node Features
        1. degree
        2. closeness
        3. betweenness
        4. eigenvector
        5. pagerank
        6. clustering coefficient
        7. k-core
        8. harmonic
        9. eccentricity
    Edge Features
        1. betweeneness
        2. jaccard
        3. adamic-adar
        4. preferential attachment
    """
    edge_index = data.edge_index.numpy()

    G = to_networkx(data, to_undirected=True)
    g = ig.Graph.from_networkx(G)

    # ----- Node Features -----
    node_features = []

    degree_arr = np.array(g.degree())
    node_features.append(degree_arr)
    closeness = np.array(g.closeness())
    closeness = np.nan_to_num(closeness, nan=0.0)
    node_features.append(closeness)
    node_features.append(np.array(g.betweenness()))
    node_features.append(np.array(g.eigenvector_centrality()))
    node_features.append(np.array(g.pagerank()))
    node_features.append(np.array(g.transitivity_local_undirected(mode="zero")))
    node_features.append(np.array(g.coreness()))
    node_features.append(np.array(g.harmonic_centrality()))
    node_features.append(np.array(g.eccentricity()))

    # Stack & normalize node features
    X_nodes = np.vstack(node_features).T  # shape: (num_nodes, num_features)
    # X_nodes = np.concatenate((data.pe,torch.Tensor(X_nodes)),axis=-1)#X_nodes
    X_nodes = StandardScaler().fit_transform(X_nodes)
    data.pe = torch.Tensor(X_nodes)
    lpe_nodes = data.lpe.cpu().numpy()
    lpe_nodes = StandardScaler().fit_transform(lpe_nodes)
    data.lpe = torch.Tensor(lpe_nodes)
    # pdb.set_trace()

    # ----- Edge Features -----
    edge_list = [tuple(e) for e in edge_index.T]
    E = len(edge_list)
    X_edges = np.zeros((E, 4))

    edge_btw = g.edge_betweenness()
    for i, val in enumerate(edge_btw):
        X_edges[i, 0] = val

    for i, (u, v) in enumerate(edge_list):
        nu = set(g.neighbors(u))
        nv = set(g.neighbors(v))
        intersection = nu & nv
        union = nu | nv
        jaccard = len(intersection) / len(union) if union else 0
        aa = sum(1 / np.log(g.degree(w)) for w in intersection if g.degree(w) > 1)
        pa = g.degree(u) * g.degree(v)
        X_edges[i, 1] = jaccard
        X_edges[i, 2] = aa
        X_edges[i, 3] = pa

    X_edges = StandardScaler().fit_transform(X_edges)

    data.rel_pe = torch.Tensor(X_edges)

    return data
