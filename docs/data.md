# Data Formats

Inside HydraGNN, we use [torch_geometric.Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) to store all graph properties.

Node, edge, and graph-level attributes are divided into different
variables as follows,

    Data.x: node-level with shape [num_nodes, num_node_features]
    Data.y: graph-level with shape [num_graph_features)
    Data.edge_attr: edge-level with shape [num_edges, num_edge_features]

We also make use of coordinates and edges,

    Data.pos: node-level coordinates with shape [num_nodes, num_dimensions]
    Data.edge_index: node indices for each edge with shape [2, num_edges]

## Adios Storage Format

The Adios data file maintains these names and shapes,
but concatenates all data over all graphs.
Adios data fields `Z.variable_count` and `Z.variable_offset`
(where `Z` = `x`, `pos`, `y`, or `edge_addr`) allow parsing out
the data pertaining to a specific graph
by offset and count.  For example (using 0-based indexing),
the starting node index for graph 0 is,

    dataset.x.variable_offset[0]

while the number of atoms in graph 0 is,

    dataset.x.variable_count[0]

To retrieve edge features for graph 20, we would use,

    off = dataset.edge_attr.variable_offset[20]
    count = dataset.edge_attr.variable_count[20]
    dataset.edge_attr[off:off+count]

Note: Since these offsets and counts index graphs,
it would be more descriptive to have called these
`graph_offset` and `graph_count`.

### Metadata

In order to describe names and dimensions of
individual features, the Adios datafile contains
supplementary fields `x_name`, `y_name`, and `edge_attr_name`.
These also have `feature_count` and `feature_offset`,
which are indexes into the feature dimensions of `x`, `y`, and `edge_attr`,
respectively.

So, for example, the *qm7x* dataset has:

    x_name = ["atomic_number", "pos", "forces, "charge", "dipole", "volume_ratio"]

    x_name.feature_count  = [1, 3, 3, 1, 3,  1]
    x_name.feature_offset = [0, 1, 4, 7, 8, 11]

## Selecting Features in the Model Configuration File

When specifying a model, we provide a list of input
and output node, edge, and graph-level features:

    "inputs": {
        "node": ["atomic_number", "pos"],
        "edge": ["length"],
        "graph": []
    },

    "outputs": [
        { "type": "graph",
          "layer_dims": [100, 100],
          "features": ["energy"],
          "loss": "mse"
        },
        { "type": "node",
          "layer_dims": [128],
          "features": ["charge", "volume_ratio"],
          "loss": "mse"
        }
    ]

    "loss": {
        "type": "sum",
        "weights": [0.001, 1.0]
    }

Note that each output head forms its own list element
and has its own loss function.  The combined, total loss,
is specified by the "loss" element.
