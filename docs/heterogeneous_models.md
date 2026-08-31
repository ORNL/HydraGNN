# Heterogeneous graph models

HydraGNN supports graphs whose node and edge types have different feature
spaces. The available heterogeneous stacks are `HeteroGIN`, `HeteroSAGE`,
`HeteroGAT`, `HeteroRGAT`, `HeteroHGT`, `HeteroHEAT`, and `HeteroPNA`.

## Data contract

Models receive a PyG `HeteroData` object. Each node store must contain `x` with
shape `(N_type, input_dim_type)`, and each relation must contain `edge_index`
with shape `(2, E_relation)`. Relations configured with an edge width must also
contain `edge_attr` with shape `(E_relation, edge_dim_relation)`.

Applications should retain semantic source attributes and construct these
internal tensors at their preprocessing boundary. The OPF example uses
`node_features`, `<relation>_features`, and named targets; see
[the OPF workflow](opf_workflow.md).

## Configuration

The principal `NeuralNetwork.Architecture` keys are:

| Key | Meaning |
| --- | --- |
| `mpnn_type` | One of the seven heterogeneous stacks listed above. |
| `node_input_dims` | Mapping from node type to its input width. |
| `edge_dim` | Mapping from relation name to edge-feature width. Relations omitted from the mapping are featureless. |
| `node_target_type` | Node type read by a node-level output head. |
| `hetero_pooling_mode` | How type-specific graph embeddings are combined for graph outputs. |
| `share_relation_weights` | Reuse compatible relation modules instead of maintaining independent weights. |
| `graph_pooling` | Pooling operation within each node type for graph outputs. |
| `use_graph_attr_conditioning` | Enable graph-level conditioning. |
| `graph_attr_conditioning_mode` | `film`, `concat_node`, or `fuse_pool`. |

Architecture-specific keys such as `hetero_attention_heads`,
`hetero_edge_type_emb_dim`, and `hetero_edge_attr_emb_dim` are used by the
attention-based stacks when present in their example configurations.

## Node-level outputs

All node types participate in message passing. A node-level head does not emit
placeholder predictions for unrelated types. It selects
`x_dict[node_target_type]` after message passing and applies the head only to
that tensor. Thus, with `node_target_type: "bus"`, predictions have shape
`(num_bus_nodes_in_batch, output_dim)`.

The target tensor must contain exactly the same selected nodes in the same PyG
batch order. This supports all nodes of one type. An arbitrary labeled subset
within one node type is not currently supported; that would require an explicit
target mask or index.

## Graph-level outputs

Each node type is pooled separately. The type-level graph representations are
then combined according to `hetero_pooling_mode` before the graph head is
applied. Node types therefore contribute to a graph prediction even when their
raw feature widths differ.

## Limitations

- Heterogeneous stacks do not currently support `global_attn_engine`; model
  construction raises an error if global attention is requested.
- `node_input_dims` and relation-specific `edge_dim` must agree with the actual
  `HeteroData` stores.
- A node output targets one complete node type, not an arbitrary subset of it.
- Convolutional node heads do not support multiple dataset branches.

