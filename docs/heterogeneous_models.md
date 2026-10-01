# Heterogeneous models and GPS

Use `Variables.graph_type: "heterogeneous"` with PyG `HeteroData` samples.
Declare all node types in `Variables.node_types`, a `node_type` on each
node-level variable, and every complete edge triple in
`NeuralNetwork.Architecture.edge_types`. See the
[Variables reference](../USER_MANUAL.md#variables) for tensor shapes and
featureless relations. Relation dimensions describe the data and must not
change when switching models.

## Model selection

| `mpnn_type` | Local message passing | Consumes edge attributes |
| --- | --- | --- |
| `HeteroGIN` | Per-relation GIN | No |
| `HeteroSAGE` | Per-relation GraphSAGE | No |
| `HeteroGAT` | Per-relation graph attention | Yes |
| `HeteroPNA` | Per-relation PNA | Yes |
| `HeteroRGAT` | Relation-specific attention | Yes |
| `HeteroHGT` | Heterogeneous graph Transformer | No |
| `HeteroHEAT` | Heterogeneous edge-enhanced attention | Yes |

Edge-unaware models still use every declared relation's topology. HeteroPNA
requires degree statistics (`pna_deg`). `hetero_attention_heads` controls local
attention in the attention-based stacks; `global_attn_heads` independently
controls GPS. For HGT, choose a hidden width divisible by its local head count.
HGT does not support convolutional node heads; use `mlp` or `mlp_per_node`.
`hetero_pooling_mode` is `sum` (default) or `mean` and combines the already
pooled representations of the node types for graph predictions.

Start with the complete [HGT graph-output configuration](../examples/opf/opf_heterogeneous_hgt.json)
or [HeteroSAGE solution configuration](../examples/opf/opf_solution_heterogeneous.json).
The [OPF training entry point](../examples/opf/train_opf_solution_heterogeneous.py)
shows dataset preparation, degree statistics, configuration updates, and model
construction with `metadata=trainset[0].metadata()` and per-type
`node_input_dims`. Supply this information before optimizer construction so
model parameters are initialized in time.

## Heterogeneous GPS

For example, merge these fields into `NeuralNetwork.Architecture` of an OPF
heterogeneous configuration (they are not a standalone dataset configuration):

```json
{
  "mpnn_type": "HeteroSAGE",
  "hidden_dim": 64,
  "global_attn_engine": "GPS",
  "global_attn_type": "multihead",
  "global_attn_heads": 4,
  "attn_node_types": ["bus"],
  "attn_only": false
}
```

`attn_node_types: null` selects all node types; an explicit list selects only
those types for global attention. `attn_only: false` combines local message
passing with global attention. Set it to `true` for attention-only layers;
`global_attn_engine: "GPS-attn-only"` is also recognized as an alias.
GPS supports `multihead` and `performer`. Select a hidden width divisible by
the global head count. This heterogeneous path supports GPS, not the
geometric `EquivariantTransformer` engine.

Complete examples compare [all-type attention](../examples/opf/configs/opf_heterosage_case118_gps_all_attention.json),
[bus-only attention](../examples/opf/configs/opf_heterosage_case118_gps_bus_attention.json),
and [no attention](../examples/opf/configs/opf_heterosage_case118_no_attention.json).
For structural attention biases and coordinates, see
[structural encodings](structural_encodings.md), including its single-type
attention restrictions. For physics constraints, use the
[OPF loss workflow](../examples/opf/README_augmented_lagrangian.md).
