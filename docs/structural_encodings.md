# Structural encodings for heterogeneous models

HydraGNN separates domain preprocessing from model consumption.

- A domain provider computes tensors and attaches them to prepared graph data.
- `Architecture.structural_encoding` declares how those tensors enter a
  heterogeneous model.
- `StructuralAttentionContext` carries prepared tensors through global
  attention without exposing their domain origin.

The core model never computes or interprets domain quantities.

## Configuration

```json
{
  "target_node_type": "entity",
  "node_inputs": [
    {"attribute": "spectral_vectors", "dim": 8, "random_sign_flip": true},
    {"attribute": "spectral_values", "dim": 8, "broadcast": "graph"}
  ],
  "pairwise": {
    "attribute": "pair_features",
    "path_attribute": "pair_features_path",
    "artifact_key": "pair_features",
    "dim": 2,
    "hidden_dim": 16,
    "zero_diagonal": true
  }
}
```

`node_inputs` are concatenated with the initial embedding of the target node
type and projected back to the hidden width. A graph-broadcast input may have
one row per graph instead of one row per node.

Exactly one attention-side representation can be configured:

- `pairwise`: dense pair features mapped to a learned bias per attention head;
- `factorized_pairwise`: low-rank complex factors used to construct pair
  features inside attention; or
- `qk_coordinates`: structural coordinates appended to Performer queries and
  keys. Its `placement` is `input`, `qk`, or `both`.

Pairwise tensors may be embedded in a single graph or loaded from one cached
artifact path per graph. Cache loading is controlled only by the declarative
attribute names above.

## Domain providers

A provider implements `StructuralEncodingProvider` and attaches the attributes
declared by `structural_encoding`. Providers belong to applications or domain
packages. For example, the OPF provider uses `Architecture.opf_preprocessing`
to compute electrical quantities, while the generic model sees only tensor
attributes, dimensions, and placement.
