# Structural encodings for heterogeneous models

HydraGNN separates domain preprocessing from model consumption.

- A domain provider computes tensors and attaches them to prepared graph data.
- `Architecture.structural_encoding` declares how those tensors enter a
  heterogeneous model.
- `StructuralAttentionContext` carries prepared tensors through global
  attention without exposing their domain origin.

The core model never computes or interprets domain quantities.

## Configuration

This first fragment is the value of `NeuralNetwork.Architecture.structural_encoding`.
For its `pairwise` branch, also set `global_attn_engine: "GPS"`,
`global_attn_type: "multihead"`, and `attn_node_types: ["entity"]`
on the architecture, as in the complete attention fragment below.

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

At most one attention-side representation can be active:

- `pairwise`: dense pair features mapped to a learned bias per attention head;
- `factorized_pairwise`: low-rank complex factors used to construct pair
  features inside attention; or
- `qk_coordinates`: structural coordinates appended to Performer queries and
  keys. Its `placement` is `input`, `qk`, or `both`.

Pairwise tensors may be embedded in a single graph or loaded from one cached
artifact path per graph. Cache loading is controlled only by the declarative
attribute names above.

## Attention configuration examples

Merge one of the following fragments into `NeuralNetwork.Architecture` of an
otherwise complete heterogeneous configuration. The dataset must declare the
`entity` node type and the preprocessing provider must attach the named
structural tensors. These settings consume prepared features; they do not
compute those features.

### Dense pairwise bias

```json
{
  "global_attn_engine": "GPS",
  "global_attn_type": "multihead",
  "global_attn_heads": 4,
  "hidden_dim": 64,
  "attn_node_types": ["entity"],
  "structural_encoding": {
    "target_node_type": "entity",
    "pairwise": {
      "attribute": "pair_features",
      "path_attribute": "pair_features_path",
      "artifact_key": "pair_features",
      "dim": 2,
      "hidden_dim": 16,
      "zero_diagonal": true
    }
  }
}
```

Dense pairwise features require `multihead` attention and the structural target
as the sole attention node type. See the complete
[effective-resistance configuration](../examples/opf/configs/opf_heterosage_case4661_effective_resistance_rpe.json)
for an OPF provider and matching data schema.

### Factorized pairwise features

```json
{
  "global_attn_engine": "GPS",
  "global_attn_type": "multihead",
  "global_attn_heads": 4,
  "hidden_dim": 64,
  "attn_node_types": ["entity"],
  "structural_encoding": {
    "target_node_type": "entity",
    "factorized_pairwise": {
      "dim": 8,
      "attributes": {
        "u_real": "svd_u_real",
        "u_imag": "svd_u_imag",
        "v_real": "svd_v_real",
        "v_imag": "svd_v_imag",
        "s": "svd_s"
      }
    }
  }
}
```

This uses the same attention setup as the complete
[SVD example](../examples/opf/configs/opf_heterosage_case118_svd_rpe.json).
The attribute mapping names the prepared real/imaginary factors and singular
values; `dim` is the retained factor width.

### Performer structural coordinates

```json
{
  "global_attn_engine": "GPS",
  "global_attn_type": "performer",
  "global_attn_heads": 4,
  "hidden_dim": 64,
  "attn_node_types": ["entity"],
  "structural_encoding": {
    "target_node_type": "entity",
    "qk_coordinates": {
      "attribute": "structural_coordinates",
      "dim": 8,
      "placement": "qk",
      "coefficient_init": 0.0
    }
  }
}
```

Placements `qk` and `both` require Performer and the structural target as the
sole attention node type. Placement `input` only augments node inputs and does
not itself require Performer or reserve the attention-side representation.
Ordinary `node_inputs` may also be used without any attention-side encoding.
See the complete [Q/K example](../examples/opf/configs/opf_heterosage_case4661_effective_resistance_qk.json).

## Domain providers

A provider implements `StructuralEncodingProvider` and attaches the attributes
declared by `structural_encoding`. Providers belong to applications or domain
packages. For example, the OPF provider uses `Architecture.opf_preprocessing`
to compute electrical quantities, while the generic model sees only tensor
attributes, dimensions, and placement.
