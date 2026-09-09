# Unit-test data fixtures

HydraGNN's model-quality tests use synthetic PyTorch Geometric samples created
by `tests/deterministic_graph_data.py`. These fixtures are test infrastructure;
they are not an example of implicit production-data preprocessing.

## Generation

Each sample is a small body-centered-cubic graph. A private PyTorch random
generator, seeded with `seed + configuration_start`, selects the cell extent
and categorical node values. A nearest-neighbor calculation provides the local
graph structure and synthetic node targets. The generator writes named source
attributes rather than pre-concatenating HydraGNN's canonical tensors.

Important named attributes include:

- `node_features`: scalar node input;
- `x_target`, `x2`, and `x3`: node targets;
- `sum_x_x2_x3` and related attributes: graph targets;
- `graph_conditioning`: the historical two-component graph input
  `[number of nodes matching the first node value, 1]`;
- `edge_lengths`: scalar nearest-neighbor distances;
- `pe` and `rel_pe`: fixture-owned positional descriptors.

The JSON schema controls which stored attributes become model inputs and
outputs. For example, `edge_lengths` is present on every fixture, but it becomes
`data.edge_attr` only when a test adds this declaration:

```json
{"name": "edge_lengths", "level": "edge", "dim": 1}
```

Likewise, HydraGNN constructs `data.x`, `data.edge_attr`, `data.graph_attr`,
`data.y`, and `data.y_loc` from the declared named attributes before batching.

## Split-local normalization

The fixture reproduces the retired test loader's numerical behavior so that
existing model-quality thresholds remain comparable:

1. Each named node or target attribute is min-max scaled using all samples
   generated in that invocation.
2. `edge_lengths` is divided by the maximum edge length over those samples.
3. Train, validation, and test directories are generated independently, so
   each split has its own scaling statistics.

This normalization is deliberately implemented in the test fixture. The
production named-data loading path does not infer transformations from an
attribute name and does not normalize edge lengths automatically. Users must
perform and record any desired physical-data transformations during their own
preprocessing.

## Cache compatibility

Each generated sample stores `fixture_schema_version`. The cache is reused only
when its sample count, required named attributes, positional descriptor shape,
and fixture version match the current contract. Otherwise, the old `.pt`
samples are removed and regenerated deterministically.

When changing any of the following, increment
`DETERMINISTIC_GRAPH_DATA_VERSION`:

- graph generation or random seeding;
- named attribute values, names, shapes, or semantics;
- connectivity or edge-feature construction;
- normalization;
- positional descriptors.

## Interpreting CI quality failures

Before relaxing a loss, RMSE, or MAE threshold, verify that the failing job and
a passing reference use the same fixture version and compare the prepared
`data.x`, `data.edge_attr`, `data.graph_attr`, and targets. A repeated value
across several tests often indicates a shared configuration or fixture, not
independent model failures. Thresholds should be recalibrated only after an
intentional deterministic fixture change or a demonstrated platform-specific
numerical baseline change.
