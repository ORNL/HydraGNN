# Optimal power flow examples

OPF configurations use HydraGNN's strict top-level named-variable schema. For
example, heterogeneous bus-solution training declares:

```json
"Variables": {
    "inputs": [
        {"name": "node_features", "level": "node", "dim": 4}
    ],
    "outputs": [
        {"name": "bus_solution", "level": "node", "dim": 2}
    ]
}
```

The raw heterogeneous importer stores domain data under those names:

- every node store has `node_features` with shape `(N_type, dim_type)`;
- the predicted node store has `bus_solution` or `generator_solution` with
  shape `(N_type, output_dim)`;
- graph targets such as `objective` and `feasibility` have shape `(1, dim)`;
- featured edge stores retain relation-specific sources such as
  `ac_line_features` and `transformer_features`, each shaped
  `(num_relation_edges, relation_dim)`;
- graph context is retained as `context` with shape `(1, context_dim)`.

At the preprocessing boundary, `compile_named_hetero_opf_sample` validates
these attributes and derives the tensors expected by PyG and existing HydraGNN
models: each node store's `x`, featured edge stores' `edge_attr`, `graph_attr`,
and the internal `y`/`y_loc` target representation. Named source attributes are
not deleted or overwritten.

Heterogeneous node types can naturally have different input widths. The public
schema describes the predicted node type, while
`NeuralNetwork.Architecture.node_input_dims` records every type-specific width
(`bus`, `generator`, `load`, and `shunt`). `update_config` verifies those widths
against the prepared dataset before model construction.

Prepared pickle, HDF5, or ADIOS datasets must contain the named attributes.
Datasets containing only legacy `x`/`y` targets are rejected with an explicit
request to rerun preprocessing; HydraGNN does not silently reconstruct the
missing named data.
