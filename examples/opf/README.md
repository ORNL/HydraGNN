# Optimal power flow examples

This directory provides heterogeneous and homogeneous optimal-power-flow
workflows for preprocessing, training, inference, physics-informed losses,
fine-tuning, and distributed hyperparameter optimization.

Start with the focused documentation:

- [OPF workflow](../../docs/opf_workflow.md)
- [Heterogeneous models](../../docs/heterogeneous_models.md)
- [OPF physics losses](../../docs/opf_physics_losses.md)
- [OPF fine-tuning and HPO](../../docs/opf_finetuning.md)

The principal entry points are:

| Task | Entry point |
| --- | --- |
| Node-level preprocessing/training | `train_opf_solution_heterogeneous.py` |
| Graph-level preprocessing/training | `train_opf_graph_output_heterogeneous.py` |
| Inference | `infer_opf_solution_heterogeneous.py` |
| Transfer learning | `finetune/train_opf_finetune.py` |
| Feasibility classification | `finetune/train_opf_ft1_classify.py` |
| Distributed HPO | `opf_deephyper_hpo.py` |

Run each script with `--help` for its complete CLI. A minimal two-stage HDF5
workflow is:

```bash
python examples/opf/train_opf_solution_heterogeneous.py \
  --preonly --hdf5 --case_name pglib_opf_case14_ieee --num_groups 1

python examples/opf/train_opf_solution_heterogeneous.py \
  --hdf5 --case_name pglib_opf_case14_ieee --num_groups 1
```

For distributed facility runs, use the supplied job scripts only after checking
the allocation, paths, modules, node count, and storage location for the target
system.

## Named-data contract

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
