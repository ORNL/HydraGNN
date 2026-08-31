# OPF data and training workflow

The OPF example represents buses, generators, loads, and shunts as distinct
node types and power-system connections as typed relations. This page describes
the supported path from raw JSON to training and inference.

## Named source data

The raw importer creates semantic attributes:

- `<node_type>.node_features`: `(N_type, dim_type)`;
- `bus.bus_solution` or `generator.generator_solution`:
  `(N_target_type, output_dim)`;
- `<relation>.<relation>_features`: `(E_relation, relation_dim)`;
- `context`: `(1, context_dim)`;
- graph targets such as `objective` or `feasibility`: `(1, target_dim)`.

`compile_named_hetero_opf_sample` validates those attributes and constructs the
internal `x`, `edge_attr`, `graph_attr`, `y`, and `y_loc` tensors. Named source
attributes remain available. Prepared caches containing only legacy `x` or `y`
are rejected; the raw data must be preprocessed again.

The JSON schema declares the predicted node type's public width, while
`Architecture.node_input_dims` records the widths of every heterogeneous node
type. `Architecture.edge_dim` maps relation names to their widths.

## Preprocessing

From the repository root, a minimal bus-target preprocessing run is:

```bash
python examples/opf/train_opf_solution_heterogeneous.py \
  --inputfile opf_solution_heterogeneous.json \
  --data_root dataset \
  --case_name pglib_opf_case14_ieee \
  --num_groups 1 \
  --node_target_type bus \
  --preonly --hdf5
```

Choose exactly one storage flag:

- `--hdf5`: streamed per-rank storage, recommended for large preprocessing;
- `--adios`: distributed ADIOS storage;
- `--pickle`: simple local serialization and the default.

Useful controls include `--case_name ...` or `all`, `--num_groups ...` or
`all`, `--max_samples`, `--topological_perturbations`, and `--nvme`.
Preprocessing partitions work across ranks, streams samples rather than holding
the full corpus in memory, reports and skips malformed raw JSON records, and can
resume appending to an existing HDF5 case dataset. Run preprocessing with rank
zero or the documented MPI job scripts; all ranks synchronize before training
opens the prepared data.

## Training

Reuse the same entry point without `--preonly` and with the matching storage
flag:

```bash
python examples/opf/train_opf_solution_heterogeneous.py \
  --inputfile opf_solution_heterogeneous.json \
  --data_root dataset \
  --case_name pglib_opf_case14_ieee \
  --num_groups 1 \
  --node_target_type bus \
  --hdf5
```

CLI values for `--mpnn_type`, `--hidden_dim`, `--num_conv_layers`,
`--learning_rate`, `--batch_size`, and `--num_epoch` override JSON values.
The selected `node_target_type` must agree with the named output:
`bus_solution` for `bus`, or `generator_solution` for `generator`.

Graph-output examples use `train_opf_graph_output_heterogeneous.py` and named
graph targets such as `objective`. Homogeneous counterparts are retained for
controlled comparisons.

## Inference

`infer_opf_solution_heterogeneous.py` loads the matching prepared dataset and
checkpoint. Keep the model configuration, target type, node/edge dimensions,
and storage format identical to training. Checkpoint binaries are external
artifacts and are intentionally excluded from Git.

## Cache compatibility

Prepared pickle, HDF5, and ADIOS datasets are authoritative. Loading does not
recreate missing semantic attributes or silently reinterpret an old layout.
Rebuild the cache after changing variable names, dimensions, target type,
relation features, topology preprocessing, or other data semantics.

See also [heterogeneous models](heterogeneous_models.md),
[OPF physics losses](opf_physics_losses.md), and
[OPF fine-tuning](opf_finetuning.md).

