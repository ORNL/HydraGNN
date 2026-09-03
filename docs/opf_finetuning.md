# OPF fine-tuning

The OPF fine-tuning examples compare transfer learning with training from
scratch for heterogeneous OPF models.

## Scenarios

- `FT1_feasibility_classification`: binary graph-level feasibility prediction;
- `FT1_topology`: adaptation to a held-out network topology;
- `FT2_operating_condition`: adaptation to a shifted load/generation regime;
- `FT3_contingency`: adaptation to N-1 topology perturbations;
- `FT4_task_specific`: transfer from bus-voltage prediction to generator
  dispatch prediction.

## Regimes

`--finetune_regime` accepts:

- `full`: train every parameter;
- `partial`: train the last convolution stage and prediction head;
- `head_only`: freeze the encoder and train only the head.

`--no_pretrained` provides the from-scratch baseline and implies `full`.
`--max_train_samples` changes only the training split for data-efficiency
studies; validation and test remain unchanged.

## Required artifacts

First preprocess the scenario's named dataset. Fine-tuning defaults to HDF5 and
loads the name stored in `ft_data_modelname`. This key is functional: it selects
`<data_root>/<ft_data_modelname>.h5`.

Pretrained checkpoint binaries are not stored in Git. Arrange them as described
in `examples/opf/pretrained_models/README.md`, then select one with
`--pretrained_model_name HeteroSAGE_best` or `HeteroHEAT_best`.

Example:

```bash
python examples/opf/finetune/train_opf_finetune.py \
  --inputfile FT1_topology/config_HeteroSAGE_full.json \
  --data_root ../dataset \
  --pretrained_model_dir ../pretrained_models \
  --pretrained_model_name HeteroSAGE_best \
  --finetune_regime full --hdf5
```

Feasibility classification uses `train_opf_ft1_classify.py`. Its mixed dataset
is created by `generate_infeasible_samples.py`; the actual overload factor is
the script's `--overload_factor` argument. `_ft_overload_factor` in generated
JSON is experiment metadata and does not override that CLI argument.

Configuration fields beginning with `_ft_` describe how generated experiment
configs were produced. `_ft_strategy` is also copied into run metadata.
`_ft_description` is informational. `ft_data_modelname` is the dataset lookup
key and is therefore not merely metadata.

Use `--resume_if_exists` where supported to continue from a run checkpoint.
The submission and sweep scripts under `examples/opf/finetune` orchestrate
Frontier campaigns; their defaults should be reviewed for the target facility
before submission.

## HPO

`examples/opf/opf_deephyper_hpo.py` launches distributed DeepHyper trials and
minimizes the best validation loss observed across completed epochs. It exposes
the candidate heterogeneous MPNNs, evaluation count, hidden-dimension range,
convolution-layer range, and learning-rate range as CLI options. The supplied
Slurm scripts define concurrency and GPU allocation through environment
variables. HPO requires an already prepared dataset matching the training
configuration.

