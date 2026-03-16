# CP2D Companion Code

Companion code for the CP2D dataset release. This folder contains the scripts needed to:

- build PyTorch Geometric graph files from the raw meshes and result CSVs
- train the final CP2D model using the top-level training files
- reuse the fixed train/validation/test split metadata

## Included Files

- `dataset_generation/create_graph.py`: batch or single-file graph generation from Abaqus `.inp` files
- `CP2D.py`: final dataset definition used by training
- `train_cp2d.py`: final training entrypoint
- `cp_helpers.py`: final edge-feature helpers used by `CP2D.py`
- `pna.json`: training configuration used by `train_cp2d.py`
- `splits/split_by_sve_seed0.json`: reproducible split metadata used for dataset partitioning

## Expected Data Layout

These scripts assume the published dataset keeps the original directory structure, for example:

```text
dataset_root/
  inp_files/
    vf10/
    vf45/
    vf90/
  csv_results3/
    vf10/
    vf45/
    vf90/
  graphs/
  processed/
```

This companion repo is designed so that generated graph files live at the repository root:

```text
cp2d/
  dataset_generation/
  splits/
  graphs/
  processed/
```

## Installation

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

`hydragnn` is only needed for `CP2D.py` and `train_cp2d.py`. If you only need raw graph generation, the other dependencies are sufficient.

## Example Workflow

1. Make sure each input deck already has its corresponding grain-info CSV available if you plan to train the final model. `create_graph.py` expects files such as `SVE10_grain_info.csv` when `--grain_info_dir auto` is used.

2. Build graph files from all published `.inp` files:

```bash
python dataset_generation/create_graph.py \
  --inp_root /path/to/dataset_root/inp_files \
  --vf_folders vf10,vf45,vf90 \
  --partition_folders 1x1,2x2,4x4,8x8 \
  --grain_info_dir auto \
  --results_root /path/to/dataset_root/csv_results3 \
  --results_root_extra "" \
  --out_dir /path/to/cp2d/graphs \
  --save_format dict
```

3. Train with the final top-level pipeline:

```bash
python train_cp2d.py \
  --root /path/to/cp2d \
  --graphs_dir /path/to/cp2d/graphs \
  --split_json /path/to/cp2d/splits/split_by_sve_seed0.json
```

## Notes

- `dataset_generation/create_graph.py` writes `y_labels.csv` into the graph output directory by default. `CP2D.py` and `train_cp2d.py` infer that file automatically from `graphs/` unless you override it.
- HydraGNN builds the processed dataset as part of the training workflow through `CP2D.py`, so there is no separate dataset-building step documented here.
- `train_cp2d.py` can either generate a split from `--split_seed` or reuse the bundled published split via `--split_json`.
- The authoritative training files are the top-level files in `publish/cp2d/`, while raw graph generation lives under `dataset_generation/`.
- The split file under `splits/` is included for reproducibility. Keep it versioned together with the exact dataset release it was derived from.
- The scripts assume filenames like `SVE10_8x8_12.inp` and result folders like `csv_results3/vf45/8x8/SVE10/sve10.csv`.
