# OPFLearn Pipeline

Robust Python workflow for downloading, inspecting, preprocessing, validating, and storing OPFLearnData for AC-OPF machine learning.

## Official Sources

- Dataset catalog: https://data.nlr.gov/submissions/177
- Dataset DOI: https://doi.org/10.7799/1827404
- Archived generator repository: https://github.com/NatLabRockies/OPFLearn.jl
- Redirecting legacy repository: https://github.com/NREL/OPFLearn.jl

## Dataset Contents

The published OPFLearnData release includes AC-OPF samples generated with OPFLearn.jl + PowerModels.jl for:

- `case5_pjm`
- `case14_ieee`
- `case30_ieee`
- `case57_ieee`
- `case118_ieee`

Each feasible dataset contains 10,000 samples.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Download Commands

Archive + extraction:

```bash
python download_opflearn.py --archive --extract
```

Individual case CSVs:

```bash
python download_opflearn.py --cases case14_ieee case118_ieee
```

Equivalent raw download endpoints used by this pipeline:

- `https://data.nlr.gov/system/files/177/OPFLearn_Datasets.zip`
- `https://data.nlr.gov/system/files/177/pglib_opf_case5_pjm.csv`
- `https://data.nlr.gov/system/files/177/pglib_opf_case14_ieee.csv`
- `https://data.nlr.gov/system/files/177/pglib_opf_case30_ieee.csv`
- `https://data.nlr.gov/system/files/177/pglib_opf_case57_ieee.csv`
- `https://data.nlr.gov/system/files/177/pglib_opf_case118_ieee.csv`

## Schema Groups

Columns are discovered dynamically by suffix (no hard-coded per-network column counts).

Input group:

- `*:pl`
- `*:ql`

Primal output group:

- `*:pg`
- `*:qg`
- `*:vm_gen`
- `*:v_bus` (raw source field, corrected during preprocessing)
- `*:p_to`, `*:p_fr`, `*:q_to`, `*:q_fr`

Dual output group:

- `*:v_min`, `*:v_max`
- `*:pg_min`, `*:pg_max`
- `*:qg_min`, `*:qg_max`
- branch bound duals such as `*:p_to_max`, `*:q_to_max`, `*:p_fr_max`, `*:q_fr_max`

## Critical `v_bus` Warning

Per the official OPFLearn repository note, released `:v_bus` angles are encoded incorrectly and must be corrected before training/analysis.

Raw value:

- `z_raw = v_bus`

Correction used by this project (exact required transformation):

- `theta_corrected_rad = -np.angle(z_raw) * 180.0 / np.pi`
- `z_corrected = np.abs(z_raw) * np.exp(1j * theta_corrected_rad)`

Equivalent Julia code from OPFLearn.jl:

```julia
function correct_angle_error(v_bus)
    return abs(v_bus) * exp(1im * -rad2deg(angle(v_bus)))
end
```

## Python Implementation

Core modules:

- `utils/parsing.py`: robust complex parser for `v_bus` strings
- `utils/voltage_correction.py`: correction equations and wrapped-angle difference
- `utils/validation.py`: required numerical validations
- `utils/preprocessing.py`: chunked CSV -> corrected Parquet conversion
- `utils/inspect.py`: schema/stats inspection report
- `utils/pyg_serialization.py`: Parquet -> PyG objects -> HydraGNN HDF5 serialization
- `utils/download.py`: redirect-aware streaming downloader with retries

## Preprocessing Behavior

For each `busK:v_bus` column:

- preserves raw string as `busK:v_bus_raw`
- adds `busK:vm_bus`
- adds `busK:va_bus_rad`
- adds `busK:va_bus_deg`
- adds `busK:v_bus_real_corrected`
- adds `busK:v_bus_imag_corrected`
- drops original `busK:v_bus`

Raw and corrected values are intentionally both retained to prevent ambiguity.

## One-File Processing Example

```bash
python preprocess_opflearn.py \
  --input dataset/opflearn/extracted/pglib_opf_case118_ieee.csv \
  --output dataset/opflearn/processed/pglib_opf_case118_ieee.parquet \
  --chunksize 1000 \
  --overwrite
```

## Process All Extracted CSVs

```bash
python preprocess_opflearn.py \
  --input-dir dataset/opflearn/extracted \
  --output-dir dataset/opflearn/processed \
  --chunksize 1000 \
  --overwrite
```

## Inspection Example

```bash
python inspect_opflearn.py \
  dataset/opflearn/processed/pglib_opf_case14_ieee.parquet
```

## Serialize To PyG HDF5

Serialize all feasible processed parquet files to HydraGNN-compatible HDF5 datasets:

```bash
python serialize_opflearn_to_hdf5.py \
  --input-dir dataset/opflearn/processed \
  --output-dir dataset/opflearn/serialized_hdf5 \
  --overwrite
```

Serialize one file only:

```bash
python serialize_opflearn_to_hdf5.py \
  --input dataset/opflearn/processed/pglib_opf_case14_ieee.parquet \
  --output dataset/opflearn/serialized_hdf5/pglib_opf_case14_ieee.h5 \
  --overwrite
```

Notes:

- Each sample is represented as a single-node PyG graph (tabular OPFLearn row -> graph sample).
- Output directories contain `trainset`, `valset`, and `testset` serialized via HydraGNN `HDF5Writer`.
- By default, `INFEASIBLE_*.parquet` files are skipped; pass `--include-infeasible` to include them.

`inspect_opflearn.py` reports:

- file name, rows, columns
- inferred counts of loads/generators/buses/branches
- detected input/primal/dual groups
- missing-value counts
- load and generation summary statistics
- raw and corrected angle ranges
- voltage magnitude range
- branch flow ranges

## Validation and Testing

Run tests:

```bash
pytest -q
```

Tests cover parsing edge cases, correction formula, magnitude preservation, multi-column correction, no-`v_bus` failure mode, non-voltage column preservation, and an end-to-end CSV-to-Parquet conversion.

## ML Representation Recommendation

Do not train directly on complex `v_bus` strings.

Recommended real-valued targets:

- `vm_bus`
- `va_bus_rad`

Optional Cartesian alternative:

- `v_bus_real_corrected`
- `v_bus_imag_corrected`

For angle errors, use wrapped differences:

```python
def wrapped_angle_difference(prediction, target):
    return np.arctan2(np.sin(prediction - target), np.cos(prediction - target))
```

## Modeling Caveat: Homogeneous-Only (No Heterogeneous Graph)

> **Note / caveat.** The training and inference examples in `examples/opflearn`
> (`train_opflearn_solution_homogeneous.py`, `infer_opflearn_solution_homogeneous.py`)
> deliberately ship **homogeneous variants only**. There is no
> `*_heterogeneous.py` / `*_heterogeneous.json`, unlike the `opf` and `pglearn`
> examples.
>
> **Why.** OPFLearnData is *tabular*: each sample is one row mapping load
> setpoints to the corresponding AC-OPF solution. When serialized to PyG, every
> sample becomes a **single-node graph with no edges** (`x = [1, in_dim]`,
> `y = [1, out_dim]`, empty `edge_index`). There is no released bus/branch
> topology to build a heterogeneous bus/branch/generator graph from, so a
> heterogeneous model would have no node types or edges to operate on and would
> add no modeling value. For the same reason, degree-sensitive layers (e.g. PNA)
> are avoided and the example uses **GIN**, which operates correctly on
> edge-free graphs.
>
> If a future OPFLearn release (or a join against `pglib-opf` case files)
> provides explicit topology, a heterogeneous variant mirroring the `pglearn`
> example could then be added.

## Units and Multi-Grid Training Notes

- Treat power quantities as per-unit unless authoritative case metadata says otherwise.
- Do not merge different grids without recording grid identity, topology, base MVA, limits, and index maps.
- For multi-grid training, use physically meaningful normalization and preserve grid identity.

## End-to-End Commands

```bash
python download_opflearn.py --archive --extract
```

```bash
python preprocess_opflearn.py --input-dir dataset/opflearn/extracted --output-dir dataset/opflearn/processed
```

```bash
python inspect_opflearn.py dataset/opflearn/processed/pglib_opf_case14_ieee.parquet
```

```bash
python serialize_opflearn_to_hdf5.py --input-dir dataset/opflearn/processed --output-dir dataset/opflearn/serialized_hdf5
```

```bash
pytest -q
```
