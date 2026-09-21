# GO Challenge 1 ML Pipeline

Reproducible Python workflow for downloading, parsing, solving, validating, and converting ARPA-E GO Competition Challenge 1 scenarios into machine-learning datasets for AC power flow (PF) and AC optimal power flow (OPF).

## Why Challenge 1

Challenge 1 is the best starting point for static PF/OPF ML tasks because it provides per-scenario network/problem-instance files (`.raw/.rop/.inl/.con`) with classical transmission-model structure. Challenge 2 and Challenge 3 are less direct for this first static PF/OPF milestone.

## Official Source

- OEDI record: https://data.openei.org/submissions/6153
- DOI: https://doi.org/10.25984/2437761

Use the OEDI page to copy current download URLs. Do not rely on legacy competition-site URLs.

## Recommended Starting Archives

Start with smaller/earlier datasets first:

- Challenge 1 Original Dataset 1 Scenarios
- Challenge 1 Original Dataset 2 Scenarios
- Challenge 1 Final Event Offline Synthetic Scenarios
- Challenge 1 Final Event Real-Time Synthetic Scenarios
- Challenge 1 Dataset Format

## Project Layout

```text
go_challenge1_ml/
├── README.md
├── requirements.txt
├── download_go_challenge1.py
├── inspect_go_challenge1.py
├── generate_pf_data.py
├── generate_opf_data.py
├── validate_dataset.py
├── convert_to_graphs.py
├── go_challenge1/
│   ├── __init__.py
│   ├── download.py
│   ├── discovery.py
│   ├── parser_raw.py
│   ├── parser_rop.py
│   ├── parser_inl.py
│   ├── parser_con.py
│   ├── network_model.py
│   ├── matpower_export.py
│   ├── pandapower_export.py
│   ├── powermodels_export.py
│   ├── pf_generation.py
│   ├── opf_generation.py
│   ├── contingency.py
│   ├── graph_conversion.py
│   ├── validation.py
│   └── metadata.py
├── tests/
│   ├── test_parsers.py
│   ├── test_network_conversion.py
│   ├── test_pf_generation.py
│   ├── test_opf_generation.py
│   └── test_graph_conversion.py
└── data/
    └── go_challenge1/
        ├── raw/
        ├── extracted/
        ├── parsed/
        ├── solved/
        └── processed/
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Downloader Features

`download_go_challenge1.py` supports:

1. User-supplied OEDI URL
2. Streaming download
3. Redirects and retries
4. Resume via HTTP Range and `.part` file
5. Atomic rename after completion
6. ZIP/TAR/TAR.GZ/TGZ extraction
7. No overwrite unless `--force`
8. Final file size logging
9. Recursive scenario-directory reporting after extraction

Example:

```bash
python download_go_challenge1.py \
  --url "<OEDI_FILE_URL>" \
  --output data/go_challenge1/raw/challenge1_dataset1.zip \
  --extract
```

## Scenario Structure

Expected files per scenario directory:

- `case.raw`
- `case.rop`
- `case.inl`
- `case.con`

Interpretation:

- RAW: topology/electrical network and statuses
- ROP: dispatch and costs
- INL: participation-factor style data
- CON: contingencies

Field assumptions are intentionally conservative. Before production runs, inspect the official format document and real files from your selected archive.

## Discovery and Inspection

```bash
python inspect_go_challenge1.py \
  --input-dir data/go_challenge1/extracted
```

```bash
python inspect_go_challenge1.py \
  --input-dir data/go_challenge1/extracted \
  --scenario "<SCENARIO_NAME>"
```

The inspector reports scenario ID, network name, required-file presence, file sizes, and parsed counts.

## Solver Backends

Implemented now:

- `pandapower` (open-source default)

Designed extension points:

- MATPOWER
- PYPOWER
- PowerModels.jl
- Ipopt
- Knitro (when available)

## PF and OPF Definitions

PF mapping:

- Inputs: topology, loads, prescribed generator controls, taps/shunts/status
- Targets: Vm, Va, Qg, slack Pg, branch flows, losses, convergence metadata

OPF mapping:

- Inputs: topology, loads, limits, costs, statuses
- Targets: optimal Pg/Qg/Vm/Va/flows/objective/status (+ duals when available)

Generator active power has a task-dependent role:

- PF: observed control (`Pg_is_observed=1`, `Pg_is_target=0`)
- OPF: target (`Pg_is_observed=0`, `Pg_is_target=1`)

Unknown values are represented with explicit observation masks, never by bare zero.

## Base-Case First, Then Contingency

Current milestone focuses on base-case PF/OPF generation and validation. Contingency-aware PF/SCOPF is scaffolded via parser + topology transforms and is intended as the next implementation stage.

## Validation Tolerances

Configurable tolerances are exposed in generation CLIs:

- `power_balance_tolerance` (default `1e-5` pu)
- `voltage_tolerance` (default `1e-6` pu)
- `branch_limit_tolerance` (default `1e-5` pu)

Solver return status is not treated as sufficient; residual/violation metrics are tracked.

## Graph Representation

`convert_to_graphs.py` emits `torch_geometric.data.HeteroData` with node types:

- `bus`
- `generator`
- `load`
- `shunt`

Edge types:

- `(generator, connected_to, bus)`
- `(load, connected_to, bus)`
- `(shunt, connected_to, bus)`
- `(bus, line, bus)`
- `(bus, transformer, bus)`

Task metadata per sample includes `task_id`, `task_name`, `scenario_id`, `grid_id`, `sample_id`, contingency flags, solver name/status.

## Angle Convention

Angles are handled internally in radians where represented in network models. Wrapped-angle errors should use:

```python
def wrapped_angle_difference(prediction, target):
    return np.arctan2(np.sin(prediction - target), np.cos(prediction - target))
```

Do not apply OPFLearn-specific voltage-angle correction to GO Challenge 1 data.

## Output Artifacts

```text
data/go_challenge1/processed/
├── pf/
│   ├── graphs/
│   ├── metadata.parquet
│   └── normalization.json
├── opf/
│   ├── graphs/
│   ├── metadata.parquet
│   └── normalization.json
├── contingency_pf/
└── scopf/
```

## Split and Leakage Guidance

Support workflows for random/sample/scenario/family/event/unseen-network splits. Avoid leakage: samples derived from the same source scenario should not cross train/test split boundaries.

## Reproducibility

Set random seeds in downstream split/perturbation scripts and log:

- solver name/version
- tolerances
- dataset URL/source archive
- parse/solve timestamps

## Important Data-Generation Caveat

GO Competition scenario files are optimization problem instances, not guaranteed complete supervised labels. Full PF/OPF labels must generally be generated by solving the scenarios.

## CLI Workflow

```bash
python download_go_challenge1.py \
  --url "<OEDI_FILE_URL>" \
  --output data/go_challenge1/raw/challenge1.zip \
  --extract
```

```bash
python inspect_go_challenge1.py \
  --input-dir data/go_challenge1/extracted
```

```bash
python generate_opf_data.py \
  --input-dir data/go_challenge1/extracted \
  --output-dir data/go_challenge1/solved/opf \
  --solver pandapower \
  --limit-scenarios 10
```

```bash
python generate_pf_data.py \
  --input-dir data/go_challenge1/extracted \
  --opf-solution-dir data/go_challenge1/solved/opf \
  --output-dir data/go_challenge1/solved/pf \
  --use-opf-controls
```

```bash
python convert_to_graphs.py \
  --scenario-dir data/go_challenge1/extracted \
  --solution-dir data/go_challenge1/solved \
  --output-dir data/go_challenge1/processed \
  --task both \
  --format pyg
```

```bash
python validate_dataset.py \
  --input-dir data/go_challenge1/processed
```

## License and Citation

Follow this repository's main license and cite the OEDI Challenge 1 DOI:

- https://doi.org/10.25984/2437761
