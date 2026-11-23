# Fatigue localization and SVE maximum prediction

This code accompanies *Extreme-value-aware graph surrogates for fatigue localization in austenitic stainless steel*, by Erdem Caliskan, Anik Das Anto, Stephanie TerMaath, Reza Abedi, and Massimiliano Lupo Pasini.

The release contains the graph model, fixed repeated-evaluation protocol, maximum estimators, conventional regressors, matched connectivity controls, residual grain head, threshold comparisons, mechanical predictions, and uncertainty calculations. Code is licensed under BSD-3-Clause; the separately deposited simulation data use CC BY 4.0.

## Source and environment

Use HydraGNN commit `63cfee31cd93266691b971e6de521bb7b17dabd1` with the supplied overlay. The two HydraGNN changes enable configured SAGE dropout and its configuration argument.

`HydraGNN_fip_core.patch` records these two changes as a zero-context patch (`git apply --unidiff-zero`); the branch helper installs their complete source files directly.

The observed environment used Python 3.11.6, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.1, PyTorch 2.8.0 and PyTorch Geometric 2.6.1, with CUDA 12.8. `requirements-observed.txt` records installed package versions; it is an environment record rather than a hardware-independent installer. Install HydraGNN from this checkout with `python -m pip install -e .`. The optional threshold likelihood library requires a C++17 compiler (`g++`) and is built from `core_native.cpp` on first use.

## Assemble the data

Extract the published data archives at the checkout root, preserving their `examples/`, `revision_restart_2026_10_05/`, `revision_R2/`, and `revision_final/` paths. Copy the downloaded `FILE_MANIFEST.json` there as well (or set `FIP_DATA_MANIFEST` to its location). The dataset and checkpoint manifests distinguish the original 47-SVE assessment from the repeated 940-SVE assessment and the aluminum diagnostic. Do not combine their results solely by model name.

Run commands from the checkout root. The code resolves data paths relative to that root; `FIP_RELEASE_ROOT` may identify another **fully assembled code-and-data workspace** with the same directory structure. Serialized preprocessing states retain their original module names and are relocated at load time. Only load the trusted, checksum-verified pickle and PyTorch files supplied with this release.

Several scientific scripts write results beside their source. Use a separate writable copy of the assembled release for a new experiment and retain the downloaded archives and checksums. New residual training gets a new source-version identifier; this identifier must not be substituted for the published checkpoint version.

## Checks and execution

```bash
python examples/fip_r2/check_release.py --imports
python examples/fip_r2/run.py replay --show
python examples/fip_r2/run.py replay -- --help
python examples/fip_r2/run.py uncertainty -- --help
```

`run.py` lists the supported study entrypoints. Use `--show` to inspect an entrypoint without executing it. Scripts that do not expose an argument parser are run without extra arguments and use the declared fixed settings. Their outputs can replace corresponding results in a writable working copy.

The main repeated-evaluation model is in `examples/fip_revision_restart_20261005/final_repeated_evaluation/gnn/runner.py`. Its factory installs the grain residual decoder and the shared graph-context projector before loading weights or building the optimizer. It has 377,974 parameters. The saved checkpoint and fitted preprocessing must come from the same repeat/fold.

For the complete primary prediction, run replay, bridge, size baselines, calibration and bridge comparison. The fitted offset/slope uses only eligible stopping SVEs, and insufficient selected tails use the fitting-set mean maximum at the same physical size. The minimum of six finite input grain peaks is distinct from the minimum of four selected peaks.

To load a published residual head independently of the version assigned to relocated source:

```bash
python examples/fip_r2/replay_residual.py --repeat 0 --fold 0 \
  --arm critical_treatment --source-version 55f907da468ecf20 \
  --checkpoint-sha256 SHA256_FROM_DATA_MANIFEST --out residual_check --check-only
```

Omit `--check-only` to reconstruct the frozen graph features and export residual grain predictions for that fold. This route verifies the published head-selection seal, strictly loads both networks and performs no optimization. Its output folder must be new.

To train the fixed residual experiment in a new working copy, run `python examples/fip_r2/run.py residual-head -- --pair all --prepare`. After all ten preparations complete, this creates `training_config_<version>.json` in the residual study directory. Pass that file to `python examples/fip_r2/run.py residual-head -- --pair all --fit --device cuda:0 --release PATH_TO_TRAINING_CONFIG`. The configuration fixes both arms, all twenty fits, partitions, preparations and source hashes. An explicit `--fit` command starts fitting after this configuration is verified. The comparison code accepts `FIP_RESIDUAL_VERSION` when analyzing a new version.

The original 47-SVE training and plotting code is under `examples/fip_unified/isaac_v10_t011_multiseed/code/`. The aluminum block-maximum calculations are under `revision_final/working/aluminum_160_audit_2026_11_02/block_maxima/`; these are observed-response diagnostics, not steel predictive validation.

## Create the Git branch

Use a clean checkout at the base commit and apply this allowlisted overlay:

```bash
bash /path/to/overlay/prepare_branch.sh --verify-overlay
bash /path/to/overlay/prepare_branch.sh --checkout /path/to/clean/HydraGNN \
  --branch paper/fip-r2-release
```

The helper creates a local branch and stages only listed source files and the publication-specific ignore rules. It refuses a dirty or different-base checkout. It does not commit or push. Inspect the staged diff, commit, and push using your preferred remote.

## Verification scope

Syntax, imports, model-state compatibility, release-path handling and focused numerical checks are recorded separately from the published scientific results. Preparing this release does not rerun every neural fit or every manuscript analysis. The code manifest binds each released source file; the data manifests bind the scientific inputs and published outputs.
