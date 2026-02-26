# SC26 Single-Model Training + Inference Jobs

This folder contains two Slurm job scripts for Frontier:

- `job-sc26-single-model-training.sh`: trains a HydraGNN model.
- `job-sc26-single-model-inference.sh`: evaluates a trained checkpoint with `inference.py`.

Both scripts are configured for distributed multi-node execution (`srun` with 16 nodes, 8 GPU tasks per node).

## Main Features

## 1) Training job (`job-sc26-single-model-training.sh`)

- Sets up Frontier modules and the HydraGNN conda environment.
- Exports runtime/distributed environment variables.
- Runs:
  - `gfm_mlip_all_mpnn.py`
- Uses key scale knobs:
  - `HYDRAGNN_MAX_NUM_BATCH`
  - `NUM_EPOCH`
  - `BATCH_SIZE`
- Computes total training samples internally as:

  `num_samples = BATCH_SIZE * HYDRAGNN_MAX_NUM_BATCH * NUM_EPOCH`

## 2) Inference job (`job-sc26-single-model-inference.sh`)

- Uses the same environment + distributed launch pattern.
- Runs:
  - `inference.py`
- Loads a hard-coded checkpoint log directory via:
  - `CHECKPOINT_LOGDIR=".../logs/multidataset_hpo-..."`
- Evaluates MAE metrics (Energy, Energy per atom, Forces).

## How To Run

From the HydraGNN root:

```bash
cd /lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2026/HydraGNN

# Training
sbatch examples/multidataset_hpo_sc26/job-sc26-single-model-training.sh

# Inference (after training finishes and CHECKPOINT_LOGDIR is set correctly)
sbatch examples/multidataset_hpo_sc26/job-sc26-single-model-inference.sh
```

## Single Dataset vs Multi Dataset

By default both scripts use one dataset:

```bash
MULTI_MODEL_LIST=$datadir0
```

To switch to multi-dataset training/inference, replace with the full list line already present in both scripts:

```bash
# MULTI_MODEL_LIST=$datadir0,$datadir1,...,$datadir15
```

(Use the explicit full line from the script to avoid typos.)

## Important Caveat (Multi Dataset + FSDP2)

For multi-dataset runs, **FSDP2 must be turned off**.

In `job-sc26-single-model-training.sh`, set:

```bash
export HYDRAGNN_USE_FSDP=0
```

If you keep FSDP enabled for single-dataset experiments, restore it as needed.

## Main Environment Variables and Behavior

## Cluster / runtime

- `OMP_NUM_THREADS`
  - CPU threads per rank. Higher values can improve host-side work but may increase contention.
- `HYDRAGNN_NUM_WORKERS`
  - DataLoader worker count. Larger values may improve input throughput but can increase memory/overhead.
- `PYTHONNOUSERSITE=1`
  - Prevents accidental imports from user site-packages.

## Data / scaling

- `BATCH_SIZE`
  - Number of graphs per step.
  - Increasing it raises memory usage and changes optimization dynamics.
- `HYDRAGNN_MAX_NUM_BATCH`
  - Caps batches per epoch (or processed chunk depending on loader behavior).
- `NUM_EPOCH`
  - Number of epochs for training.
- `NUM_SAMPLES`
  - Derived in scripts as `BATCH_SIZE * HYDRAGNN_MAX_NUM_BATCH * NUM_EPOCH`.
  - Increasing it expands total data processed.

## Distributed networking

- `MASTER_ADDR`, `MASTER_PORT`
  - Rendezvous endpoint for distributed initialization.
- `HYDRAGNN_MASTER_ADDR`, `HYDRAGNN_MASTER_PORT`
  - HydraGNN-side distributed endpoint variables.
- `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`
  - Force control/data traffic over the intended network interface (`hsn0`).

## Model/distributed strategy

- `HYDRAGNN_USE_FSDP`
  - Enables/disables FSDP.
- `HYDRAGNN_FSDP_VERSION`
  - FSDP implementation variant.
- `HYDRAGNN_FSDP_STRATEGY`
  - Sharding strategy when FSDP is enabled.
- `TASK_PARALLEL` and `TASK_PARALLEL_ARG`
  - Enables task-parallel mode when set.

## Dataset selection

- `datadir0 ... datadir15`
  - Named dataset aliases used to build `MULTI_MODEL_LIST`.
- `MULTI_MODEL_LIST`
  - Comma-separated dataset list consumed by both training and inference.

## Inference-specific

- `CHECKPOINT_LOGDIR`
  - Directory containing `config.json` and checkpoint `.pk` files.
  - If incorrect, inference exits early with a clear error.
- `INFER_PRECISION`
  - Precision passed to `inference.py` (for example `fp64`).

## Practical Workflow

1. Launch training script.
2. Wait for completion and identify the produced log directory in `HydraGNN/logs`.
3. Update hard-coded `CHECKPOINT_LOGDIR` in inference script (or change it before submission if you choose to reintroduce an override).
4. Launch inference script.

If you switch from single-dataset to multi-dataset, apply the `MULTI_MODEL_LIST` change in **both** scripts and disable FSDP2 as noted above.
