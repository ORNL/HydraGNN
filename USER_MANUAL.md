# HydraGNN User Manual

A comprehensive guide to using HydraGNN for distributed graph neural network training and inference.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Data Pre-processing](#data-pre-processing)
4. [Model Configuration and Construction](#model-configuration-and-construction)
5. [Scalable Data Management](#scalable-data-management)
6. [Training Pipeline](#training-pipeline)
7. [Advanced Features](#advanced-features)
8. [Examples and Use Cases](#examples-and-use-cases)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

HydraGNN is a distributed PyTorch implementation of multi-headed graph convolutional neural networks designed for supercomputing environments. It provides:

- **Multi-headed Prediction**: Support for both graph-level and node-level property prediction
- **Distributed Data Parallelism**: Scalable training across multiple nodes and GPUs
- **Multiple Architectures**: Various GNN architectures (PNA, EGNN, CGCNN, etc.)
- **Advanced Features**: Global attention, geometric equivariance, and multi-dataset training

### Key Capabilities

- Train on molecular and materials datasets with millions of samples
- Handle multiple datasets simultaneously with different data distributions
- Scale to hundreds of GPUs using MPI and distributed computing frameworks
- Support various data formats including ADIOS2 for high-performance I/O
- Flexible configuration system for rapid prototyping and experimentation

---

## Installation and Setup

### Dependencies and Installation

HydraGNN uses a modular requirements system for flexible and reproducible installation. The recommended way to install all necessary dependencies is to use the provided installation script:

> **Python versions:** HydraGNN is tested and supported on **Python 3.11
> through 3.14**. Facility installation scripts default to Python 3.11 unless
> documented otherwise for that system.

#### Recommended: Automated Installation
```bash
./install_dependencies.sh
```
This script installs the following requirements in order:
- `requirements-base.txt`: Core Python dependencies for HydraGNN
- `requirements-torch.txt`: PyTorch and related dependencies
- `requirements-pyg.txt`: PyTorch Geometric and extensions

You can also install development or optional dependencies:
```bash
# For development tools (testing, linting, etc.)
./install_dependencies.sh dev

# For all optional features (including development and extra packages)
./install_dependencies.sh all optional
```

#### Manual Installation (Advanced)
If you prefer, you can install requirements manually:
```bash
pip install --no-build-isolation -v -r requirements-base.txt
pip install --no-build-isolation -v -r requirements-torch.txt
pip install --no-build-isolation -v -r requirements-pyg.txt
# For development tools (optional)
pip install --no-build-isolation -v -r requirements-dev.txt
# For DeepSpeed support (optional)
pip install --no-build-isolation -v -r requirements-deepspeed.txt
# For additional optional features
pip install --no-build-isolation -v -r requirements-optional.txt
```

### Installation Methods

#### Developer Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/ORNL/HydraGNN.git
cd HydraGNN

# Install in developer mode
python -m pip install -e .
```

#### Environment Variable Setup
```bash
# Alternative to installation
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### Static Installation
```bash
# For production environments
python -m pip install .
```

### Verification

Test your installation:
```bash
python -c "import hydragnn; print('HydraGNN installed successfully')"
```

### DOE Supercomputer Installation

Tested installation scripts are organized by facility and system under
`scripts/hpc/<facility>/<system>/installation/`:

- **Frontier** (AMD, ROCm 6.4, 7.1, 7.2, and 7.13)
- **Aurora** (Intel XPU)
- **Perlmutter** (NVIDIA)
- **Andes** (NVIDIA)
```

---

## Data Pre-processing

HydraGNN supports multiple data formats and provides comprehensive preprocessing capabilities.

Dataset creation and application-specific preprocessing are caller
responsibilities. HydraGNN training consumes prepared PyG graph samples; the
legacy `run_training`, `run_prediction`, and `SerializedDataLoader`
orchestration APIs have been removed. Prepared pickle, per-sample `.pt`, ADIOS,
and DDStore-backed datasets remain available through their current loaders.
Loading an already prepared dataset does not silently reconstruct, renormalize,
or otherwise mutate its stored features. Rebuild the dataset explicitly when
its preprocessing contract changes.

Reusable data utilities are documented in
[dataset downloads](docs/dataset_downloads.md) and
[materials preprocessing](docs/materials_preprocessing.md). Variable-size graph
training can use [cost-aware batching](docs/cost_aware_batching.md).

### Supported Data Formats

#### 1. Raw source formats

HydraGNN does not parse application-specific raw formats such as LSMS or CFG.
The application owns that parser and must convert every source record into a
PyG `Data` object whose named attributes match the top-level JSON `Variables`
schema. This separation prevents HydraGNN from guessing column meanings,
units, dimensions, or target semantics. `examples/lsms/lsms_preprocess.py` is
an example-owned LSMS converter and `examples/lsms/lsms.py` can invoke it.
Likewise, `examples/eam/eam_preprocess.py` owns the EAM CFG mapping, and the
Ising example owns its synthetic-data generator. These conveniences do not
make LSMS, CFG, or Ising parsing HydraGNN core APIs.

#### 2. Serialized Formats

**Prepared pickle or per-sample PyG `.pt` data**:
```python
from hydragnn.preprocess.graph_dataset import load_prepared_graph_dataset

dataset = load_prepared_graph_dataset("dataset/train.pkl")
# A directory containing serialized PyG .pt samples is also accepted.
# dataset = load_prepared_graph_dataset("dataset/train_samples")
```

These files are pickle-backed and must come from a trusted source. The loader
returns their stored graph objects without rebuilding connectivity, compiling
named variables, normalizing features, or regenerating descriptors.

**ADIOS2 Format**: High-performance binary format for large datasets
```python
# Loading ADIOS datasets
from hydragnn.utils.datasets.adiosdataset import AdiosDataset
dataset = AdiosDataset(filename, "trainset", comm)
```

### Data Loading Pipeline

#### Configuration-Based Loading

```json
{
    "Dataset": {
        "name": "FePt_32atoms",
        "path": {
            "train": "./dataset/FePt_train.pkl",
            "validate": "./dataset/FePt_validate.pkl",
            "test": "./dataset/FePt_test.pkl"
        }
    },
    "Variables": {
        "inputs": [
            {"name": "num_of_protons", "level": "node", "dim": 1}
        ],
        "outputs": [
            {"name": "free_energy_per_atom", "level": "graph", "dim": 1},
            {"name": "charge_density", "level": "node", "dim": 1},
            {"name": "magnetic_moment", "level": "node", "dim": 1}
        ]
    }
}
```

#### Programmatic Data Loading

```python
from hydragnn.preprocess.load_data import dataset_loading_and_splitting

# Load prepared splits and construct data loaders
train_loader, val_loader, test_loader = dataset_loading_and_splitting(config)
```

### Data Splitting Strategies

#### Standard Splitting
```json
{
    "NeuralNetwork": {
        "Training": {
            "perc_train": 0.7,  // 70% training, 15% validation, 15% test
        }
    }
}
```

#### Compositional Stratified Splitting
For materials datasets with varying compositions:
```json
{
    "Dataset": {
        "compositional_stratified_splitting": true
    }
}
```

#### Energy Linear Regression
A preprocessing tool for computing element-wise energy linear regression baselines before training:
```bash
# Example usage
cd examples/multidataset
python energy_linear_regression.py --notestset OMat24
```
The regression coefficients are stored in the ADIOS dataset under `energy_linear_regression_coeff`.

### Custom Data Preprocessing

#### Writing Custom Data Loaders

```python
from torch_geometric.data import Data

# Users parse their source format and construct named PyG attributes.
sample = Data(
    atomic_numbers=atomic_numbers,  # (N, 1)
    pos=positions,                  # (N, 3)
    energy=energy,                  # (1, 1)
    forces=forces,                  # (N, 3)
)
```

#### Data Serialization for Performance

```python
from hydragnn.utils.input_config_parsing.variable_schema import (
    parse_variable_schema,
    prepare_data_from_schema,
)
from hydragnn.utils.datasets.serializeddataset import SerializedWriter

# Compile named source attributes before writing a training-ready artifact.
schema = parse_variable_schema(config["Variables"])
prepared = [prepare_data_from_schema(sample.clone(), schema) for sample in dataset]
writer = SerializedWriter(prepared, basedir, dataset_name, "trainset")
```

---

## Model Configuration and Construction

HydraGNN provides extensive configuration options for building graph neural networks.

### Architecture Configuration

#### Basic Architecture Setup

```json
{
    "NeuralNetwork": {
        "Architecture": {
            "mpnn_type": "PNA",           // Graph neural network type
            "hidden_dim": 128,            // Hidden layer dimensions
            "num_conv_layers": 4,         // Number of convolution layers
            "radius": 7.0,                // Cutoff radius for neighborhoods
            "max_neighbours": 100,        // Maximum neighbors per node
            "activation_function": "relu" // Activation function
        }
    }
}
```

#### Supported MPNN Types

1. **PNA (Principal Neighbourhood Aggregation)**
   ```json
   {
       "mpnn_type": "PNA",
       "hidden_dim": 128,
       "num_conv_layers": 4
   }
   ```

2. **EGNN (E(n) Equivariant Graph Neural Networks)**
   ```json
   {
       "mpnn_type": "EGNN",
       "equivariance": true,
       "hidden_dim": 128
   }
   ```

3. **CGCNN (Crystal Graph Convolutional Neural Networks)**
   ```json
   {
       "mpnn_type": "CGCNN",
       "num_gaussians": 50,
       "hidden_dim": 128
   }
   ```

4. **SchNet**
   ```json
   {
       "mpnn_type": "SchNet",
       "num_gaussians": 50,
       "num_filters": 128
   }
   ```

5. **PAINN (Physics-Aware Graph Neural Networks)**
   ```json
   {
       "mpnn_type": "PAINN",
       "num_radial": 6,
       "num_spherical": 7
   }
   ```

6. **MACE (Multi-Atomic Cluster Expansion)**
   ```json
   {
       "mpnn_type": "MACE",
       "num_radial": 8,
       "interaction_order": 3
   }
   ```

### Output Head Configuration

#### Multi-Task Learning Setup

```json
{
    "output_heads": {
        "graph": {
            "num_sharedlayers": 2,
            "dim_sharedlayers": 50,
            "num_headlayers": 2,
            "dim_headlayers": [100, 50]
        },
        "node": {
            "num_headlayers": 2,
            "dim_headlayers": [100, 50],
            "type": "mlp"
        }
    },
    "task_weights": [1.0, 1.0, 1.0]  // Relative weights for different tasks
}
```

#### Variables

The `Variables` section is the public contract between a dataset and
HydraGNN. A dataset importer only stores named source tensors on each PyG
`Data` object. It must not concatenate those tensors into `x`, `edge_attr`,
`graph_attr`, or `y`; HydraGNN constructs those internal tensors from the
schema.

```json
{
    "Variables": {
        "inputs": [
            {"name": "node_features", "level": "node", "dim": 3}
        ],
        "outputs": [
            {"name": "energy", "level": "graph", "dim": 1},
            {"name": "forces", "level": "node", "dim": 3}
        ]
    }
}
```

Every configured name must be an attribute of the PyG sample with exactly the
declared shape:

- a node attribute has shape `(N, dim)`, where `N` is the sample's node count;
- an edge attribute has shape `(E, dim)`, where `E` is the number of columns in
  `edge_index`; and
- a graph attribute has shape `(1, dim)` for an individual sample.

For example, the configuration above expects an importer to create data such
as:

```python
data = Data(
    node_features=node_features,  # (N, 3)
    energy=energy,                # (1, 1)
    forces=forces,                # (N, 3)
    edge_index=edge_index,
)
```

The importer does not set `data.x`, `data.y`, or `data.y_loc`. During schema
preparation, HydraGNN validates the named tensors and creates its internal
representation according to these rules:

| JSON variables | Internal tensor | Construction |
|---|---|---|
| node inputs with role `feature` | `data.x` | concatenate columns in JSON order |
| node input with role `position` | `data.pos` | validate as `(N, 3)`; do not concatenate |
| edge inputs | `data.edge_attr` | concatenate columns in JSON order |
| graph inputs | `data.graph_attr` | concatenate columns in JSON order |
| node outputs | `data.node_output` | concatenate columns in JSON order |
| edge outputs | `data.edge_output` | concatenate columns in JSON order |
| graph outputs | `data.graph_output` | concatenate columns in JSON order |
| all outputs | `data.y` | flatten each output to `(-1, 1)`, then concatenate in overall JSON order |
| output boundaries | `data.y_loc` | cumulative offsets delimiting each output inside `data.y` |

As a more explicit input example:

```json
"inputs": [
  {"name": "atomic_numbers", "level": "node", "dim": 1},
  {"name": "pos", "level": "node", "dim": 3, "role": "position"}
]
```

causes HydraGNN to validate `data.pos` as `(N, 3)` while constructing an
`(N, 1)` internal node-feature tensor equivalent to:

```python
data.x = data.atomic_numbers
```

The `position` role is a geometric input contract, not a feature channel.
HydraGNN keeps it in `data.pos`, excludes it from `data.x` and `input_dim`, and
passes it separately to geometry-aware and equivariant models. Declaring `pos`
without `"role": "position"` is rejected to prevent accidental loss of
translation invariance or rotation equivariance.

#### Geometric positions and equivariance

Although Cartesian positions are model inputs, they must not be handled like
ordinary invariant node features. A translation changes the numerical value of
every coordinate, while a rotation mixes the x, y, and z components. Naively
concatenating those components into `data.x` exposes coordinate-frame-dependent
numbers as scalar feature channels and can invalidate the guarantees of an
invariant or equivariant architecture.

The required declaration is therefore:

```json
{"name": "pos", "level": "node", "dim": 3, "role": "position"}
```

and each PyG sample must provide:

```python
data.pos  # torch.Tensor with shape (data.num_nodes, 3)
```

During schema preparation, HydraGNN performs these operations separately:

1. It validates `data.pos`, including its node count and three coordinate
   columns, and preserves the tensor as `data.pos`.
2. It concatenates only node inputs whose role is `feature` to construct
   `data.x`.
3. It computes `Architecture.input_dim` from those feature inputs only; the
   three position dimensions are not included.
4. It passes `data.pos` through the dedicated geometric path used by neighbor
   construction, geometry-aware local message passing, equivariant global
   attention, and energy-gradient force calculations. The original tensor and
   its autograd relationship are preserved.

For example, if `atomic_numbers` has dimension 1, `chemical_state` has
dimension 4, and `pos` is the position input, the result is `data.x` with shape
`(N, 5)` and a separate `data.pos` with shape `(N, 3)`—not `data.x` with shape
`(N, 8)`. Dataset importers should assign the three named source attributes and
must not perform either concatenation themselves.

This is a general geometry contract, not an MLIP-only convention. Any
HydraGNN model that relies on translation invariance or rotational
invariance/equivariance should receive Cartesian coordinates this way.

To prevent silent misuse, schema validation rejects all of the following:

- `pos` declared without `"role": "position"`;
- a position input whose name is not `pos`;
- a position input that is not node-level or does not have `dim: 3`;
- `role: "position"` on an output; and
- a schema with no ordinary node feature input from which to build `data.x`.

In particular, users must never work around this contract by manually
concatenating `data.pos` into `data.x`.

The names of HydraGNN's derived tensors are reserved and cannot be declared as
source variables. This includes `x`, `edge_attr`, `graph_attr`, `y`, `y_loc`,
`node_output`, `edge_output`, and `graph_output` (as well as structural PyG
names managed internally). For example, declare a source input as
`node_features`, not `x`; schema preparation preserves `data.node_features`
and constructs `data.x` from it. Rejecting collisions guarantees that a named
source attribute is never overwritten by the tensor derived from that source.

The named schema intentionally does not provide compatibility with the removed
`Variables_of_interest` format. In particular, `update_config_minmax` was tied
to column indices (`input_node_features` and `output_index`) and has been
removed from both `config_utils` and the package exports. Downstream code must
not import it. Normalize named attributes explicitly in application-owned
preprocessing and retain normalization statistics alongside the serialized
dataset or application configuration.

The migration is intentionally direct rather than compatibility-based:

| Removed convention | Current replacement |
|---|---|
| `NeuralNetwork.Variables_of_interest` | top-level `Variables.inputs` and `Variables.outputs` |
| feature and target column indices | exact named PyG attributes |
| manual construction of `data.x` and target tensors | `prepare_data_from_schema` or `SchemaPreparedDataset` |
| HydraGNN-owned CFG/LSMS parsing | application-owned parsing into PyG `Data` objects |
| `update_config_minmax` | application-owned normalization of named attributes |

There is no legacy fallback. A configuration or raw sample using the removed
contract fails instead of being interpreted heuristically.

If the outputs are `energy` with shape `(1, 1)` followed by `forces` with
shape `(N, 3)`, HydraGNN constructs `data.y` with shape `(1 + 3*N, 1)` and
`data.y_loc = [[0, 1, 1 + 3*N]]`. The offsets preserve the two output-head
boundaries. The original attributes (`node_features`, `energy`, `forces`, and
so on) remain available on the `Data` object.

If a schema has no inputs or outputs at a particular level, HydraGNN removes a
stale internal tensor for that level. This makes repeated schema preparation
idempotent and prevents undeclared features from silently reaching the model.
More precisely:

- At least one node feature input is mandatory, so `data.x` is always reconstructed
  from the declared node attributes. Any pre-existing `data.x` is overwritten.
- When edge inputs are declared, `data.edge_attr` is reconstructed from them;
  otherwise a pre-existing `data.edge_attr` is removed.
- When graph inputs are declared, `data.graph_attr` is reconstructed from them;
  otherwise a pre-existing `data.graph_attr` is removed.
- When outputs are declared, HydraGNN reconstructs `data.y`, `data.y_loc`, and
  the applicable level-specific output tensors. With no outputs, these internal
  tensors are removed. A level-specific output tensor is also removed when the
  current schema contains no outputs at that level.

These removals apply only to HydraGNN's derived internal tensors. The named
source attributes declared by the user remain on the PyG object. Consequently,
third-party PyG samples may initially contain conventional attributes such as
`x`, `edge_attr`, or `y`, but those attributes cannot bypass the current JSON
schema or survive from a previous schema preparation unnoticed.

#### Reusing a prepared dataset

Schema compilation is a preprocessing operation, not a load-time migration.
Before serialization, each training-ready graph must already contain the
internal tensors required by its configuration: `x`, `edge_index`, any
declared `edge_attr` or `graph_attr`, and `y`/`y_loc` for configured outputs.
Their column dimensions must agree with the architecture derived from the
schema; in particular, `x.shape[1]` must equal `Architecture.input_dim`, and an
edge feature tensor must have one row per edge and
`Architecture.edge_dim` columns.

When HydraGNN later opens a prepared pickle, `.pt`, ADIOS, or DDStore artifact,
it treats those stored tensors as authoritative and does not recover missing
named source attributes, rerun schema compilation, or reinterpret an older
cache. If the schema or any preprocessing choice changes, create a new prepared
artifact rather than silently adapting the old one.

Serialized PyG `.pt` samples are pickle-backed. Loading them with PyTorch may
execute code embedded in the file, so HydraGNN treats these files as trusted
local dataset artifacts. Do not train from a `.pt` dataset directory obtained
from an untrusted source; inspect or convert such data through a safe format
before loading it with HydraGNN.

### Global Attention Mechanisms

#### GPS (Graph Positional and Structural Attention)

```json
{
    "Architecture": {
        "global_attn_engine": "GPS",
        "global_attn_type": "multihead",
        "global_attn_heads": 8,
        "pe_dim": 16,  // Positional encoding dimension
        "hidden_dim": 128  // Must be divisible by global_attn_heads
    }
}
```

### Graph-level conditioning (concat_node default)

- Enable with `NeuralNetwork.Architecture.use_graph_attr_conditioning: true` and select `graph_attr_conditioning_mode`:
    - `"concat_node"` (default): concatenate `graph_attr` onto node embeddings and project back to `hidden_dim`.
    - `"film"`: per-graph FiLM scale/shift on invariant node channels.
    - `"fuse_pool"`: fuse `graph_attr` with the pooled graph embedding before prediction heads.
- Conditioning does not touch equivariant channels, so equivariance is preserved only if `graph_attr` is itself invariant.
- Provide `graph_attr` tensors during data loading; a missing attribute raises an error when conditioning is enabled.
- Orientation-dependent attributes will intentionally break rotation/translation equivariance—use only when that is desired.

### Geometric Features

#### Periodic Boundary Conditions
```json
{
    "Architecture": {
        "periodic_boundary_conditions": true
    }
}
```

#### Rotational Invariance
```json
{
    "Dataset": {
        "rotational_invariance": true
    }
}
```

### Model Creation

#### Programmatic Model Creation

```python
from hydragnn.models.create import create_model_config

# Create model from configuration
model = create_model_config(
    config=config["NeuralNetwork"],
    verbosity=config["Verbosity"]["level"]
)

# Print model architecture
from hydragnn.utils.model import print_model
print_model(model)
```

---

## Scalable Data Management

HydraGNN is designed for large-scale distributed computing environments.

### Distributed Data Loading

#### MPI-Based Data Distribution

```python
from mpi4py import MPI
from hydragnn.utils.datasets.adiosdataset import AdiosDataset

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load data with automatic distribution
dataset = AdiosDataset(filename, "trainset", comm)
```

#### Shared Memory Optimization

```python
# Enable shared memory for efficiency
opt = {
    "preload": True,
    "shmem": True,
}
dataset = AdiosDataset(filename, "trainset", comm, **opt)
```

### High-Performance I/O with ADIOS2

#### Writing Large Datasets

```python
from hydragnn.utils.datasets.adiosdataset import AdiosWriter

# Create ADIOS writer for large datasets
writer = AdiosWriter(filename, comm)
writer.add("trainset", train_dataset)
writer.add("valset", val_dataset)
writer.add("testset", test_dataset)

# Add global metadata
writer.add_global("minmax_node_feature", minmax_features)
writer.save()
```

#### Variable Graph Sizes

```bash
# Enable variable graph size support
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
```

#### Worker Configuration

```bash
# Set number of data loading workers
export HYDRAGNN_NUM_WORKERS=4

# Disable workers for memory-constrained environments
export HYDRAGNN_NUM_WORKERS=0
```

### Multi-Dataset Training

#### Configuration for Multiple Datasets

```python
# Command line example for multi-dataset training
python train.py --multi --multi_model_list=dataset1.bp,dataset2.bp,dataset3.bp
```

#### Load Balancing Across Datasets

```python
# Automatic load balancing based on dataset sizes
process_list = np.ceil(ndata_list / sum(ndata_list) * comm_size).astype(np.int32)
```

### Distributed Storage Systems

#### Data Preprocessing at Scale

```bash
# Use multiple nodes for data preprocessing
srun -N 4 -n 32 python preprocess_data.py --format adios
```

---

## Training Pipeline

HydraGNN provides a complete training pipeline with extensive configuration options.

### Basic Training Configuration

#### Training Parameters

```json
{
    "NeuralNetwork": {
        "Training": {
            "num_epoch": 100,
            "batch_size": 32,
            "loss_function_type": "mse",
            "precision": "fp32",
            "conv_checkpointing": false,
            "EarlyStopping": true,
            "patience": 10,
            "Optimizer": {
                "type": "AdamW",
                "learning_rate": 1e-3
            }
        }
    }
}
```

#### Checkpointing

```json
{
    "Training": {
        "Checkpoint": true,
        "checkpoint_warmup": 5,  // Start checkpointing after 5 epochs
        "continue": 0,           // Continue from epoch 0 (new training)
        "startfrom": "existing_model"  // Or "new_model"
    }
}
```

### Distributed Training

#### Basic Distributed Setup

```python
import hydragnn

# Initialize distributed execution before constructing the distributed model.
world_size, rank = hydragnn.utils.distributed.setup_ddp()
```

#### MPI Execution

```bash
# Single node, multiple GPUs
mpirun -np 8 python -u train.py --inputfile config.json

# Multi-node training
srun -N 4 -n 32 --gpus-per-task=1 python -u train.py --inputfile config.json
```

#### Environment Variables for Distributed Training

```bash
# MPI backend configuration
export HYDRAGNN_AGGR_BACKEND=mpi

# NCCL configuration for multi-GPU
export NCCL_PROTO=Simple
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

# AMD GPU setting
export MIOPEN_DISABLE_CACHE=1
```

### Advanced Training Features

#### Precision Control

HydraGNN supports multiple training precisions:

- **`"fp32"`** (default): Standard single-precision training.
- **`"bf16"`**: Mixed-precision with BF16 compute and FP32 master parameters, using `torch.autocast`.
- **`"fp64"`**: Double-precision training for high-accuracy applications.

```json
{
    "Training": {
        "precision": "bf16"
    }
}
```

#### Gradient Checkpointing

Enable activation recomputation to reduce GPU memory at the cost of extra compute:

```json
{
    "Training": {
        "conv_checkpointing": true
    }
}
```

#### Loss Function Options

```json
{
    "Training": {
        "loss_function_type": "mae",  // or "mse", "huber"
        "task_weights": [1.0, 10.0]   // Weight different tasks
    }
}
```

#### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode="min", 
    factor=0.5, 
    patience=5, 
    min_lr=0.00001
)
```

#### Machine-Learned Interatomic Potentials (MLIP)

HydraGNN supports energy-conserving interatomic potential workflows. When enabled, the model dynamically constructs the radius graph at each forward pass and computes forces as the negative gradient of predicted energy with respect to atomic positions.

```json
{
    "Architecture": {
        "enable_interatomic_potential": true
    },
    "Training": {
        "compute_grad_energy": true
    }
}
```

With `enable_interatomic_potential`, the training loss includes energy, per-atom energy, and force components. Set `compute_grad_energy` to `true` to derive forces via automatic differentiation of the energy prediction.
```

### Training Execution

#### Training Interface

```python
import hydragnn

# Dataset parsing and splitting are application responsibilities.
train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, batch_size=32
)
# Construct the model, optimizer, scheduler, and writer explicitly, then train.
hydragnn.train.train_validate_test(
    model, optimizer, train_loader, val_loader, test_loader,
    writer, scheduler, config["NeuralNetwork"], log_name, verbosity
)
```

#### Custom Training Loop

```python
from hydragnn.train.train_validate_test import train_validate_test

# Custom training with full control
train_validate_test(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    writer=writer,
    scheduler=scheduler,
    config=config["NeuralNetwork"],
    model_with_config_name=log_name,
    verbosity=verbosity,
    create_plots=True
)
```

### Model Persistence

#### Saving Models

```python
from hydragnn.utils.model import save_model

# Save model and optimizer state
save_model(model, optimizer, log_name)
```

#### Loading Pre-trained Models

```python
from hydragnn.utils.model import load_existing_model_config

# Load existing model
load_existing_model_config(model, config["Training"], optimizer=optimizer)
```

---

## Advanced Features

### DeepSpeed Integration

#### DeepSpeed Configuration

```json
{
    "NeuralNetwork": {
        "ds_config": {
            "train_batch_size": 64,
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": true,
                "reduce_scatter": true,
                "overlap_comm": false,
                "contiguous_gradients": true
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-3,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-3,
                    "warmup_num_steps": 1000
                }
            }
        }
    }
}
```

#### Running with DeepSpeed

Initialize the model and optimizer with `deepspeed.initialize`, then pass the
resulting engine and explicit loaders to `train_validate_test` with
`use_deepspeed=True`. See `examples/ogb/train_gap.py` for the complete setup;
HydraGNN no longer hides this initialization inside a convenience wrapper.

### Infrequent synchronization with post-local SGD

Ordinary DDP averages gradients on every backward pass. HydraGNN can instead
use PyTorch's native post-local-SGD components to reduce global collective
frequency after a synchronized warm-up:

```json
{
    "NeuralNetwork": {
        "Training": {
            "LocalSGD": {
                "enabled": true,
                "warmup_steps": 100,
                "synchronization_period": 4,
                "optimizer_state_policy": "local",
                "optimizer_state_bucket_bytes": 26214400
            }
        }
    }
}
```

Both integer values count optimizer steps, not epochs:

- `warmup_steps` must be nonnegative. Before this many steps, DDP globally
  averages gradients after every backward pass.
- `synchronization_period` must be positive. After warm-up, gradient
  all-reduce is disabled and every rank applies its optimizer locally. Model
  parameters are globally averaged at the first local step and every stated
  number of local steps thereafter.

The default is `{"enabled": false}`, which retains conventional synchronous
DDP. A period of `1` averages parameters after every local optimizer step and
therefore does not reduce the number of global parameter collectives; values
greater than `1` provide the communication reduction.

This option changes the optimization algorithm rather than merely optimizing
DDP communication. Replicas intentionally diverge between parameter averages.
PyTorch's `PostLocalSGDOptimizer` accepts standard optimizers such as SGD,
AdamW, and RMSprop. `optimizer_state_policy` controls persistent state:

- `"local"` (default) matches native PyTorch post-local SGD. Momentum buffers,
  adaptive moments, and other optimizer history remain rank-local.
- `"synchronize"` applies optimizer-specific reductions whenever model
  parameters are averaged. This costs additional collectives and communication
  but restarts replicas with the same model and optimizer history.

HydraGNN uses the following strict synchronized-state policies:

| Optimizer | State synchronization |
| --- | --- |
| SGD | mean `momentum_buffer` when present |
| Adam / AdamW | mean `exp_avg` and `exp_avg_sq`; require equal `step` |
| Adamax | mean `exp_avg`; elementwise maximum `exp_inf`; require equal `step` |
| Adagrad | mean `sum`; require equal `step` |
| Adadelta | mean `square_avg` and `acc_delta`; require equal `step` |
| RMSprop | mean `square_avg` and optional `momentum_buffer`/`grad_avg`; require equal `step` |

Unknown optimizers or state keys fail explicitly under `"synchronize"` rather
than being silently ignored. FusedLAMB is currently unsupported for state
synchronization. `optimizer_state_bucket_bytes` bounds the temporary flat
communication buckets used to combine compatible state tensors; it must be a
positive integer and defaults to 25 MiB. State tensors are bucketed by
reduction, device, and dtype. State layouts and step counters are collectively
checked before tensor reductions.

The local policy communicates less. Larger periods can cause the model and
rank-local optimizer history to become inconsistent after parameter averaging,
especially for adaptive optimizers or non-identically distributed data. The
synchronized policy avoids that mismatch but can substantially increase
communication—for Adam-like methods, two additional parameter-sized moment
buffers are transferred at every synchronization point.

HydraGNN stores the model-averager step in optimizer checkpoints so resumed
runs retain the warm-up and averaging schedule. All ranks must execute the same
number of optimizer steps. The setting only changes training-time gradient and
parameter synchronization; validation/test reductions, metric collectives,
data-preprocessing collectives, and checkpoint coordination are unaffected.
If an epoch ends between scheduled averages, HydraGNN performs one conditional
parameter average before validation/checkpointing; no extra collective is
issued when the last optimizer step already performed the scheduled average.
With synchronized optimizer state, a single-rank checkpoint is sufficient
because all replicas have identical state at checkpoint time. With local state,
the checkpoint contains the checkpoint-writing rank's optimizer history; all
ranks load that history on resume and then diverge during later local steps.

Post-local SGD currently supports ordinary PyTorch DDP only. HydraGNN rejects
combinations with FSDP, DeepSpeed, `SyncBatchNorm`, task/model-parallel wrappers,
or `ZeroRedundancyOptimizer` rather than silently applying different semantics.
Pass the full configuration to the distributed wrapper so it can configure the
DDP communication hook before checkpoint loading:

```python
model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
    model, optimizer, verbosity, config=config
)
```

### FSDP (Fully Sharded Data Parallel) Integration

Pytorch's FSDP (Fully Sharded Data Parallel) provides functionality similar to DeepSpeed ZeRO. 

FSDP can be turned on or off using the `HYDRAGNN_USE_FSDP` env:

```bash
# Disable FSDP (default)
export HYDRAGNN_USE_FSDP=0
# Enable FSDP
export HYDRAGNN_USE_FSDP=1
```

The level of ZeRO optimization can be chosen by `HYDRAGNN_FSDP_STRATEGY` env:

```bash
# Choose one of them. FULL_SHARD is default.
export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD
export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
export HYDRAGNN_FSDP_STRATEGY=NO_SHARD
```

The FSDP implementation version can be selected using `HYDRAGNN_FSDP_VERSION`:

```bash
# FSDP v1 (default, FlatParameter-based wrapper)
export HYDRAGNN_FSDP_VERSION=1

# FSDP v2 (composable fully_shard path)
export HYDRAGNN_FSDP_VERSION=2
```

Notes:
- FSDP v2 currently supports `HYDRAGNN_FSDP_STRATEGY` values `FULL_SHARD` and `SHARD_GRAD_OP`.
- Multi-branch model-parallel mode (`MultiTaskModelMP`) supports `HYDRAGNN_FSDP_VERSION=2`.
- For task-parallel runs (`--task_parallel`) without `--use_devicemesh`, branch groups are split proportionally by dataset size by default.
    - Controlled by `HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT` (default `1`).
    - Set `HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT=0` to force uniform split.

### Multi-Branch Training

Multi-branch training allows training on multiple datasets with different data distributions simultaneously.

#### Multi-Branch Configuration

```python
# Run multi-branch training
python examples/multibranch/train.py \
    --multi \
    --multi_model_list=dataset1.bp,dataset2.bp,dataset3.bp \
    --inputfile=multibranch_config.json
```

#### Branch-Specific Output Heads

```json
{
    "output_heads": {
        "graph": [
            {
                "type": "branch-0",
                "architecture": {
                    "num_sharedlayers": 2,
                    "dim_sharedlayers": 50,
                    "num_headlayers": 3,
                    "dim_headlayers": [100, 100, 100]
                }
            },
            {
                "type": "branch-1",
                "architecture": {
                    "num_sharedlayers": 2,
                    "dim_sharedlayers": 50,
                    "num_headlayers": 3,
                    "dim_headlayers": [100, 100, 100]
                }
            }
        ]
    }
}
```

### Hyperparameter Optimization

HydraGNN supports HPO through **DeepHyper** (distributed, scalable) and **Optuna** (TPE, random, CMA-ES samplers). Both are optional dependencies (see `requirements-optional.txt`).

#### DeepHyper

```bash
# Example: multi-dataset HPO with DeepHyper
cd examples/multidataset_hpo
python gfm_deephyper_multi.py
```

#### Optuna

```bash
# Example: QM9 HPO with Optuna
cd examples/qm9_hpo
python qm9_optuna.py
```

See the `examples/qm9_hpo/`, `examples/multidataset_hpo/`, and `examples/multidataset_hpo_sc26/` directories for working HPO examples.

### Global Attention with Transformers

#### GPS (Graph Positional and Structural Attention)

```json
{
    "Architecture": {
        "global_attn_engine": "GPS",
        "global_attn_type": "multihead",
        "global_attn_heads": 8,
        "pe_dim": 16,
        "hidden_dim": 128  // Must be divisible by global_attn_heads
    }
}
```

### Geometric Equivariance

Several architectures support geometric equivariance:

- **EGNN**: E(n)-equivariant GNN (also supports periodic boundary conditions)
- **PaiNN**: Equivariant message passing with scalar and vector channels
- **PNAEq**: Equivariant variant of PNA
- **MACE**: E(3)-equivariant multi-atomic cluster expansion
- **DimeNet**: Directional message passing with angular information

```json
{
    "Architecture": {
        "mpnn_type": "EGNN",
        "equivariance": true,
        "hidden_dim": 128,
        "num_conv_layers": 4
    }
}
```

Equivariance is automatically configured based on the selected `mpnn_type`.

---

## Examples and Use Cases

### 1. Materials Property Prediction

#### Training on preprocessed LSMS-derived data

```bash
# Navigate to LSMS example
cd examples/lsms

# Parse raw LSMS, compile named variables, write splits, and train.
python lsms.py --raw-data ./dataset/FePt_enthalpy --pickle

# Perform only the preprocessing and serialization stage.
python lsms.py --raw-data ./dataset/FePt_enthalpy --pickle --preonly

# Reuse existing prepared splits without parsing raw LSMS again.
python lsms.py --inputfile lsms.json --loadexistingsplit
```

Configuration highlights:
- Keeps LSMS parsing in the application example rather than HydraGNN core
- Supports prepared pickle or ADIOS splits through the same training script
- Predicts free energy, charge density, and magnetic moments
- Uses PNA architecture with 6 convolution layers
- Multi-task learning with graph and node predictions

#### EAM CFG and Ising preprocessing

The EAM and Ising scripts follow the same preprocess/train contract. EAM reads
CFG records through its example-owned field mapping; Ising generates named PyG
samples directly rather than writing and reparsing intermediate text files.

```bash
# Parse CFG, compile the configured named variables, serialize, and train.
python examples/eam/eam.py --raw-data /path/to/cfg-directory --pickle

# Preprocess only, or reuse the resulting splits later.
python examples/eam/eam.py --raw-data /path/to/cfg-directory --pickle --preonly
python examples/eam/eam.py --pickle --loadexistingsplit

# Generate, compile, and serialize Ising samples without training.
python examples/ising_model/train_ising.py --pickle --preonly --natom 3 --cutoff 10

# A later invocation reuses that matching artifact and starts training.
python examples/ising_model/train_ising.py --pickle --natom 3 --cutoff 10
```

Only rank zero performs these example-owned preprocessing steps. Any error is
broadcast before the MPI barrier so other ranks do not hang. `--preonly`
stops after serialization; without it, EAM preprocesses unless
`--loadexistingsplit` is given, while Ising automatically reuses the artifact
matching `--natom`, `--cutoff`, and the selected storage format.

### 2. Molecular Property Prediction

#### QM9 Dataset Example

```bash
# Train on QM9 molecular dataset
cd examples/qm9
python qm9.py --inputfile qm9.json
```

Key features:
- Molecular graph representation
- Multiple molecular properties
- Rotational invariance for molecules

### 3. Large-Scale Multi-Dataset Training

#### Multi-Branch Training Example

```bash
# Train on multiple large datasets simultaneously
cd examples/multibranch

# Run with SLURM on HPC systems
sbatch SC25-multibranch-omnistat.sh
```

Features:
- Trains on 5 different datasets simultaneously
- Automatic load balancing across datasets
- Optimized for supercomputing environments

### 4. Force Prediction with Energy Conservation

#### Energy and Forces Example

```json
{
    "Training": {
        "compute_grad_energy": true
    },
    "Variables": {
        "inputs": [
            {"name": "atomic_numbers", "level": "node", "dim": 1},
            {"name": "pos", "level": "node", "dim": 3, "role": "position"}
        ],
        "outputs": [
            {"name": "energy", "level": "graph", "dim": 1},
            {"name": "forces", "level": "node", "dim": 3}
        ]
    }
}
```

### 5. Custom Dataset Integration

#### Creating Custom Data Loaders

```python
import torch_geometric.data as pygdata

# Parsing the external format is application code. Its result must be a
# collection of PyG objects whose attribute names match the JSON Variables.
dataset = [
    pygdata.Data(
        node_features=sample.node_features,
        edge_index=sample.edge_indices,
        bond_features=sample.edge_features,
        pos=sample.positions,
        target=sample.target,
    )
    for sample in parse_my_format(source)
]
```

### 6. High-Performance Computing Deployment

#### SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH -A PROJECT_ID
#SBATCH -J HydraGNN_training
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 16
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

# Load modules and environment
module load python/3.11
source hydragnn_env/bin/activate

# Set environment variables
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

# Run training
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) \
     -c7 --gpus-per-task=1 --gpu-bind=closest \
     python -u train.py --inputfile config.json
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues

**Problem**: Out of memory errors during training
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
```bash
# Reduce batch size
{
    "Training": {
        "batch_size": 16  # Reduce from 32 or 64
    }
}

# Disable data loading workers
export HYDRAGNN_NUM_WORKERS=0

# Enable gradient checkpointing
{
    "Training": {
        "conv_checkpointing": true
    }
}
```

#### 2. Distributed Training Issues

**Problem**: Hanging during distributed initialization

**Solutions**:
```bash
# Check MPI installation
mpirun --version

# Verify GPU binding
nvidia-smi

# Debug with verbose output
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

#### 3. Data Loading Problems

**Problem**: Slow data loading or I/O bottlenecks

**Solutions**:
```bash
# Use shared memory
opt = {"preload": True, "shmem": True}

```

#### 4. Convergence Issues

**Problem**: Model not converging or poor performance

**Solutions**:
```json
{
    "Training": {
        "learning_rate": 1e-4,  // Reduce learning rate
        "num_epoch": 500,       // Increase epochs
        "EarlyStopping": false  // Disable early stopping
    },
    "Architecture": {
        "hidden_dim": 256,      // Increase model capacity
        "num_conv_layers": 6    // Add more layers
    }
}
```

#### 5. Configuration Errors

**Problem**: Invalid configuration parameters

**Solutions**:
```python
# Update configuration with defaults for missing fields
from hydragnn.utils.input_config_parsing.config_utils import update_config
update_config(config, train_loader, val_loader, test_loader)

# Check required fields
required_fields = ["Dataset", "NeuralNetwork", "Verbosity"]
for field in required_fields:
    assert field in config, f"Missing required field: {field}"
```

### Debugging Tools

#### Enable Verbose Logging

```json
{
    "Verbosity": {
        "level": 4  // Maximum verbosity
    }
}
```

#### Memory Profiling

```python
from hydragnn.utils.distributed import print_peak_memory

# Monitor memory usage
print_peak_memory(verbosity_level, "After model creation")
```

#### Performance Profiling

```python
from hydragnn.utils.profiling_and_tracing.time_utils import Timer

timer = Timer("data_loading")
timer.start()
# ... your code ...
timer.stop()
```

---

## Best Practices

### 1. Data Management

#### Efficient Data Storage
- Use ADIOS2 format for datasets larger than 1GB
- Implement data caching for frequently accessed datasets
- Use shared memory when training on single nodes

```python
# Recommended data pipeline
1. Raw data → Preprocessing → ADIOS2 format
2. ADIOS2 → Distributed loading → Training
```

#### Data Preprocessing Tips
- Normalize input features to [0, 1] or [-1, 1] range
- Use compositional stratified splitting for materials datasets
- Implement data validation checks

### 2. Model Design

#### Architecture Selection Guidelines

| Dataset Type | Recommended Architecture | Key Parameters |
|--------------|-------------------------|----------------|
| Molecules | EGNN, SchNet, PaiNN, DimeNet | equivariance=true |
| Crystals | CGCNN, MACE, EGNN | periodic_boundary_conditions=true |
| Interatomic Potentials | MACE, PaiNN, SchNet | enable_interatomic_potential=true |
| General | PNA, PNAPlus | Balanced performance |
| Large graphs | GPS with attention | global_attn_engine="GPS" |

#### Hyperparameter Tuning
```python
# Recommended starting points
{
    "hidden_dim": 128,        # Start with 128, scale up if needed
    "num_conv_layers": 4,     # 3-6 layers typically optimal
    "learning_rate": 1e-3,    # Conservative starting point
    "batch_size": 32          # Balance memory and convergence
}
```

### 3. Scalability Optimization

#### Distributed Training Guidelines

```bash
# Optimal resource allocation
- 1 GPU per MPI rank
- 4-8 CPU cores per GPU
- 16-32 GB RAM per GPU

# Environment optimization
export OMP_NUM_THREADS=4-8
export HYDRAGNN_NUM_WORKERS=0  # For HPC environments
export HYDRAGNN_AGGR_BACKEND=mpi
```

#### Performance Monitoring

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Profile memory usage
export CUDA_LAUNCH_BLOCKING=1

# Monitor network I/O
export NCCL_DEBUG=INFO
```

### 4. Training Optimization

#### Convergence Strategies
1. **Start Simple**: Begin with basic architecture
2. **Gradual Complexity**: Add features incrementally
3. **Learning Rate Schedule**: Use ReduceLROnPlateau
4. **Early Stopping**: Prevent overfitting

#### Multi-Task Learning
```json
{
    "task_weights": [1.0, 10.0],  // Weight important tasks higher
    "loss_function_type": "mae",  // Often more stable than MSE
    "batch_size": 32              // Larger batches for stable gradients
}
```

### 5. Production Deployment

#### Model Checkpointing
```json
{
    "Training": {
        "Checkpoint": true,
        "checkpoint_warmup": 10,  // Start after initial convergence
        "continue": 0             // For new training runs
    }
}
```

#### Reproducibility
```python
# Set random seeds
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

# Deterministic algorithms (may impact performance)
torch.use_deterministic_algorithms(True)
```

### 6. Error Handling and Logging

#### Comprehensive Logging Setup
```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hydragnn.log'),
        logging.StreamHandler()
    ]
)
```

#### Graceful Error Handling
```python
try:
    hydragnn.train.train_validate_test(
        model, optimizer, train_loader, val_loader, test_loader,
        writer, scheduler, config["NeuralNetwork"], log_name, verbosity
    )
except Exception as e:
    logging.error(f"Training failed: {e}")
    # Save partial results
    # Clean up resources
    raise
```

---

## Conclusion

This manual provides comprehensive guidance for using HydraGNN effectively. For additional support:

- **GitHub Issues**: Report bugs and request features
- **Wiki**: Detailed technical documentation
- **Examples**: Working code samples in the `examples/` directory
- **Community**: Connect with other HydraGNN users

HydraGNN continues to evolve with new features and optimizations. Stay updated with the latest releases and documentation updates.

---

*Last updated: April 2026*
*Version: Compatible with HydraGNN v5.0*
