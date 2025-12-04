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
python setup.py install
```

### Verification

Test your installation:
```bash
python -c "import hydragnn; print('HydraGNN installed successfully')"
```

---

## Data Pre-processing

HydraGNN supports multiple data formats and provides comprehensive preprocessing capabilities.

### Supported Data Formats

#### 1. Raw Data Formats

**LSMS Format**: For magnetic materials and alloy datasets
- Used for datasets like FePt, CuAu, FeSi
- Contains atomic positions, magnetic moments, and energies
- Automatically processed into graph representations

**CFG Format**: Configuration files for atomic structures
- Standard format for molecular dynamics simulations
- Contains atomic coordinates and properties

#### 2. Serialized Formats

**Pickle Format**: Python-native serialization
```python
# Loading pickle datasets
from hydragnn.utils.datasets.serializeddataset import SerializedDataset
dataset = SerializedDataset(basedir, dataset_name, "trainset")
```

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
        "path": {"total": "./dataset/FePt_enthalpy"},
        "format": "LSMS",
        "compositional_stratified_splitting": true,
        "node_features": {
            "name": ["num_of_protons", "charge_density", "magnetic_moment"],
            "dim": [1, 1, 1],
            "column_index": [0, 5, 6]
        },
        "graph_features": {
            "name": ["free_energy_scaled_num_nodes"],
            "dim": [1],
            "column_index": [0]
        }
    }
}
```

#### Programmatic Data Loading

```python
from hydragnn.preprocess.load_data import dataset_loading_and_splitting

# Load and split data automatically
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

### Custom Data Preprocessing

#### Writing Custom Data Loaders

```python
from hydragnn.preprocess.raw_dataset_loader import RawDatasetLoader

class CustomRawDataLoader(RawDatasetLoader):
    def __init__(self, config):
        super().__init__(config)
    
    def load_raw_data(self):
        # Implement custom data loading logic
        pass
    
    def get_data(self):
        # Return processed torch_geometric.data.Data objects
        return self.dataset
```

#### Data Serialization for Performance

```python
from hydragnn.utils.datasets.serializeddataset import SerializedWriter

# Save processed datasets for fast loading
writer = SerializedWriter(dataset, basedir, dataset_name, "trainset")
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

#### Variables of Interest

```json
{
    "Variables_of_interest": {
        "input_node_features": [0, 1, 2],     // Input feature indices
        "output_names": ["energy", "forces"], // Output property names
        "output_index": [0, 1],               // Output target indices
        "type": ["graph", "node"],            // Prediction types
        "output_dim": [1, 3],                 // Output dimensions
        "denormalize_output": true            // Whether to denormalize predictions
    }
}
```

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

# Simple distributed training
hydragnn.run_training("config.json")
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

#### Gradient Computation for Forces

```json
{
    "Training": {
        "compute_grad_energy": true  // Compute forces from energy gradients
    }
}
```

### Training Execution

#### High-Level Training Interface

```python
import hydragnn

# Train with configuration file
hydragnn.run_training("examples/lsms/lsms.json")

# Train with configuration dictionary
config = {...}  # Configuration dictionary
hydragnn.run_training(config)
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
    log_name=log_name,
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

```python
# Enable DeepSpeed during training
hydragnn.run_training(config, use_deepspeed=True)
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

#### Ray Tune Integration

```python
# HPO configuration example
from hydragnn.utils.hpo import run_hpo

# Run hyperparameter optimization
run_hpo(config_file="hpo_config.json")
```

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

#### EGNN for Equivariant Predictions

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

---

## Examples and Use Cases

### 1. Materials Property Prediction

#### LSMS Dataset Example

```bash
# Navigate to LSMS example
cd examples/lsms

# Train on magnetic materials dataset
python lsms.py --inputfile lsms.json
```

Configuration highlights:
- Predicts free energy, charge density, and magnetic moments
- Uses PNA architecture with 6 convolution layers
- Multi-task learning with graph and node predictions

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
    "Variables_of_interest": {
        "output_names": ["energy", "forces"],
        "type": ["graph", "node"]
    }
}
```

### 5. Custom Dataset Integration

#### Creating Custom Data Loaders

```python
from hydragnn.preprocess.raw_dataset_loader import RawDatasetLoader
import torch_geometric.data as pygdata

class MyCustomLoader(RawDatasetLoader):
    def __init__(self, config):
        super().__init__(config)
        
    def load_raw_data(self):
        # Load your custom data format
        raw_data = self.load_my_format()
        
        # Convert to PyTorch Geometric format
        self.dataset = []
        for sample in raw_data:
            data = pygdata.Data(
                x=sample.node_features,
                edge_index=sample.edge_indices,
                edge_attr=sample.edge_features,
                y=sample.targets,
                pos=sample.positions
            )
            self.dataset.append(data)
    
    def load_my_format(self):
        # Implement your data loading logic
        pass
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
module load python/3.9
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
# Validate configuration
from hydragnn.utils.input_config_parsing.config_utils import validate_config
validate_config(config)

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
| Molecules | EGNN, SchNet, PAINN | equivariance=true |
| Crystals | CGCNN, MACE | periodic_boundary_conditions=true |
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
    hydragnn.run_training(config)
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

*Last updated: [Current Date]*
*Version: Compatible with HydraGNN v1.x*
