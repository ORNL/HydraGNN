# HydraGNN

Scalable PyTorch Implementation of Multi-Headed Graph Neural Networks

<img width="1408" height="736" alt="HydraGNN-Logo" src="https://github.com/user-attachments/assets/eb550b3a-23a4-4736-8de2-ec737c7ae37a" />

<img src="https://github.com/ORNL/HydraGNN/assets/2488656/a6d5369b-2a70-4eee-aa39-b2cf8dedf262" alt="HydraGNN_QRcode" width="300" />


## Capabilities

<img src="images/HydraGNN-Overview.png" alt="HydraGNN Overview" width="1100" />

- **Multi-headed Prediction** for graph and node-level properties  
- **Distributed Data Parallelism** at supercomputing level
- **Convolutional Layers** as a hyperparameter  
- **Geometric Equivariance** in convolution and prediction  
- **Global Attention**

### Optional graph-level conditioning
- Enable with `NeuralNetwork.Architecture.use_graph_attr_conditioning` (off by default) and choose mode via `graph_attr_conditioning_mode` (`"concat_node"` default, `"film"`, or `"fuse_pool"`).
- `concat_node` (default) appends `graph_attr` to node embeddings and projects back to hidden dimension; FiLM scales/shifts invariant channels per graph; `fuse_pool` fuses `graph_attr` with the pooled graph embedding before the heads.
- Conditioning consumes `data.graph_attr` and requires those global attributes to be rotation/translation invariant; providing orientation-dependent values will break equivariance by design.


## Dependencies

To install required packages with only basic capability (`torch`,
`torch_geometric`, and related packages)
and to serialize+store the processed data for later sessions (`pickle5`):

**Recommended approach - standard installation:**
```bash
# Install all core dependencies (base + PyTorch + PyTorch Geometric)
pip install -r requirements.txt

# Or use the installation script
./install_dependencies.sh all
```

**Alternative approach for reproducible installation:**
```bash
# Use the provided installation script
./install_dependencies.sh

# Or install manually with consistent settings:
pip install --no-build-isolation -v -r requirements.txt
```

**Modular installation (choose what you need):**
```bash
# Base only (scientific computing, materials science, visualization)
pip install -r requirements-base.txt

# Add PyTorch
pip install -r requirements-torch.txt

# Add PyTorch Geometric  
pip install -r requirements-pyg.txt

# Add optional features (HPO, FAIRChem, etc.)
pip install -r requirements-optional.txt
```

If you plan to modify the code, include packages for formatting (`black`) and
testing (`pytest`) the code:
```bash
pip install -r requirements-dev.txt
# Or with the script:
./install_dependencies.sh all dev
```

Detailed dependency installation instructions are available on the
[Wiki](https://github.com/ORNL/HydraGNN/wiki/Install)


## Installation

After checking out HydraGNN, we recommend installing HydraGNN in
developer mode so that you can use the files in your current location
and update them if needed:
```bash
python -m pip install -e .
```

Or, simply type the following in the HydraGNN directory:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Alternatively, if you have no plans to update, you can install
HydraGNN in your python tree as a static package:
```bash
python setup.py install
```

## Quick Start

**For detailed instructions, configuration options, and advanced features, please refer to the [Comprehensive User Manual](USER_MANUAL.md).**

Below are the four main functionalities for running the code.
1. Training a model, including continuing from a previously trained model using configuration options:
```python
import hydragnn
hydragnn.run_training("examples/configuration.json")
```
2. Saving a model state:
```python
import hydragnn
model_name = model_checkpoint.pk
hydragnn.save_model(model, optimizer, model_name, path="./logs/")
```
3. Loading a model state:
```python
import hydragnn
model_name = model_checkpoint.pk
hydragnn.load_existing_model(model, model_name, path="./logs/")
```
4. Making predictions from a previously trained model:
```python
import hydragnn
hydragnn.run_prediction("examples/configuration.json", model)
```
The `run_training` and `run_predictions` functions are convenient routines that encapsulate all the steps of the training process (data generation, data pre-processing, training of HydraGNN models, and use of trained HydraGNN models for inference). Both `run_training` and `run_predictions` require a JSON input file for configurable options. The `save_model` and `load_model` functions store and retrieve model checkpoints for continued training and subsequent inference. 

Example scripts showing data pre-processing, training, and inference for specific datasets are provided in the `examples/` folder. See the [User Manual](USER_MANUAL.md) for comprehensive documentation.

### Datasets

Built-in examples are provided in the `examples/` directory, covering various domains:
- **Molecular datasets**: QM9, ANI-1x, QM7x, ZINC
- **Materials datasets**: Alexandria, Open Catalyst (2020/2022), Open Materials 2024, Open Molecules 2025
- **Physics simulations**: Lennard-Jones, Ising Model, Power Grid

One source of data to create HydraGNN surrogate predictions is DFT output on the OLCF Constellation: https://doi.ccs.ornl.gov/

Detailed instructions are available on the [Wiki](https://github.com/ORNL/HydraGNN/wiki/Datasets).

### Configuration

**For complete configuration documentation, see the [User Manual](USER_MANUAL.md).**

HydraGNN uses a JSON configuration file to specify all aspects of model architecture, training, and data processing. Modern HydraGNN configurations use a feature-based approach for defining inputs and outputs.

#### Key Configuration Sections

**Dataset Configuration:**
```json
{
    "Dataset": {
        "name": "my_dataset",
        "format": "XYZ",  // or "LSMS", "Pickle", "ADIOS2"
        "node_features": {
            "name": ["atomic_number", "coordinates"],
            "dim": [1, 3],
            "column_index": [0, 1]
        },
        "graph_features": {
            "name": ["energy"],
            "dim": [1],
            "column_index": [0]
        }
    }
}
```

**Neural Network Architecture:**
```json
{
    "NeuralNetwork": {
        "Architecture": {
            "mpnn_type": "EGNN",  // See supported architectures below
            "hidden_dim": 64,
            "num_conv_layers": 4,
            "output_heads": {
                "graph": {
                    "num_headlayers": 2,
                    "dim_headlayers": [50, 25]
                }
            }
        }
    }
}
```

**Supported MPNN Architectures:**
- `CGCNN`, `DimeNet`, `EGNN`, `GAT`, `GIN`, `MACE`, `MFC`, `PAINN`, `PNA`, `PNAPlus`, `PNAEq`, `SAGE`, `SchNet`

**Training Configuration:**
```json
{
    "NeuralNetwork": {
        "Training": {
            "num_epoch": 100,
            "batch_size": 32,
            "Optimizer": {
                "type": "Adam",
                "learning_rate": 0.001
            }
        }
    }
}
```

**Variables of Interest (Modern Approach):**

HydraGNN now uses a declarative feature configuration with `node_features` and `graph_features` in the `Dataset` section. Each feature specifies:
- `name`: Descriptive name(s) 
- `dim`: Dimensionality of each feature
- `column_index`: Which columns in `data.x` or `data.y` to extract (optional but recommended)
- `role`: `"input"` or `"output"` (specified in `Variables_of_interest` section)

Example in `Variables_of_interest` section:
```json
{
    "NeuralNetwork": {
        "Variables_of_interest": {
            "node_features": {
                "atomic_number": {"dim": 1, "role": "input"},
                "forces": {"dim": 3, "role": "output"}
            },
            "graph_features": {
                "energy": {"dim": 1, "role": "output"}
            }
        }
    }
}
```

**Note:** Legacy configurations using `input_node_features` and `output_index` are still supported but not recommended for new projects.

For comprehensive configuration details including:
- Node feature extraction rules and `column_index` usage
- Multi-dataset training
- DeepSpeed integration
- Global attention mechanisms
- Hyperparameter optimization

Please refer to the [User Manual](USER_MANUAL.md).


## Contributing

We encourage you to contribute to HydraGNN! Please check the [guidelines](CONTRIBUTING.md) on how to do so.

## Citations

"HydraGNN: Distributed PyTorch implementation of multi-headed graph convolutional neural networks", Copyright ID#: 81929619  
https://doi.org/10.11578/dc.20211019.2

## Documentation

- **Quick Start**: This README provides basic usage and installation
- **[Comprehensive User Manual](USER_MANUAL.md)**: Detailed guide covering all features, data pre-processing, model construction, scalable data management, and training
- **[Wiki](https://github.com/ORNL/HydraGNN/wiki)**: Additional technical documentation and datasets
- **Examples**: The `examples/` directory contains working configurations for various datasets and use cases
