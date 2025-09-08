# HydraGNN Interatomic Potential Enhancements

This document describes the enhanced interatomic potential capabilities added to HydraGNN for improved performance in molecular simulations.

## Overview

The interatomic potential enhancements provide specialized functionality for machine learning-based interatomic potentials (MLIPs), including:

1. **Enhanced Geometric Features**: Improved encoding of interatomic distances, angles, and local environment descriptors
2. **Many-body Interactions**: Three-body interaction terms for capturing angular dependencies
3. **Atomic Environment Descriptors**: Sophisticated local chemical environment descriptions
4. **Force Consistency**: Improved energy-force consistency for molecular dynamics

**Note for MLIPs**: These enhancements are designed to work with node-level atomic energy predictions, which are essential for computing forces via automatic differentiation. Total molecular energies can be obtained by summing the individual atomic energies.

## Usage

### Enabling Interatomic Potential Enhancements

To enable the interatomic potential enhancements, add the following to your configuration:

```json
{
  "NeuralNetwork": {
    "Architecture": {
      "enable_interatomic_potential": true,
      ...
    }
  }
}
```

### Configuration Example

See `examples/interatomic_potential_example.json` for a complete configuration example.

### Programmatic Usage

```python
from hydragnn.models.create import create_model

# Create model with interatomic potential enhancements for MLIPs
model = create_model(
    mpnn_type='SchNet',
    input_dim=1,
    hidden_dim=64,
    output_dim=[1],
    pe_dim=6,
    global_attn_engine='',
    global_attn_type='',
    global_attn_heads=1,
    output_type=['node'],  # Node-level for atomic energies (required for MLIPs)
    output_heads={'node': {'num_sharedlayers': 2, 'dim_sharedlayers': 32, 'num_headlayers': 2, 'dim_headlayers': [32, 16]}},
    activation_function='relu',
    loss_function_type='mse',
    task_weights=[1.0],
    num_conv_layers=4,
    enable_interatomic_potential=True,  # Enable enhancements
    use_gpu=True
)
```

## Features

### 1. Enhanced Geometric Features

The enhanced geometric feature computation includes:

- **Distance Features**: Improved edge vector and distance calculations
- **Local Environment**: Coordination numbers, average distances, and local density
- **Periodic Boundary Conditions**: Support for crystal systems with edge shifts

### 2. Three-body Interactions

Three-body interactions capture angular dependencies between atomic triplets:

- **Angular Information**: Considers angles between bonds from central atoms
- **Neighbor Interactions**: Combines features from neighboring atoms
- **Efficient Computation**: Optimized for computational efficiency

### 3. Atomic Environment Descriptors

Local atomic environment descriptors provide detailed chemical environment information:

- **Coordination Numbers**: Number of neighbors within cutoff distance
- **Average Distances**: Mean distances to neighboring atoms
- **Local Density**: Inverse of average distance for density estimation

### 4. Force Consistency

Enhanced force consistency ensures proper energy-force relationships:

- **Gradient Computation**: Forces computed as negative gradients of energy
- **Energy Conservation**: Improved energy conservation in dynamics
- **Automatic Differentiation**: Leverages PyTorch's autograd for consistent derivatives

## Implementation Details

### InteratomicPotentialMixin

The `InteratomicPotentialMixin` class provides the core MLIP functionality:

```python
class InteratomicPotentialMixin:
    def forward(self, data):
        # 1. Dynamic graph construction for proper gradient flow
        # 2. Standard message passing (respects native architecture features)
        # 3. Node-level prediction enforcement
        
    # Enhanced features (disabled by default to avoid interference):
    def _compute_enhanced_geometric_features(self, data, conv_args):
        # Only used if use_enhanced_geometry=True
        
    def _compute_three_body_interactions(self, node_features, data, conv_args):
        # Only used if use_three_body_interactions=True
        
    def _apply_atomic_environment_descriptors(self, node_features, conv_args):
        # Only used if use_atomic_environment_descriptors=True
```

### InteratomicPotentialBase

The `InteratomicPotentialBase` class combines the mixin with the standard HydraGNN Base model:

```python
class InteratomicPotentialBase(InteratomicPotentialMixin, Base):
    # HydraGNN model with essential MLIP capabilities
    # Enhanced features disabled by default to avoid interference
    # with native message passing architectures (MACE, DimeNet++, etc.)
```

### Compatibility with Message Passing Architectures

**Important**: This implementation does NOT interfere with native message passing features:

- **MACE**: Already includes n-body interactions and equivariant geometric features
- **DimeNet++**: Already computes three-body interactions and spherical basis functions  
- **Simple architectures**: Can optionally enable enhanced features if needed

To enable enhanced features for simple architectures:
```python
model = InteratomicPotentialBase(
    use_enhanced_geometry=True,           # For basic architectures only
    use_three_body_interactions=True,     # For basic architectures only  
    use_atomic_environment_descriptors=True  # For basic architectures only
)
```

## Benefits for Molecular Simulations

1. **Essential MLIP Requirements**: Dynamic graph construction and node-level predictions
2. **Architecture Compatibility**: Works with any HydraGNN message passing layer
3. **Proper Gradient Flow**: Graph construction within forward pass for force calculations
3. **Chemical Awareness**: Atomic environment descriptors capture local chemical environments
4. **Computational Efficiency**: Optimized implementations for large-scale simulations

## Compatibility

The interatomic potential enhancements are compatible with:

- All HydraGNN model architectures (SchNet, DimeNet, PAINN, MACE, etc.)
- **Node-level predictions for atomic energies (primary for MLIPs)**
- Graph-level predictions (can be derived by summing atomic energies)
- Energy and force training
- Periodic and non-periodic systems

## Examples

### Atomic Energy Prediction for MLIPs

```python
# Configure for atomic energy prediction (required for MLIPs)
config = {
    "Architecture": {
        "mpnn_type": "SchNet",
        "enable_interatomic_potential": True,
        "output_heads": {
            "node": {
                "num_sharedlayers": 2,
                "dim_sharedlayers": 32,
                "num_headlayers": 1,
                "dim_headlayers": [1]
            }
        }
    },
    "Variables_of_interest": {
        "output_names": ["atomic_energy"],
        "output_dim": [1],
        "type": ["node"]  # Node-level for atomic energies
    }
}
```

### Force Prediction for Atoms

```python
# Configure for atomic force prediction
config = {
    "Architecture": {
        "mpnn_type": "PAINN",
        "enable_interatomic_potential": True,
        "output_heads": {
            "node": {
                "num_headlayers": 2,
                "dim_headlayers": [64, 3],
                "type": "mlp"
            }
        }
    },
    "Variables_of_interest": {
        "output_names": ["forces"],
        "output_dim": [3],
        "type": ["node"]
    }
}
```

## Future Enhancements

Potential future improvements include:

1. **Higher-order interactions**: Four-body and beyond
2. **Specialized descriptors**: SOAP, ACSF, and other molecular descriptors
3. **Uncertainty quantification**: Improved uncertainty estimates for predictions
4. **Multi-scale modeling**: Integration with classical force fields

## References

- [SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/abs/1706.08566)
- [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123)
- [E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials](https://arxiv.org/abs/2101.03164)