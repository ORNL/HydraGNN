# E(3) Equivariant GPS Layer Implementation

## Overview

This document explains the implementation of a truly equivariant Graph GPS layer that maintains E(3) equivariance - meaning the model respects rotations and translations in 3D space.

## Key Principles of E(3) Equivariance

### 1. **Scalar (Invariant) Features**
- Remain unchanged under rotations and translations
- Examples: distances, angles, norms
- Operations: Can use any neural network operations (MLPs, attention, etc.)

### 2. **Vector (Equivariant) Features** 
- Transform correctly under rotations: `f(R @ x) = R @ f(x)`
- Examples: positions, velocities, forces
- Constraints: Cannot use element-wise nonlinearities, must preserve transformation properties

## Implementation Details

### Architecture Components

1. **Separate Processing Streams**:
   ```python
   # Scalar features: can use full neural network operations
   scalar_mlp = Sequential(Linear, ReLU, Dropout, Linear)
   
   # Vector features: linear operations only (no nonlinearities)
   vector_mlp = Sequential(Linear, Dropout)  # No ReLU!
   ```

2. **Position Encoding**:
   ```python
   # Invariant positional information (scalar)
   pos_norm = torch.norm(positions, dim=1, keepdim=True)
   scalar_features += pos_invariant_proj(pos_norm)
   
   # Equivariant vector initialization
   vector_features = positions.unsqueeze(-1).expand(-1, -1, channels)
   ```

3. **Equivariance-Preserving Operations**:
   - **Allowed**: Linear transformations, element-wise multiplication with scalars
   - **Forbidden**: Element-wise nonlinearities on vectors (ReLU, tanh, etc.)
   - **Gating**: Use scalar features to gate vector features

### Key Implementation Features

#### 1. **Vector Feature Initialization**
```python
# Create vector features from positions
vector_feat = positions.unsqueeze(-1).expand(-1, -1, channels) * 0.01
```
- Initializes vector features aligned with position directions
- Small scaling factor (0.01) prevents dominance over learned features

#### 2. **Scalar-Gated Vector Processing**
```python
# Use scalar features to create gates for vector features
vector_gates = torch.sigmoid(vector_gate(scalar_out))
vector_out = vector_feat * vector_gates  # Preserves equivariance
```
- Scalar gates control vector feature magnitude
- Element-wise multiplication preserves equivariance

#### 3. **Dimension-wise Vector Processing**
```python
# Process each spatial dimension separately
for i in range(3):
    vector_out_transformed[:, i, :] = vector_mlp(vector_out[:, i, :])
```
- Applies same linear transformation to each spatial dimension
- Maintains equivariance property

#### 4. **Equivariant-Safe Normalization**
```python
# Normalize only along feature dimension, not spatial dimensions
for i in range(3):
    vector_out_norm[:, i, :] = vector_norm(vector_out[:, i, :])
```
- LayerNorm applied to feature dimension only
- Preserves spatial transformation properties

## Verification of Equivariance

### Mathematical Property
For a rotation matrix R, the equivariance property requires:
```
f(scalar_features, R @ positions) = (scalar_features, R @ f_vector_output)
```

### Why This Implementation Works

1. **Scalar Features**: 
   - Only use invariant quantities (position norms)
   - Unaffected by rotations ✓

2. **Vector Features**:
   - Linear operations preserve equivariance ✓
   - No element-wise nonlinearities ✓
   - Gating with scalars preserves equivariance ✓

3. **Attention Mechanism**:
   - Applied only to scalar features ✓
   - Maintains invariance ✓

## Comparison with Previous Implementation

### Before (Pseudo-Equivariant):
```python
# Only used invariant information
pos_norm = torch.norm(positions, dim=1, keepdim=True)
inv_node_feat += pos_proj(pos_norm)
return inv_node_feat, positions  # Just passed through positions
```

### After (Truly Equivariant):
```python
# Maintains both scalar and vector features
scalar_out = process_scalar_features(inv_node_feat, pos_norm)
vector_out = process_vector_features(positions, scalar_gates)
return scalar_out, vector_out  # Both properly transformed
```

## Usage Guidelines

### When to Use This Layer
- ✅ Molecular property prediction requiring geometric awareness
- ✅ Force prediction (vectors must transform correctly)
- ✅ Crystal structure analysis
- ✅ Any task where rotational equivariance is important

### Integration Notes
- Returns both scalar and vector features
- Vector features should be used for equivariant predictions
- Scalar features can be used for invariant predictions
- Compatible with other equivariant layers (EGNN, PaiNN, etc.)

## Performance Considerations

### Memory Usage
- Vector features: `[N, 3, channels]` vs scalar `[N, channels]`
- Approximately 3x memory increase for vector features
- Justified by improved geometric representation

### Computational Cost
- Additional vector processing overhead
- Separate normalization and MLP operations
- Still efficient for typical molecular system sizes

## Future Enhancements

1. **Higher-Order Features**: Extend to rank-2 tensors (stress, strain)
2. **Spherical Harmonics**: More sophisticated angular representations
3. **Attention on Vectors**: Develop equivariant attention mechanisms
4. **Periodic Boundary Conditions**: Handle crystal systems properly

This implementation provides a solid foundation for truly equivariant graph neural networks while maintaining compatibility with the existing HydraGNN framework.