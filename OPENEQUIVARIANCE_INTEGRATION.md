# OpenEquivariance Integration for HydraGNN MACE

## Overview

This integration adds optional OpenEquivariance optimization support to the MACE model in HydraGNN, providing significant performance improvements for tensor operations while maintaining full backward compatibility through e3nn fallback.

## Features

### 1. Optional Usage
- Controlled via the `enable_openequivariance` boolean parameter in JSON configuration
- Defaults to `False` to ensure backward compatibility
- Can be enabled per-model configuration

### 2. Automatic Detection and Fallback
- Automatically detects OpenEquivariance availability
- Checks for proper CUDA configuration
- Gracefully falls back to e3nn when OpenEquivariance is unavailable
- Provides detailed warning messages for troubleshooting

### 3. Optimized Operations
The following tensor operations are optimized when OpenEquivariance is available:

- **TensorProduct**: Core equivariant tensor product operations
- **SphericalHarmonics**: Spherical harmonic transformations
- **Linear**: Equivariant linear layers
- **einsum**: Einstein summation operations

### 4. Comprehensive Error Handling
- CUDA environment validation
- Module availability checks
- Detailed error messages and warnings
- Graceful degradation to e3nn

## Usage

### Configuration

Add the following parameter to your JSON configuration file:

```json
{
  "NeuralNetwork": {
    "Architecture": {
      "mpnn_type": "MACE",
      "enable_openequivariance": true,
      // ... other MACE parameters
    }
  }
}
```

### Example Configuration

See `examples/LennardJones/LJ_openequivariance_test.json` for a complete example.

### Installation Requirements

1. **Basic Usage (e3nn fallback)**:
   - Standard HydraGNN dependencies
   - e3nn library

2. **Optimized Usage (OpenEquivariance)**:
   - OpenEquivariance library: `pip install openequivariance`
   - CUDA toolkit properly installed
   - `CUDA_HOME` environment variable set

## Implementation Details

### Core Components

1. **`hydragnn/utils/model/openequivariance_utils.py`**
   - Main integration utilities
   - Optimized wrapper classes
   - Availability detection logic

2. **Modified MACE Components**:
   - `MACEStack.py`: Integration with main MACE model
   - `blocks.py`: Optimized tensor operations in MACE blocks
   - `symmetric_contraction.py`: Optimized contraction operations
   - `cg.py`: Optimized Clebsch-Gordan operations

3. **Configuration System**:
   - `config_utils.py`: Parameter handling and defaults
   - `create.py`: Parameter passing to model creation

### Architecture

```
JSON Config → config_utils.py → create.py → MACEStack → Optimized Blocks
     ↓                                            ↓
enable_openequivariance                    openequivariance_utils.py
     ↓                                            ↓
Availability Check ←――――――――――――――――――――→ OpenEquivariance OR e3nn
```

## Error Handling

The integration handles several common scenarios:

1. **OpenEquivariance not installed**:
   ```
   Warning: OpenEquivariance is not available in the environment.
   Please install OpenEquivariance from https://github.com/PASSIONLab/OpenEquivariance
   to enable optimized tensor operations. Falling back to e3nn.
   ```

2. **CUDA not configured**:
   ```
   Warning: OpenEquivariance is installed but CUDA is not properly configured.
   Please ensure CUDA is installed and CUDA_HOME environment variable is set
   to enable OpenEquivariance optimizations. Falling back to e3nn.
   ```

3. **Runtime errors**:
   - Individual operation fallbacks
   - Detailed error logging
   - Continued execution with e3nn

## Testing

### Test Files

1. **`test_openequivariance_basic.py`**: Basic functionality tests
2. **`test_openequivariance_final.py`**: Comprehensive integration tests
3. **`test_openequivariance_integration.py`**: Full model integration tests

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run basic tests
python test_openequivariance_basic.py

# Run comprehensive tests
python test_openequivariance_final.py
```

### Expected Output

With OpenEquivariance properly configured:
```
✅ All tests passed successfully!
🚀 OpenEquivariance is available and will be used for optimization
```

Without OpenEquivariance or with CUDA issues:
```
✅ All tests passed successfully!
🔄 OpenEquivariance not available, using e3nn fallback (fully functional)
```

## Performance Benefits

When OpenEquivariance is properly configured:

- **Faster tensor product operations**: Optimized kernel implementations
- **Improved memory usage**: More efficient tensor operations
- **Better GPU utilization**: CUDA-optimized routines
- **Reduced computational overhead**: Streamlined operation graphs

## Backward Compatibility

- **Default behavior unchanged**: `enable_openequivariance` defaults to `False`
- **Existing configurations work**: No changes needed to existing JSON files
- **Full e3nn fallback**: Identical results when OpenEquivariance unavailable
- **Same API**: No changes to model interface or usage patterns

## Troubleshooting

### Common Issues

1. **"CUDA_HOME environment variable is not set"**
   - Install CUDA toolkit
   - Set `export CUDA_HOME=/path/to/cuda` in your environment

2. **"OpenEquivariance is not available"**
   - Install with: `pip install openequivariance`
   - Ensure compatible versions with PyTorch and CUDA

3. **Model creation fails**
   - Check that `enable_openequivariance` is properly set in configuration
   - Verify JSON syntax is correct
   - Review error messages for specific issues

### Debug Information

Enable verbose logging to see detailed information:
```json
{
  "Verbosity": {
    "level": 2
  }
}
```

## Future Enhancements

Potential areas for expansion:

1. **Additional Operations**: More tensor operations could be optimized
2. **Performance Profiling**: Built-in benchmarking capabilities
3. **Automatic Optimization**: Smart selection of operations to optimize
4. **Memory Management**: Advanced memory optimization strategies

## Contributing

When contributing to the OpenEquivariance integration:

1. Maintain backward compatibility
2. Add comprehensive tests for new functionality
3. Update documentation
4. Ensure graceful fallback behavior
5. Follow existing code patterns and style

## References

- [OpenEquivariance GitHub](https://github.com/PASSIONLab/OpenEquivariance)
- [e3nn Documentation](https://e3nn.org/)
- [HydraGNN Documentation](https://github.com/ORNL/HydraGNN)
- [MACE Paper](https://arxiv.org/abs/2206.07697)