# OpenEquivariance Integration in HydraGNN

This document describes the integration of OpenEquivariance for accelerated Clebsch-Gordon tensor products in HydraGNN's MACE models.

## Overview

OpenEquivariance is a CUDA and HIP kernel generator for Clebsch-Gordon tensor products, providing significant performance improvements over e3nn's tensor product implementations. HydraGNN now supports OpenEquivariance as an optional dependency that can provide up to an order of magnitude acceleration for MACE models.

## Installation

### Standard Installation (e3nn only)
```bash
pip install -r requirements-torch.txt  # Installs e3nn
```

### With OpenEquivariance Acceleration
```bash
pip install -r requirements-torch.txt
pip install -r requirements-optional.txt  # Includes OpenEquivariance
```

Or install OpenEquivariance directly:
```bash
pip install openequivariance
```

**Note**: OpenEquivariance requires CUDA and works best on GPU. For CPU-only environments, HydraGNN will automatically fall back to e3nn.

## Usage

### Automatic Backend Selection

By default, HydraGNN will automatically detect and use OpenEquivariance if available:

```python
from hydragnn.models import MACEStack

# Will automatically use OpenEquivariance if available
model = MACEStack(config)
```

### Manual Backend Control

You can control which backend to use through the compatibility layer:

```python
from hydragnn.utils.model.equivariance_compat import set_default_backend, get_backend_info

# Check available backends
info = get_backend_info()
print(f"Available backends: {info}")

# Force e3nn backend
set_default_backend("e3nn")

# Force OpenEquivariance (will fall back to e3nn if not available)
set_default_backend("openequivariance") 

# Automatic selection (default)
set_default_backend("auto")
```

### Using the Compatibility TensorProduct

The compatibility wrapper can be used directly:

```python
from hydragnn.utils.model.equivariance_compat import TensorProduct
from e3nn import o3

# Create tensor product with automatic backend selection
tp = TensorProduct(
    irreps_in1=o3.Irreps("1x1e"),
    irreps_in2=o3.Irreps("1x1e"), 
    irreps_out=o3.Irreps("1x0e + 1x2e"),
    shared_weights=False,
    internal_weights=False
)

# Force specific backend
tp_e3nn = TensorProduct(..., use_openequivariance=False)
tp_oeq = TensorProduct(..., use_openequivariance=True)
```

## Performance Benefits

OpenEquivariance provides several performance advantages:

1. **Accelerated Tensor Products**: Up to 10x faster Clebsch-Gordon tensor products on GPU
2. **Optimized Memory Usage**: More efficient memory layouts for large batch sizes
3. **Fused Operations**: Combined tensor product and message passing operations

### When to Use OpenEquivariance

OpenEquivariance is most beneficial when:
- Training or inference on GPU hardware
- Using MACE models with high-order tensor products (correlation > 2)
- Working with large batch sizes or complex irrep structures
- Performance is critical

### Fallback to e3nn

The system automatically falls back to e3nn when:
- OpenEquivariance is not installed
- Running on CPU-only systems
- Specific tensor product configurations are not supported by OpenEquivariance
- OpenEquivariance initialization fails

## Technical Details

### Compatibility Layer Architecture

The integration consists of three main components:

1. **`equivariance_compat.py`**: Compatibility wrapper providing unified API
2. **Updated MACE modules**: `blocks.py` uses the compatibility wrapper for tensor products
3. **Accelerated CG functions**: `cg.py` provides optimized Clebsch-Gordon calculations

### API Compatibility

The compatibility layer maintains full API compatibility with e3nn while adding OpenEquivariance acceleration:

```python
# Same API as e3nn.o3.TensorProduct
tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions=instructions)
result = tp(x1, x2, weight)
```

### Equivariance Preservation

Both backends preserve SO(3) equivariance. The compatibility layer includes tests to verify that:
- Rotational equivariance is maintained
- Numerical results are consistent (within floating-point precision)
- All MACE model properties are preserved

## Configuration

### Environment Variables

You can control backend selection using environment variables:

```bash
# Force e3nn backend
export HYDRAGNN_EQUIVARIANCE_BACKEND=e3nn

# Force OpenEquivariance backend  
export HYDRAGNN_EQUIVARIANCE_BACKEND=openequivariance

# Automatic selection (default)
export HYDRAGNN_EQUIVARIANCE_BACKEND=auto
```

### Configuration File

In your HydraGNN configuration file, you can specify backend preferences:

```json
{
    "NeuralNetwork": {
        "Architecture": {
            "model_type": "MACE",
            "equivariance_backend": "auto"  // "auto", "e3nn", or "openequivariance"
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **OpenEquivariance not found**: Install with `pip install openequivariance`
2. **CUDA errors**: Ensure CUDA is properly installed and tensors are on GPU
3. **Performance issues**: Check that you're using GPU and batch sizes are appropriate

### Debugging

Enable debug logging to see backend selection:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from hydragnn.utils.model.equivariance_compat import get_backend_info
print(get_backend_info())
```

### Verification

Test the integration with the provided test script:

```bash
cd HydraGNN
python test_integration_simple.py
```

## Examples

### Basic MACE Model

```python
import hydragnn
from hydragnn.models import MACEStack

# Configuration for MACE model
config = {
    "model_type": "MACE",
    "hidden_dim": 128,
    "max_degree": 2,
    "num_layers": 4,
    # OpenEquivariance will be used automatically if available
}

model = MACEStack(config)
```

### Performance Comparison

```python
import time
import torch
from hydragnn.utils.model.equivariance_compat import TensorProduct
from e3nn import o3

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
irreps_in1 = o3.Irreps("8x1e + 4x2e")
irreps_in2 = o3.Irreps("8x1e + 4x2e") 
irreps_out = o3.Irreps("8x0e + 8x1e + 4x2e + 2x3e")
batch_size = 1000

# Create tensor products
tp_e3nn = TensorProduct(..., use_openequivariance=False).to(device)
tp_oeq = TensorProduct(..., use_openequivariance=True).to(device)

# Benchmark (results will vary by hardware)
# OpenEquivariance typically 2-10x faster on GPU
```

## References

- [OpenEquivariance GitHub](https://github.com/PASSIONLab/OpenEquivariance)
- [OpenEquivariance Documentation](https://passionlab.github.io/OpenEquivariance)
- [e3nn Documentation](https://docs.e3nn.org/)
- [MACE Paper](https://arxiv.org/abs/2206.07697)