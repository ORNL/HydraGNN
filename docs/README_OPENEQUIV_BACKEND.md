### Equivariance Backend Selection (e3nn vs OpenEquivariance)
HydraGNN supports selecting an alternative backend for Clebsch-Gordon tensor products.

Backends:
- `e3nn` (default)
- `openequivariance` (requires installing the [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) package and a CUDA-capable device)

Config example:
```yaml
NeuralNetwork:
  Architecture:
    equivariance_backend: openequivariance
```

Environment variable override:
```bash
export HYDRAGNN_EQUIVARIANCE_BACKEND=openequivariance
```
Priority: environment variable > config > default.

If `openequivariance` is requested but not installed, HydraGNN logs a warning and falls back to the e3nn implementation automatically.

No architecture code changes are required; the existing interaction block becomes backend-agnostic.