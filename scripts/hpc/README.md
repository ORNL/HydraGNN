# HPC facility assets

Facility-specific environment, monitoring, and launch assets are organized as:

```text
scripts/hpc/<facility>/<system>/
```

For example, `olcf/frontier/environments` contains interactive Frontier setup
scripts, while `olcf/frontier/omnistat` contains the corresponding Omnistat
collector configurations and `olcf/frontier/installation` contains installation
scripts. Facility-wide helpers, such as `olcf/proxy-env.sh`, live at the facility
level. Keep portable HydraGNN utilities outside this hierarchy.

## Installation scripts

Run an installation script from the HydraGNN repository root. Each script
creates or reuses its own installation directory and documents supported
environment-variable overrides near the top of the file.

| System | Script | Accelerator dependency selection |
| --- | --- | --- |
| Aurora | `alcf/aurora/installation/install.sh` | PyTorch/XPU stack supplied by the ALCF `frameworks` module |
| Perlmutter | `nersc/perlmutter/installation/install.sh` | CUDA 13.0 (`cu130`), PyTorch 2.13.0, torchvision 0.28.0 |
| Andes | `olcf/andes/installation/install.sh` | CPU PyTorch wheel |
| Andes (parallel) | `olcf/andes/installation/install-parallel.sh` | CPU PyTorch wheel with parallel source builds |
| Frontier ROCm 6.4 | `olcf/frontier/installation/install-rocm64.sh` | PyTorch wheels from the ROCm 6.4 index |
| Frontier ROCm 6.4 (parallel) | `olcf/frontier/installation/install-parallel-rocm64.sh` | PyTorch wheels from the ROCm 6.4 index with parallel source builds |
| Frontier ROCm 7.1 | `olcf/frontier/installation/install-rocm71.sh` | PyTorch wheels from the ROCm 7.1 index |
| Frontier ROCm 7.2 | `olcf/frontier/installation/install-rocm72.sh` | PyTorch 2.14.0 and torchvision 0.29.0 from the ROCm 7.2 index |
| Frontier ROCm 7.13 | `olcf/frontier/installation/install-rocm713.sh` | Platform-tested ROCm 7.13 PyTorch wheel set |

For example:

```bash
scripts/hpc/nersc/perlmutter/installation/install.sh
scripts/hpc/olcf/frontier/installation/install-rocm72.sh
```

The generic installation currently selects NumPy 2.4.6, PyTorch 2.13.0,
torchvision 0.28.0, and PyTorch Geometric 2.8.0. HydraGNN accepts PyTorch 2.13
or 2.14 so facility installers can retain tested accelerator wheels. Facility
installers keep shared Python dependencies aligned but remain
authoritative for accelerator-specific PyTorch wheels and framework modules.
Do not install `requirements-torch.txt` over an environment created by a
facility installer. HydraGNN does not require torchaudio.
