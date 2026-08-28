#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J hetero-gps-tests
#SBATCH -o job-hetero-gps-tests-%j.out
#SBATCH -e job-hetero-gps-tests-%j.out
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1

set -eo pipefail

HYDRAGNN_ROOT=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN
HYDRAGNN_ENV=${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "${HYDRAGNN_ENV}"

export PYTHONPATH="${HYDRAGNN_ROOT}:${HYDRAGNN_ENV}/lib/python3.11/site-packages:${PYTHONPATH:-}"

cd "${HYDRAGNN_ROOT}"

which python
python --version
python -c "import torch, torch_geometric, pytest; print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__); print('pytest', pytest.__version__)"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -m pytest -vv -s \
    tests/test_hetero_gps_generalization.py \
    tests/test_heterogeneous_message_passing.py
