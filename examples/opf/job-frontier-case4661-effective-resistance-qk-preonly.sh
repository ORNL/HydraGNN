#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J OPF4661-ERQK-PRE
#SBATCH -o job-opf4661-erqk-pre-%j.out
#SBATCH -e job-opf4661-erqk-pre-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 4

set -euo pipefail

HYDRAGNN_ROOT=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN
HYDRAGNN_ENV=${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
OPF_DIR=${HYDRAGNN_ROOT}/examples/opf
CONFIG=configs/opf_heterosage_case4661_effective_resistance_qk.json
DATASET_NAME=case4661_effective_resistance_qk

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "${HYDRAGNN_ENV}"

export PYTHONPATH=${HYDRAGNN_ROOT}:${HYDRAGNN_ENV}/lib/python3.11/site-packages:${PYTHONPATH:-}
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

module unload darshan-runtime

cd "${OPF_DIR}"

srun --export=ALL \
  -N"${SLURM_JOB_NUM_NODES}" -n$((SLURM_JOB_NUM_NODES * 8)) -c7 \
  --gpus-per-task=1 --gpu-bind=closest \
  python -u train_opf_solution_heterogeneous.py \
  --inputfile "${CONFIG}" \
  --case_name pglib_opf_case4661_sdet \
  --num_groups all \
  --preonly --hdf5 \
  --modelname "${DATASET_NAME}" \
  --data_root "${HYDRAGNN_ROOT}/examples/opf/dataset"
