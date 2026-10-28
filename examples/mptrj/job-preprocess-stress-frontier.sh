#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J MPTrj-stress-ADIOS
#SBATCH -o job-preprocess-stress-%j.out
#SBATCH -e job-preprocess-stress-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 32
#SBATCH -p batch

set -euo pipefail

HYDRAGNN_ROOT=/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN
EXAMPLE_DIR=${HYDRAGNN_ROOT}/examples/mptrj
OUTPUT_DIR=${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26/datasest
MODEL_NAME=MPTrj-v3
LINEAR_MODEL_NAME=MPTrj-v3-linear

module reset
ml cpe/24.07 cce/18.0.0 rocm/7.2.0 amd-mixed/7.2.0 craype-accel-amd-gfx90a PrgEnv-gnu miniforge3/23.11.0-0 git-lfs
module unload darshan-runtime
source activate ${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72

export PYTHONNOUSERSITE=1
export PYTHONPATH=${HYDRAGNN_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}
export HYDRAGNN_BACKEND=gloo
export HYDRAGNN_AGGR_BACKEND=mpi
export OMP_NUM_THREADS=14

cd ${EXAMPLE_DIR}
rm -rf dataset/${MODEL_NAME}.bp

srun -N${SLURM_JOB_NUM_NODES} -n${SLURM_JOB_NUM_NODES} -c14 \
    --cpu-bind=cores --kill-on-bad-exit=1 \
    python -u train.py \
        --preonly \
        --adios \
        --inputfile=mptrj_energy.json \
        --modelname=${MODEL_NAME} \
        --precision=fp32

test -d dataset/${MODEL_NAME}.bp
rm -rf ${OUTPUT_DIR}/${MODEL_NAME}.bp
mv dataset/${MODEL_NAME}.bp ${OUTPUT_DIR}/${MODEL_NAME}.bp

rm -rf ${OUTPUT_DIR}/${LINEAR_MODEL_NAME}.bp
srun -N${SLURM_JOB_NUM_NODES} -n${SLURM_JOB_NUM_NODES} -c14 \
    --cpu-bind=cores --kill-on-bad-exit=1 \
    python -u ${HYDRAGNN_ROOT}/examples/multidataset/energy_linear_regression.py \
        ${MODEL_NAME} \
        --input=${OUTPUT_DIR}/${MODEL_NAME}.bp \
        --output=${OUTPUT_DIR}/${LINEAR_MODEL_NAME}.bp
