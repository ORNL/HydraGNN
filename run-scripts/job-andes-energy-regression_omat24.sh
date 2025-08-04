#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 10

NN=$SLURM_JOB_NUM_NODES

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-andes.sh

source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_andes/bin/activate
conda activate hydragnn_andes

export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_andes/adios2-install/lib/python3.11/site-packages:$PYTHONPATH

export PYTHONPATH=$PWD:$PYTHONPATH

cd examples/open_materials_2024

ln -snf ../multidataset/energy_linear_regression.py .
ln -snf ../multidataset/energy_per_atom_linear_regression.py .

srun -n$((SLURM_JOB_NUM_NODES*8)) -c 4 -l python -u ./energy_per_atom_linear_regression.py --notestset OMat24
sleep 5

srun -n$((SLURM_JOB_NUM_NODES*8)) -c 4 -l python -u ./energy_linear_regression.py --notestset OMat24

