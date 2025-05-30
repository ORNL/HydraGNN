#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-adios-linreg-%j.out
#SBATCH -e job-adios-linreg-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch 
##SBATCH -q debug
#SBATCH -N 16 #1
##SBATCH -S 1



# Load conda environment
module reset
# source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-andes.sh
# source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm631.sh
# source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_andes/bin/activate
conda deactivate
# conda activate hydragnn_rocm624
conda activate hydragnn_andes
# conda activate hydragnn_rocm631
 
# #export python path to use ADIOS2 v.2.9.2
# export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH
#export python path to use ADIOS for Andes
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_andes/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH
# #export python path to use ADIOS2 for ROCM 6.31 (was v.2.10.2 but v.2.9.2 for now because of problem)
# export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm631/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

export PYTHONPATH=/lustre/orion/lrn070/world-shared/rylieweaver/Direct-vs-Autodiff-ForceComp/HydraGNN:$PYTHONPATH

which python
python -c "import numpy; print(numpy.__version__)"


echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA


#PreOnly
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./energy_linear_regression.py