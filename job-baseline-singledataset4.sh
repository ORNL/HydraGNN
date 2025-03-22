#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-basesingle4-%j.out
#SBATCH -e job-basesingle4-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch 
##SBATCH -q debug
#SBATCH -N 1 #16 
##SBATCH -S 1

 
# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm624
 
#export python path to use ADIOS2 v.2.9.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

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


export datadir0=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/ANI1x-v3.bp
export datadir1=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/qm7x-v3.bp
export datadir2=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/MPTrj-v3.bp
export datadir3=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/Alexandria-v3.bp
export datadir4=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/transition1x-v3.bp


srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py --log=GFM_baseline_data4 \
--inputfile=baseline_GFM260.json --num_samples=100000 --multi --ddstore --multi_model_list=$datadir4
