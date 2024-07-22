#!/bin/bash
#SBATCH -A LRN031
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -S 1

ulimit -n 65536

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0 
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU


source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-frontier.sh

source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn

export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier/install/lib/python3.8/site-packages/:$PYTHONPATH

export PYTHONPATH=$PWD:$PYTHONPATH


# both commands should work
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gres=gpu:8 \
   python -u ./examples/ogb/train_gap.py gap --adios --use_deepspeed
# srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
#     python -u ./examples/ogb/train_gap.py gap --adios --use_deepspeed
