#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 1 #16 
##SBATCH -S 1

#module reset
#module load PrgEnv-gnu
#module load rocm/5.7.1
#module load cmake
#module load craype-accel-amd-gfx90a
#module load amd-mixed/5.7.1
#module load cray-mpich/8.1.26
#module load miniforge3/23.11.0
#module unload darshan-runtime
#source activate /lustre/orion/world-shared/cph161/jyc/frontier/sw/envs/hydragnn-py39-rocm571-amd

module reset
module load PrgEnv-gnu
module load cpe/23.09
module load cray-mpich/8.1.26
module load libfabric/1.15.2.0
module load craype-accel-amd-gfx90a
module load rocm/5.7.1
module load cmake
module load miniforge3/23.11.0
module unload darshan-runtime
source /sw/frontier/miniforge3/23.11.0-0/bin/activate
conda activate /lustre/orion/world-shared/cph161/jyc/frontier/sw/envs/hydragnn-py3.12-rocm5.7.1-mpich8.1.26


## Use ADM build
#PYTORCH_DIR=/autofs/nccs-svm1_sw/crusher/amdsw/karldev/pytorch-2.2.2-rocm5.7.1
#PYG_DIR=/autofs/nccs-svm1_sw/crusher/amdsw/karldev/pyg-rocm5.7.1
#export PYTHONPATH=$PYG_DIR:$PYTORCH_DIR:$PYTHONPATH

#module use -a /lustre/orion/world-shared/cph161/jyc/frontier/sw/modulefiles
#module load adios2/2.9.2-mpich-8.1.26

#module load aws-ofi-rccl/devel-rocm5.7.1
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

#export PYTHONPATH=/dir/to/HydraGNN:$PYTHONPATH
export PYTHONPATH=/lustre/orion/cph161/proj-shared/zhangp/HydraGNN:$PYTHONPATH

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py  --multi --ddstore --multi_model_list=ANI1x-v3,MPTrj-v3,OC2020-20M-v3,OC2022-v3,qm7x-v3

export datadir1=/lustre/orion/cph161/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/qm7x/dataset/qm7x.bp
export datadir2=/lustre/orion/cph161/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/ani1_x/dataset/ANI1x.bp

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py \
--inputfile=gfm_multibranch_physics-informed.json --num_samples=10000 --multi --ddstore --multi_model_list=$datadir1,$datadir2 

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py --inputfile=gfm_multibranch.json --num_samples=100 --multi --ddstore --multi_model_list=ANI1x-v3,MPTrj-v3

