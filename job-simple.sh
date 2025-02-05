#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J EnergyProfiling
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
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
export HYDRAGNN_TRACE_LEVEL=1
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export FI_MR_CACHE_MONITOR=disabled


# ROCM 517Modules
# ml PrgEnv-gnu
# ml rocm/5.7.1
# ml cmake/3.23.2
# ml craype-accel-amd-gfx90a
# ml amd-mixed/5.7.1
# ml cray-mpich/8.1.28
# source /sw/frontier/miniforge3/23.11.0-0/bin/activate
# conda activate /lustre/orion/world-shared/cph161/jyc/frontier/sw/envs/hydragnn-py3.12-rocm5.7.1-mpich8.1.26

# ROCM 613 Modules
# module reset
# ml PrgEnv-gnu
# ml rocm/6.1.3
# ml cmake/3.23.2
# ml craype-accel-amd-gfx90a
# ml amd-mixed/6.1.3
# ml cray-mpich/8.1.30

# source /lustre/orion/cph161/world-shared/adithya/anaconda3/bin/activate
# conda activate hydragnn_rocm613_adi
# #export python path to use ADIOS2 v.2.9.2
# export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier_rocm613/install/lib/python3.12/site-packages/

# ROCM 620 Modules
source /lustre/orion/cph161/world-shared/adithya/frontier/module-to-load-frontier-rocm620.sh
source /lustre/orion/cph161/world-shared/adithya/anaconda3/bin/activate
conda deactivate
conda activate hydragnn_rocm620_amdsmi

# export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier_rocm620/install/lib/python3.11/site-packages/

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA
env | grep ^NCCL

#export PYTHONPATH=/dir/to/HydraGNN:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

which python
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.__file__)"

model_size="SMALL"



# srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest --ntasks-per-node=8 --gpus-per-node=8 \
#      python -u examples/multidataset/train.py --modelname=GFM --multi_model_list='MPTrj-v2'\
#      --inputfile=SMALL_MTL.json --num_epoch=1 --multi --ddstore --everyone \
#      --log=GFM_${SLURM_JOB_ID}_NN${SLURM_JOB_NUM_NODES} --num_samples=3500


## ONLY OC2022 with 3500 SAMPLES
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8))  --ntasks-per-node=8 --gpus-per-node=8 \
     bash -c "export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID; python examples/multidataset/train.py --modelname=GFM --multi_model_list='MPTrj-v2'
     --inputfile=${model_size}_MTL.json --num_epoch=3 --multi --ddstore --everyone
     --log=GFM_${SLURM_JOB_ID}_NN${SLURM_JOB_NUM_NODES} --num_samples=3500"

# srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) --ntasks-per-node=8 --gpus-per-node=8 \
#        bash -c "export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID; python examples/multidataset/train.py --modelname=GFM --multi_model_list='OC2020-20M' --inputfile=${model_size}_MTL.json --num_epoch=2 --multi --ddstore --everyone  --log=GFM_${SLURM_JOB_ID}_NN${SLURM_JOB_NUM_NODES}_${model_size}"


# srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest --ntasks-per-node=8 --gpus-per-node=8 \
#       python -u examples/multidataset/train.py --modelname=GFM --multi_model_list='ANI1x-v2,MPTrj-v2,OC2020-20M,OC2022,qm7x-v2' --inputfile=${model_size}_MTL.json --num_epoch=2 --multi --ddstore --everyone --log=GFM_${SLURM_JOB_ID}_NN${SLURM_JOB_NUM_NODES} --slurmjobid $SLURM_JOB_ID


# srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest --ntasks-per-node=8 --gpus-per-node=8 python amd_test.py

