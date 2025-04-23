#!/bin/bash
#SBATCH -A m4505
#SBATCH -J HydraGNN
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 5

umask 002

source /global/cfs/cdirs/m4716/jyc/sw/anaconda3/2024.10/bin/activate
conda activate /global/cfs/cdirs/m4716/jyc/sw/envs/hydragnn

## HydraGNN
export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6
export PYTHONPATH=$PWD:$PYTHONPATH

## Envs
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=0

export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0
export HYDRAGNN_TRACE_LEVEL=1

export HYDRAGNN_MAX_NUM_BATCH=5
[ -z $BATCH_SIZE ] && BATCH_SIZE=320 ## 320 (Perlmutter) 160 (Frontier)
export BATCH_SIZE=$BATCH_SIZE

export datadir0=/global/cfs/cdirs/m4716/HydraGNN-sc25-comm/ANI1x-v3.bp
export datadir1=/global/cfs/cdirs/m4716/HydraGNN-sc25-comm/qm7x-v3.bp
export datadir2=/global/cfs/cdirs/m4716/HydraGNN-sc25-comm/MPTrj-v3.bp
export datadir3=/global/cfs/cdirs/m4716/HydraGNN-sc25-comm/Alexandria-v3.bp
export datadir4=/global/cfs/cdirs/m4716/HydraGNN-sc25-comm/transition1x-v3.bp


srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --gpu-bind=none \
python -u ./examples/multibranch/train.py --log=GFM_taskparallel_weak-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-BS$BATCH_SIZE-TP0-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
--inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
--multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --batch_size=$BATCH_SIZE --num_epoch=4 \
--oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))

sleep 5

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --gpu-bind=none \
python -u ./examples/multibranch/train.py --log=GFM_taskparallel_weak-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-BS$BATCH_SIZE-TP1-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
--inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
--multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --batch_size=$BATCH_SIZE --num_epoch=4 \
--task_parallel --use_devicemesh --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))
