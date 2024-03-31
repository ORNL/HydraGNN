#!/bin/bash

#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 0:30:00
#SBATCH -p batch
#SBATCH -N 4

# Frontier User Guide: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html

set -x

export MIOPEN_DISABLE_CACHE=1
#export HSA_DISABLE_CACHE=1

#export ROCM_HOME=/opt/rocm-5.4.2
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=INFO
# export settings
#export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# setup env file
#echo "PATH=$PATH" > .deepspeed_env
#echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
#echo "CPATH=$CPATH" >> .deepspeed_env
#echo "TORCH_EXTENSIONS_DIR=$PWD/deepspeed" >> .deepspeed_env
#echo "HF_HOME=$PWD/hfdata" >> .deepspeed_env
#echo "ROCM_HOME=/opt/rocm-5.4.0" >> .deepspeed_env

# Configuration 
export NNODES=$SLURM_JOB_NUM_NODES # e.g., 100 total nodes
export NNODES_PER_TRIAL=2
export NUM_CONCURRENT_TRIALS=2
export NTOTGPUS=$(( $NNODES * 8 )) # e.g., 800 total GPUs
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL )) # e.g., 32 GPUs per training
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL )) # e.g., 25 total DH ranks
export OMP_NUM_THREADS=4 # e.g., 8 threads per rank
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!!" 

#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID 
mkdir -p $DEEPHYPER_LOG_DIR
export DEEPHYPER_DB_HOST=$HOST
# Start Redis server (shared memory between search processes)
# TODO: install Redis and set the `redis.conf` path here
#export REDIS_CONF=...
#pushd $DEEPHYPER_LOG_DIR
#redis-server $REDIS_CONF &
#popd

# Safe sleep to let everything start
sleep 5

echo "Doing something"

# Launch DeepHyper (1 rank per node, NTOT_DEEPHYPER_RANKS <= NNODES here)
# meaning NGPUS_PER_TRAINING >= 8
#$NTOT_DEEPHYPER_RANKS 
#srun -n1 python qm9_deephyper_multi.py
python qm9_deephyper_multi.py
