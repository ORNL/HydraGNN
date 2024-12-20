#!/bin/bash
#SBATCH -A m4716
#SBATCH -J HydraGNN
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32

# Retrieve the number of nodes set via `sbatch -N` or in the script
echo "Number of nodes allocated: $SLURM_NNODES"

WIDTH=${1:-50}     # Default to 50 hidden_dim if not specified
DEPTH=${2:-3}      # Default to 3 num_conv_layers if not specified
DS=${3:-1.2}       # Default to 1.2TB data if not specified
ZERO=${4:-False}   # Default to False if not specified
CKPT=${5:-False}   # Default to False if not specified

## Remove write permission for others in terms of newly created files and dirs
umask 002

## Load Basic Envs
module reset
module load pytorch/2.0.1

module use -a /global/cfs/cdirs/m4133/jyc/perlmutter/sw/modulefiles
module load hydragnn/pytorch2.0.1-v2
module use -a /global/cfs/cdirs/m4133/c8l/sw/modulefiles
module load deepspeed

## MPI Envs
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=0

## HYDRAGNN Envs
HYDRAGNN_DIR=/global/cfs/cdirs/m4716/c8l/HydraGNN
export PYTHONPATH=$HYDRAGNN_DIR:$PYTHONPATH

export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_TRACE_LEVEL=0

## Dataset Envs
DATASET_PATH="/global/cfs/projectdirs/m4716/mlupopa/HydraGNN/examples/multidataset_hpo/dataset"
DATASET_LIST="MPTrj-v3,ANI1x-v3,OC2020-20M-v3,OC2022-v3,qm7x-v3"

## Log Envs
LOG_NAME="exp-${DEPTH}_depth-${WIDTH}_width-${DS}_TB_data-${SLURM_NNODES}_nodes"

## Calculate batch size and num_samples
BS=$((32 * 32 / SLURM_NNODES)) # Dynamic calculation of batch size, default setting: 32 nodes with 32 batch size per GPU
NS=$(echo "scale=0; 285715 / 1.2 * ${DS} * 32 / $SLURM_NNODES" | bc) # Calculate number of samples, default setting: 32 nodes with 285715 num_samples per GPU

## Handle optional arguments
EXTRA_ARGS=""
if [ "$ZERO" = "True" ]; then
    EXTRA_ARGS+=" --zero_opt"
fi
if [ "$CKPT" = "True" ]; then
    EXTRA_ARGS+=" --conv_checkpointing"
fi

## run scripts
set -x

srun -N${SLURM_NNODES} -n$((SLURM_NNODES*4)) -c32 --ntasks-per-node=4 --gpus-per-task=1 \
    python -u $HYDRAGNN_DIR/examples/multidataset_deepspeed/train.py \
        --inputfile=base.json \
        --dataset_path=$DATASET_PATH \
        --multi \
        --multi_model_list=$DATASET_LIST \
        --num_epoch=10 \
        --everyone --ddstore \
        --log=$LOG_NAME \
        --hidden_dim=${WIDTH} \
        --num_conv_layers=${DEPTH} \
        --full_test \
        --batch_size=${BS} \
        --num_samples=${NS} \
        ${EXTRA_ARGS}

set +x
