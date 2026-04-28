#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1024
#SBATCH -C nvme

function cmd() {
    echo "$@"
    time $@
}

## Default location
[ -z $HYDRAGNN_ROOT ] && HYDRAGNN_ROOT=$PWD
export HYDRAGNN_ROOT=$HYDRAGNN_ROOT

# Load conda environment
module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/7.1.1
ml amd-mixed/7.1.1
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
module unload darshan-runtime

source activate $HYDRAGNN_ROOT/HydraGNN-Installation-Frontier/hydragnn_venv

# Add HydraGNN in PYTHONPATH
export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH

echo ""
echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo ""
echo "===== Module List ====="
module list

echo ""
echo "===== Check LD_LIBRARY_PATH ====="
echo $LD_LIBRARY_PATH  | tr ':' '\n'

echo "===== HydraGNN envs ====="
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_FSDP_VERSION=2
export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=20

export BATCH_SIZE=200
export NUM_EPOCH=4

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2
# export HYDRAGNN_DDSTORE_METHOD=0
# export HYDRAGNN_CUSTOM_DATALOADER=0
# export HYDRAGNN_NUM_WORKERS=0


## Check dataset
if [ ! -d $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/dataset/OC2020-v2.bp ]; then
    pushd $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26 > /dev/null
    [ ! -d dataset ] && mkdir -p dataset
    ln -snf /lustre/orion/lrn070/world-shared/kmehta/hydragnn/datasets/v2/OC2020-v2.bp dataset/OC2020-v2.bp
    popd > /dev/null
fi

MULTI_MODEL_LIST=$datadir3

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-${SLURM_JOB_ID}-NN${SLURM_JOB_NUM_NODES} --everyone \
    --inputfile=gfm_mlip.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --multi --ddstore --multi_model_list=OC2020 --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=fp64 --startfrom="none"
