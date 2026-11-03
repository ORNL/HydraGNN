#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1024
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

function cmd() {
    echo "$@"
    time $@
}

## Default location
[ -z $HYDRAGNN_ROOT ] && HYDRAGNN_ROOT=$PWD
export HYDRAGNN_ROOT=$HYDRAGNN_ROOT

# Load conda environment
# --- ROCm 7.2.4 toolchain ---
module reset
module load rocm/7.2.4
module load gcc/14.4.0
module load git-lfs/3.7.1
module load mpich/5.0.1
module load miniforge3/26.3.2-3

source activate $HYDRAGNN_ROOT/HydraGNN-Installation-Lux-ROCm72/hydragnn_venv_rocm72

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

# 32-node UCX error fix
export UCX_TLS=sm,self,rocm_copy,rocm_ipc,cma,tcp
export UCX_NET_DEVICES=bond0

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
    ln -snf /lustre/orion/lrn070/world-shared/kmehta/hydragnn/datasets/v2/*.bp dataset/
    popd > /dev/null
fi

## Full list of datasets
# MULTI_MODEL_LIST=Alexandria,ANI1x,MPTrj,OC2020,OC2022,OC25,ODAC23,OMat24,OMol25-neutral,OMol25-non-neutral,OMol25,OPoly2026,QCML,QM7X,transition1x
MULTI_MODEL_LIST=OC2020

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-node=8 \
python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-${SLURM_JOB_ID}-NN${SLURM_JOB_NUM_NODES} --everyone \
    --inputfile=gfm_mlip.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --multi --ddstore --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=fp64 --startfrom="none"
