#!/bin/bash
#PBS -N HydraGNN
#PBS -l select=32
#PBS -l place=scatter
#PBS -l walltime=2:00:00
#PBS -l filesystems=flare
#PBS -q prod
#PBS -A HydraGNN
#PBS -j oe
#PBS -m abe

function cmd() {
    echo "$@"
    time $@
}

export NNODES=`wc -l < $PBS_NODEFILE`
export NPROCS_PER_NODE=12 # Number of MPI ranks to spawn per node
export NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
export NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)
export NTOTPROCS=$(( NNODES * NPROCS_PER_NODE ))

## Default location
[ -z $HYDRAGNN_ROOT ] && HYDRAGNN_ROOT=$PWD
export HYDRAGNN_ROOT=$HYDRAGNN_ROOT

# module reset
module load frameworks
source $HYDRAGNN_ROOT/HydraGNN-Installation-Aurora/hydragnn_venv/bin/activate

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

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTPROCS} RANKS_PER_NODE= ${NPROCS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

cd ${PBS_O_WORKDIR}

echo $PWD
export PYTHONPATH=$PWD:$PYTHONPATH

## Aurora envs for preventing torch ddp hangs
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=900
export FI_MR_CACHE_MONITOR=userfaultfd

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

##export BATCH_SIZE=200 ## segfault with 32 nodes
export BATCH_SIZE=100
export NUM_EPOCH=4

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2

## Check dataset
if [ ! -d $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/dataset/OC2020-v2.bp ]; then
    pushd $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26 > /dev/null
    [ ! -d dataset ] && mkdir -p dataset
    ln -snf /lus/flare/projects/HydraGNN/SC26-dataset/*.bp dataset/
    popd > /dev/null
fi

## Full list of datasets
# MULTI_MODEL_LIST=Alexandria,ANI1x,MPTrj,OC2020,OC2022,OC25,ODAC23,OMat24,OMol25-neutral,OMol25-non-neutral,OMol25,OPoly2026,QCML,QM7X,transition1x
MULTI_MODEL_LIST=OC2020

export CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"

cmd mpiexec -n ${NTOTPROCS} --ppn ${NPROCS_PER_NODE} ${CPU_BIND_SCHEME} --label \
python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-${PBS_JOBID}-NN${NNODES} --everyone \
    --inputfile=gfm_mlip.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --multi --ddstore --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=fp64 --startfrom="none"
