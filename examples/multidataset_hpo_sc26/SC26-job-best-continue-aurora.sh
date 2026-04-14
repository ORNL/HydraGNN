#!/bin/bash
#PBS -N HydraGNN
#PBS -l select=2048
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=flare
#PBS -q prod
#PBS -A HydraGNN
#PBS -j oe

function cmd() {
    echo "$@"
    time $@
}

export NNODES=`wc -l < $PBS_NODEFILE`
export NPROCS_PER_NODE=12 # Number of MPI ranks to spawn per node
export NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
export NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)
export NTOTPROCS=$(( NNODES * NPROCS_PER_NODE ))

module reset
module load frameworks
source /lus/flare/projects/HydraGNN/jychoi/HydraGNN/HydraGNN-Installation-Aurora/hydragnn_venv/bin/activate
export PYTHONNOUSERSITE=1

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTPROCS} RANKS_PER_NODE= ${NPROCS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

cd ${PBS_O_WORKDIR}

echo $PWD
export PYTHONPATH=$PWD:$PYTHONPATH

## Aurora envs for preventing torch ddp hangs
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=300
export FI_MR_CACHE_MONITOR=userfaultfd

export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

export HYDRAGNN_TRACE_LEVEL=1
export BATCH_SIZE=$((200*8/12)) ## OOM
export BATCH_SIZE=$((200*8/12/2))
export NUM_EPOCH=1000

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2
export OMP_NUM_THREADS=8

# ## FIXME: Turn on sync. This helped with hangs when tensor detach().cpu(). Might impact on performance. Need to investigate more.
# export CUDA_LAUNCH_BLOCKING=1
# export AMD_SERIALIZE_KERNEL=3

# Dataset ordering matches gfm_deephyper_multi_all_mpnn.py multi_model_list
export datadir0=Alexandria
export datadir1=ANI1x
export datadir2=MPTrj
export datadir3=OC2020
export datadir4=OC2022
export datadir5=OC25
export datadir6=ODAC23
export datadir7=OMat24
export datadir8=OMol25
export datadir9=OMol25-neutral
export datadir10=OMol25-non-neutral
export datadir11=OPoly2026
export datadir12=Nabla2DFT
export datadir13=QCML
export datadir14=QM7X
export datadir15=transition1x

export MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15

echo "ulimit:"
ulimit -a

## HYDRAGNN_USE_FSDP: 1 (enabled), 0 (disabled)
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD
# export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
# export HYDRAGNN_FSDP_STRATEGY=NO_SHARD

export CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
# export GPU_BIND_SCHEME="--gpu-bind=list:0.0:0.1:1.0:1.1:2.0:2.1:3.0:3.1:4.0:4.1:5.0:5.1"

[ -z $BEST ] && BEST=9
[ -z $PRECISION ] && PRECISION=fp32

export HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT=1
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1
export FI_CXI_RX_MATCH_MODE=hybrid

ulimit -c unlimited

## Checking
env | grep ^MPICH
env | grep ^HYDRA
env | grep ^FI


cmd mpiexec -n ${NTOTPROCS} --ppn ${NPROCS_PER_NODE} ${CPU_BIND_SCHEME} \
    python -u examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-BEST${BEST}-${PRECISION} --logfilename=run-${PBS_JOBID}.log --everyone \
    --inputfile=multidataset_hpo-BEST${BEST}-fp64/config.json \
    --multi --ddstore --ddstore_width=$((256*12)) --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=$PRECISION --startfrom=multidataset_hpo-BEST${BEST}-${PRECISION}
