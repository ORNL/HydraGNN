#!/bin/bash
#SBATCH -A m4828
#SBATCH -J HydraGNN
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 128

function cmd() {
    echo "$@"
    time $@
}

umask 002

source module-to-load-perlmutter.sh
export PYTHONPATH=$PWD:$PYTHONPATH
# export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

# module reset
# ml pytorch/2.8.0
# VENV_PATH=/global/cfs/cdirs/m4716/jyc/HydraGNN-sc26/HydraGNN-Installation-Perlmutter/hydragnn_venv

# export PYTHONUSERBASE=$VENV_PATH
# export PATH=$VENV_PATH/bin:$PATH
# export PYTHONPATH=$VENV_PATH/lib/python3.12/site-packages:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH


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

# echo "===== Performance envs ====="
# export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
# export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
# export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
# export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
# export FI_CXI_RDV_PROTO=alt_read
# export FI_CXI_DISABLE_HOST_REGISTER=1

# export TORCH_NCCL_HIGH_PRIORITY=1     # Use high priority stream for the NCCL/RCCL Communicator.
# export GPU_MAX_HW_QUEUES=2

# export HSA_FORCE_FINE_GRAIN_PCIE=1

# # below are optional to debug RCCL stuff
# # export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=INIT

# # The following have been found to help avoid hangs, but are not yet
# # documented elsewhere
# export FI_CXI_RDZV_EAGER_SIZE=0
# export FI_CXI_RDZV_GET_MIN=0
# export FI_CXI_RDZV_THRESHOLD=0

echo "===== HydraGNN envs ====="
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export PYTHONNOUSERSITE=1

export MPICH_OFI_VERBOSE=1
#export MPICH_OFI_USE_PROVIDER="tcp;ofi_rxm"

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

[ -z $NUM_NODES ] && NUM_NODES=$SLURM_JOB_NUM_NODES
[ -z $DEPTH ] && DEPTH=1
[ -z $BATCH_SCALE ] && BATCH_SCALE=1
[ -z $EFFECTIVE_BATCH_SIZE ] && EFFECTIVE_BATCH_SIZE=$((400*32*8/BATCH_SCALE))
export EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE
export BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/SLURM_JOB_NUM_NODES/4))
export NUM_EPOCH=4

export HYDRAGNN_DDSTORE_METHOD=0
export HYDRAGNN_CUSTOM_DATALOADER=0
export HYDRAGNN_NUM_WORKERS=0
# export HYDRAGNN_DDSTORE_METHOD=1
# export HYDRAGNN_CUSTOM_DATALOADER=1
# export HYDRAGNN_NUM_WORKERS=2

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

[ -z $BEST ] && BEST=9
[ -z $PRECISION ] && PRECISION=fp64

# MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15
MULTI_MODEL_LIST=$datadir0

## Weak scaling: fixed batch size
for SCALE in `seq 0 $DEPTH`; do
    NNODES=$((SLURM_JOB_NUM_NODES / 2**SCALE))
    export BATCH_SIZE=200

    cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-node=4 --gres=gpu:4 \
    python -u ./examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
        --log=multidataset_hpo-$SLURM_JOB_ID-NN$NNODES-Weak-BEST${BEST}-${PRECISION}-B${BATCH_SIZE} --everyone \
        --inputfile=multidataset_hpo-BEST${BEST}-fp64/config.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
        --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
        --multi --ddstore --ddstore_width=128 --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
        --precision=$PRECISION --startfrom="none"
    sleep 60
done
wait

## Strong scaling: variable batch size
for SCALE in `seq 0 $DEPTH`; do
    NNODES=$((SLURM_JOB_NUM_NODES / 2**SCALE))
    # [ $NNODES -lt 32 ] && break
    export BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE / NNODES / 4))

    cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-node=4 --gres=gpu:4 \
    python -u ./examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
        --log=multidataset_hpo-$SLURM_JOB_ID-NN$NNODES-Strong-BEST${BEST}-${PRECISION}-B${EFFECTIVE_BATCH_SIZE} --everyone \
        --inputfile=multidataset_hpo-BEST${BEST}-fp64/config.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
        --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
        --multi --ddstore --ddstore_width=128 --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
        --precision=$PRECISION --startfrom="none"
    sleep 60
done
wait
