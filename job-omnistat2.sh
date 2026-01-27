#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -N 10
#SBATCH --network=disable_rdzv_get

function cmd() {
    echo "$@"
    time $@
} 

HYDRAGNN_ROOT=/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2026/HydraGNN

# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm640.sh
source activate /lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier/hydragnn_venv
 
#export python path to HydragNN
export PYTHONPATH=$PWD:$PYTHONPATH

#export python path to use ADIOS2 v.2.10.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH
 
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

module unload darshan-runtime
module list


echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

## Getting error without these after 20 nodes
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export FI_MR_CACHE_MONITOR=disabled

## aws-ofi-rccl plugin settings
export TORCH_NCCL_HIGH_PRIORITY=1
export FI_CXI_RDV_PROTO=alt_read

export PATH_TO_THE_PLUGIN_DIRECTORY=/lustre/orion/lrn070/world-shared/mlupopa/AWI_OFI_RCCL_ROCm631/aws-ofi-rccl/lib
export LD_LIBRARY_PATH=${PATH_TO_THE_PLUGIN_DIRECTORY}:$LD_LIBRARY_PATH
 
export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
 
export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=100
[ -z $BATCH_SIZE ] && BATCH_SIZE=20 ## 320 (Perlmutter) 160 (Frontier)
export BATCH_SIZE=$BATCH_SIZE
export NUM_EPOCH=4

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2

# Dataset ordering matches gfm_deephyper_multi_all_mpnn.py multi_model_list
export datadir0=Alexandria
export datadir1=ANI1x
export datadir2=MPTrj
export datadir3=OC2020
export datadir4=OC2022
export datadir5=ODAC23
export datadir6=OMat24
export datadir7=OMol25
export datadir8=OC2025
export datadir9=OPoly2026
export datadir10=Nabla2DFT
export datadir11=QCML
export datadir12=QM7-X
export datadir13=transition1x

# (A) Setup omnistat sampling environment
ml use /sw/frontier/amdsw/modulefiles/
ml omnistat-wrapper
export OMNISTAT_CONFIG=$HYDRAGNN_ROOT/omnistat.hydragnn-external-fp64.config

# (B) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 1

## HYDRAGNN_USE_FSDP: 1 (enabled), 0 (disabled)
export HYDRAGNN_USE_FSDP=0
# export HYDRAGNN_USE_FSDP=1
export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD
# export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
# export HYDRAGNN_FSDP_STRATEGY=NO_SHARD

## Getting strange omnistat counts with the following line. Otherswise, FSDP runs fine.

# [ -z $HIDDEN_DIM ] && HIDDEN_DIM=889
# [ -z $HIDDEN_DIM ] && HIDDEN_DIM=1000
# [ -z $HIDDEN_DIM ] && HIDDEN_DIM=2000
[ -z $HIDDEN_DIM ] && HIDDEN_DIM=3000


${OMNISTAT_DIR}/omnistat-annotate --mode start --text  "DDP + fp64"
export HYDRAGNN_USE_FSDP=0
# cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gres=gpu:8 \
cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-FSDP$HYDRAGNN_USE_FSDP --everyone \
    --inputfile=gfm_mlip.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH)) \
    --multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2 --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --task_parallel --use_devicemesh --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --precision=fp64 \
    --mpnn_type=EGNN \
    --hidden_dim=$HIDDEN_DIM \
    --num_conv_layers=6 \
    --num_headlayers=3 \
    --dim_headlayers=2000
${OMNISTAT_DIR}/omnistat-annotate --mode stop
sleep 10

# (C) End of job: stop data collection
${OMNISTAT_WRAPPER} usermode --stop
