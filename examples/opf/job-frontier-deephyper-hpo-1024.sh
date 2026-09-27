#!/bin/bash
#SBATCH -A LRN078
#SBATCH -J OPF-DeepHyper-1024
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/job-opf-hpo-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/job-opf-hpo-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 1024

function cmd() {
    echo "$@"
    time $@
}

HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN

# Load conda environment
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate /lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

#export python path to HydraGNN
export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH

#export python path to use ADIOS2 v.2.10.2
export PYTHONPATH=/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH

which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

module unload darshan-runtime
module list

echo $LD_LIBRARY_PATH  | tr ':' '\n'

## aws-ofi-rccl plugin settings
export PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PLUGIN_PATH}/lib

export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1

export NCCL_NET_PLUGIN=${PLUGIN_PATH}/lib/librccl-net.so
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"

export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2

export HSA_FORCE_FINE_GRAIN_PCIE=1

export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

## OPF HPO settings
export NUM_EPOCH=10
export BATCH_SIZE=32

# DeepHyper configuration
# Each trial uses NNODES_PER_TRIAL nodes; total nodes are split across concurrent trials
export NNODES=$SLURM_JOB_NUM_NODES
export NNODES_PER_TRIAL=16
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 ))
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL ))
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL ))
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR: GPU count mismatch!"

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-opf-hpo"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR

cd $HYDRAGNN_ROOT/examples/opf

which python
python -c "import numpy; print(numpy.__version__)"

# Use MPNN_TYPE env var if set, otherwise default to all heterogeneous models
[ -z $MPNN_TYPE ] && MPNN_TYPE=HeteroPNA,HeteroSAGE,HeteroGAT,HeteroRGAT,HeteroHGT,HeteroHEAT

cmd python -u $HYDRAGNN_ROOT/examples/opf/opf_deephyper_hpo.py \
    --mpnn_type=$MPNN_TYPE \
    --max_evals=200
