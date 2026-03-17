#!/bin/bash
#SBATCH -A LRN078
#SBATCH -J HydraGNN
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/job-hydragnn-grid-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/job-hydragnn-grid-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -N 5 #16 
##SBATCH -S 1
 
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

function cmd() {
    echo "$@"
    time $@
} 

HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN

# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate /lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
 
#export python path to HydragNN
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

export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1

export NCCL_NET_PLUGIN=${PLUGIN_PATH}/lib/librccl-net.so
export NCCL_NET_GDR_LEVEL="PHB"       # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export NCCL_NET="AWS Libfabric"

export TORCH_NCCL_HIGH_PRIORITY=1     # Use high priority stream for the NCCL/RCCL Communicator.
export GPU_MAX_HW_QUEUES=2

export HSA_FORCE_FINE_GRAIN_PCIE=1

# below are optional to debug RCCL stuff
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

# The following have been found to help avoid hangs, but are not yet
# documented elsewhere
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

cd $HYDRAGNN_ROOT/examples/opf
 
which python
python -c "import numpy; print(numpy.__version__)"

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u train_mlip.py --preonly --adios --ddstore
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u train_opf_heterogeneous.py --preonly --adios 
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u train_opf_solution_heterogeneous.py --preonly --adios --case_name pglib_opf_case118_ieee pglib_opf_case14_ieee pglib_opf_case2000_goc pglib_opf_case30_ieee pglib_opf_case500_goc pglib_opf_case57_ieee
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u infer_opf_solution_heterogeneous.py --adios 
