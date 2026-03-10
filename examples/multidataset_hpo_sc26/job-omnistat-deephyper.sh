#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN-DeepHyper
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 256
#SBATCH -C nvme

function cmd() {
    echo "$@"
    time $@
}

function setup_bb()
{
    # Move a copy of the env to the NVMe on each node
    if [ -d /mnt/bb/${USER} ]; then
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
        for FILENAME in HydraGNN-Installation-Frontier.tar.gz; do
            echo "Checking ${FILENAME}"
            if [ ! -f /mnt/bb/${USER}/${FILENAME} ]; then
                echo "Copying ${FILENAME} to each local NVME"
                time sbcast -pfv ${HYDRAGNN_ROOT}/${FILENAME} /mnt/bb/${USER}/${FILENAME}
                if [ ! "$?" == "0" ]; then
                    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
                    # your application may pick up partially complete shared library files, which would give you confusing errors.
                    echo "SBCAST failed!"
                    break
                fi
            fi
            if [ ! -d /mnt/bb/${USER}/HydraGNN-Installation-Frontier ]; then
                echo "Untar ${FILENAME}"
                time srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u tar -xf /mnt/bb/${USER}/${FILENAME} -C  /mnt/bb/${USER}/
            fi
        done
        echo "NVME is ready to use"
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
    fi
}

HYDRAGNN_ROOT=/lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN

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

setup_bb
if [ -d /mnt/bb/${USER}/HydraGNN-Installation-Frontier ]; then
    source activate /mnt/bb/${USER}/HydraGNN-Installation-Frontier/hydragnn_venv
    # export PYTHONPATH=/mnt/bb/${USER}/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH
    # export PATH=/mnt/bb/${USER}/HydraGNN-Installation-Frontier/hydragnn_venv/bin/:$PATH
else
    source activate $HYDRAGNN_ROOT/HydraGNN-Installation-Frontier/hydragnn_venv
fi

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

echo "===== Performance envs ====="
export PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
export LD_LIBRARY_PATH=${PLUGIN_PATH}/lib:${LD_LIBRARY_PATH}

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

echo "===== HydraGNN envs ====="
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

export HYDRAGNN_USE_FSDP=0
# export HYDRAGNN_FSDP_VERSION=2
# export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=100
export BATCH_SIZE=200
export NUM_EPOCH=20

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2

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

# Configuration
export NNODES=$SLURM_JOB_NUM_NODES # e.g., 100 total nodes
export NNODES_PER_TRIAL=256
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 )) # e.g., 800 total GPUs
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL )) # e.g., 32 GPUs per training
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL )) # e.g., 25 total DH ranks
export OMP_NUM_THREADS=7 # e.g., 8 threads per rank
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!!"

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR

# (A) Setup omnistat sampling environment
ml use /sw/frontier/amdsw/modulefiles/
ml omnistat-wrapper
export OMNISTAT_DIR=$OMNISTAT_DIR
export OMNISTAT_CONFIG=$HYDRAGNN_ROOT/omnistat.hydragnn-external-fp64.config

# (B) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 15

cmd python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_deephyper_multi_all_mpnn.py \
    --mpnn_type=SchNet
    # --mpnn_type=EGNN,SchNet,DimeNet,MACE,PAINN,PNAEq

# (C) End of job: stop data collection
${OMNISTAT_WRAPPER} usermode --stop
