#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN-DeepHyper
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -N 4
#SBATCH --network=disable_rdzv_get
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
export PYTHONPATH=$PWD:$HYDRAGNN_ROOT:$PYTHONPATH

#export python path to use ADIOS2 v.2.10.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH

which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

module unload darshan-runtime
module list

echo $LD_LIBRARY_PATH | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

# NCCL/RCCL tuning
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export FI_MR_CACHE_MONITOR=disabled
export TORCH_NCCL_HIGH_PRIORITY=1
export FI_CXI_RDV_PROTO=alt_read

export PATH_TO_THE_PLUGIN_DIRECTORY=/lustre/orion/lrn070/world-shared/mlupopa/AWI_OFI_RCCL_ROCm631/aws-ofi-rccl/lib
export LD_LIBRARY_PATH=${PATH_TO_THE_PLUGIN_DIRECTORY}:$LD_LIBRARY_PATH

export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid

export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0

# DeepHyper runtime configuration
export NNODES=${SLURM_JOB_NUM_NODES}
export NTOTGPUS=$((SLURM_JOB_NUM_NODES * 8))
export NNODES_PER_TRIAL=${NNODES_PER_TRIAL:-1}
export NGPUS_PER_TRIAL=${NGPUS_PER_TRIAL:-8}
export NUM_CONCURRENT_TRIALS=${NUM_CONCURRENT_TRIALS:-$((NTOTGPUS / NGPUS_PER_TRIAL))}
export NTOT_DEEPHYPER_RANKS=${NTOT_DEEPHYPER_RANKS:-$NUM_CONCURRENT_TRIALS}
export DEEPHYPER_LOG_DIR=${DEEPHYPER_LOG_DIR:-$PWD/deephyper_logs}
export DEEPHYPER_DB_HOST=${DEEPHYPER_DB_HOST:-localhost}

mkdir -p "$DEEPHYPER_LOG_DIR"

# (A) Setup omnistat sampling environment
ml use /sw/frontier/amdsw/modulefiles/
ml omnistat-wrapper
export OMNISTAT_CONFIG=$HYDRAGNN_ROOT/omnistat.hydragnn-external-fp64.config

# (B) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 1

cd ./examples/multidataset_hpo_sc26

${OMNISTAT_DIR}/omnistat-annotate --mode start --text "DeepHyper HPO"
cmd python -u $PWD/gfm_deephyper_multi_all_mpnn.py
${OMNISTAT_DIR}/omnistat-annotate --mode stop
sleep 10

# (C) End of job: stop data collection
${OMNISTAT_WRAPPER} usermode --stop
