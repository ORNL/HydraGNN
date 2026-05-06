#!/bin/bash
# =============================================================================
#  Frontier Slurm job — FT3 N-1 Contingency OPF Regression (multi-method)
#
#  Runs all 4 training methods in parallel within a single allocation:
#    - full fine-tuning (all layers)
#    - partial fine-tuning (last conv + head)
#    - head-only fine-tuning (linear probe)
#    - from-scratch baseline (random init)
#
#  Each method gets N_PER_RUN nodes; total allocation = 4 × N_PER_RUN.
#
#  Required env vars (set via sbatch --export or environment):
#    FT_ARCH             HeteroSAGE | HeteroHEAT
#
#  Optional env vars:
#    MAX_TRAIN_SAMPLES   limit training samples (default: use all)
#    N_PER_RUN           nodes per method (default: 8)
#    PRETRAINED_MODEL    override pretrained model name
#
#  Usage (from examples/opf/finetune/):
#    sbatch --export=ALL,FT_ARCH=HeteroSAGE,MAX_TRAIN_SAMPLES=5000 \
#           job-frontier-FT3-multi.sh
# =============================================================================
#SBATCH -A LRN070
#SBATCH -J OPF-FT3-MULTI
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT3-multi-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT3-multi-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 32

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FT_ARCH=${FT_ARCH:-HeteroSAGE}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
N_PER_RUN=${N_PER_RUN:-8}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-${FT_ARCH}_best}
FT_STRATEGY=FT3_contingency

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate /lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

export PYTHONPATH=$HYDRAGNN_ROOT:${PYTHONPATH:-}
export PYTHONPATH=/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:${PYTHONPATH:-}

module unload darshan-runtime

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

# -----------------------------------------------------------------------------
# Derived
# -----------------------------------------------------------------------------
FT_DIR=$HYDRAGNN_ROOT/examples/opf/finetune
DATA_ROOT=$FT_DIR/../dataset
_n_tag=${MAX_TRAIN_SAMPLES:+_n${MAX_TRAIN_SAMPLES}}

echo "============================================================"
echo " FT3 N-1 Contingency OPF — multi-method job"
echo "  Arch          : $FT_ARCH  (pretrained: $PRETRAINED_MODEL)"
echo "  Methods       : full / partial / head_only / scratch (parallel)"
echo "  Nodes per run : $N_PER_RUN  (total: $((N_PER_RUN * 4)))"
echo "  Max train samples: ${MAX_TRAIN_SAMPLES:-all}"
echo "  Job ID        : $SLURM_JOB_ID"
echo "============================================================"

cd $FT_DIR

# Helper: launch one srun in the background, redirect output to its own log
_launch() {
    local REGIME="$1"
    local SCRATCH_FLAG="${2:-}"
    local _scratch_tag=""
    [[ -n "$SCRATCH_FLAG" ]] && _scratch_tag="_scratch"
    local LOG_NAME="finetune_${FT_STRATEGY}_${FT_ARCH}_${REGIME}${_scratch_tag}${_n_tag}"
    local LOGFILE="$HYDRAGNN_ROOT/${LOG_NAME}-${SLURM_JOB_ID}.out"

    echo "  Launching $LOG_NAME → $LOGFILE"

    srun --exact -N${N_PER_RUN} -n$((N_PER_RUN * 8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_finetune.py \
            --inputfile ${FT_STRATEGY}/config_${FT_ARCH}_${REGIME}.json \
            --hdf5 \
            --modelname "$LOG_NAME" \
            --data_root $DATA_ROOT \
            --pretrained_model_dir $HYDRAGNN_ROOT/examples/opf/pretrained_models \
            --pretrained_model_name $PRETRAINED_MODEL \
            --finetune_regime $REGIME \
            $SCRATCH_FLAG \
            ${MAX_TRAIN_SAMPLES:+--max_train_samples $MAX_TRAIN_SAMPLES} \
        > "$LOGFILE" 2>&1 &
}

# Launch all 4 methods in parallel
_launch full
_launch partial
_launch head_only
_launch full "--no_pretrained"  # scratch baseline

echo ""
echo "All 4 methods launched. Waiting for completion..."
wait
echo ""
echo "All methods finished. Job ID: $SLURM_JOB_ID"
