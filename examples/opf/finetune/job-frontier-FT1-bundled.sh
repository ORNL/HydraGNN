#!/bin/bash
# =============================================================================
#  Frontier Slurm job — FT1 Feasibility Classification (bundled data-efficiency)
#
#  Runs all sample sizes sequentially; within each sample size the 4 training
#  methods (full / partial / head_only / scratch) run concurrently with srun.
#
#  Required env vars (set via sbatch --export or environment):
#    FT_ARCH             HeteroSAGE | HeteroHEAT
#
#  Optional env vars:
#    N_PER_RUN           nodes per method (default: 8)
#    PRETRAINED_MODEL    override pretrained model name
#    SAMPLE_SIZES        space-separated list (default: 100 500 1000 2500 5000
#                        10000 25000 50000 100000)
#
#  Usage (from examples/opf/finetune/):
#    sbatch --export=ALL,FT_ARCH=HeteroSAGE \
#           job-frontier-FT1-bundled.sh
# =============================================================================
#SBATCH -A eng164
#SBATCH -J FT1-bundled
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT1-bundled-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT1-bundled-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 32

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FT_ARCH=${FT_ARCH:-HeteroSAGE}
N_PER_RUN=${N_PER_RUN:-8}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-${FT_ARCH}_best}
read -ra SAMPLE_SIZES <<< "${SAMPLE_SIZES:-100 500 1000 2500 5000 10000 25000 50000 100000}"

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

echo "============================================================"
echo " FT1 Feasibility Classification — bundled data-efficiency"
echo "  Arch          : $FT_ARCH  (pretrained: $PRETRAINED_MODEL)"
echo "  Methods       : full / partial / head_only / scratch (concurrent)"
echo "  Nodes per run : $N_PER_RUN  (total: $((N_PER_RUN * 4)))"
echo "  Sample sizes  : ${SAMPLE_SIZES[*]}"
echo "  Job ID        : $SLURM_JOB_ID"
echo "============================================================"

cd $FT_DIR

# Helper: launch one srun in the background for a given regime and sample size
_launch() {
    local REGIME="$1"
    local MAX_TRAIN_SAMPLES="$2"
    local SCRATCH_FLAG="${3:-}"
    local _scratch_tag=""
    [[ -n "$SCRATCH_FLAG" ]] && _scratch_tag="_scratch"
    local LOG_NAME="FT1_feasibility_${FT_ARCH}_${REGIME}${_scratch_tag}_n${MAX_TRAIN_SAMPLES}"
    local LOGFILE="$HYDRAGNN_ROOT/${LOG_NAME}-${SLURM_JOB_ID}.out"

    echo "  Launching $LOG_NAME → $LOGFILE"

    srun --exact -N${N_PER_RUN} -n$((N_PER_RUN * 8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_ft1_classify.py \
            --inputfile FT1_feasibility_classification/config_${FT_ARCH}_${REGIME}.json \
            --modelname "$LOG_NAME" \
            --data_root $DATA_ROOT \
            --pretrained_model_dir $HYDRAGNN_ROOT/examples/opf/pretrained_models \
            --pretrained_model_name $PRETRAINED_MODEL \
            --finetune_regime $REGIME \
            $SCRATCH_FLAG \
            --max_train_samples $MAX_TRAIN_SAMPLES \
        > "$LOGFILE" 2>&1 &
}

# -----------------------------------------------------------------------------
# Main loop: iterate sample sizes sequentially, methods concurrently
# -----------------------------------------------------------------------------
ROUND=0
for N in "${SAMPLE_SIZES[@]}"; do
    ROUND=$((ROUND + 1))
    TOTAL=$((N * 2))   # balanced: N feasible + N infeasible
    echo ""
    echo "── Round $ROUND / ${#SAMPLE_SIZES[@]}: N=$N (total train samples=$TOTAL) ──"

    _launch full       "$TOTAL"
    _launch partial    "$TOTAL"
    _launch head_only  "$TOTAL"
    _launch full       "$TOTAL" "--no_pretrained"   # scratch baseline

    echo "  Waiting for round $ROUND to finish..."
    wait
    echo "  Round $ROUND done."
done

echo ""
echo "All rounds finished. Job ID: $SLURM_JOB_ID"
