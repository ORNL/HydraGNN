#!/bin/bash
# =============================================================================
#  Frontier Slurm job — FT1 Feasibility Classification (HeteroSAGE)
#
#  FT1 pipeline differs from FT2-FT4 in two ways:
#    Phase 1 — generate the mixed feasible/infeasible HDF5 dataset with
#              generate_infeasible_samples.py  (runs once, shared by both archs)
#    Phase 2 — fine-tune with train_opf_ft1_classify.py  (graph-level BCE)
#
#  Usage:
#    sbatch job-frontier-FT1-HeteroSAGE.sh
#
#  Override regime or phases from CLI:
#    sbatch --export=ALL,FT_REGIME=partial,PHASES=train job-frontier-FT1-HeteroSAGE.sh
# =============================================================================
#SBATCH -A LRN078
#SBATCH -J OPF-FT1-SAGE
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/finetune-FT1-SAGE-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/finetune-FT1-SAGE-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 8

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FT_ARCH=HeteroSAGE
FT_REGIME=${FT_REGIME:-full}                  # full | partial | head_only
PRETRAINED_MODEL=${PRETRAINED_MODEL:-HeteroSAGE_best}
PHASES=${PHASES:-preonly,train}               # preonly,train | preonly | train
# Set NO_PRETRAINED=1 to train from random initialisation (baseline)
NO_PRETRAINED=${NO_PRETRAINED:-0}

# FT1 dataset name (shared across both archs)
FT1_DATASET=FT1_feasibility_data

# Source dataset for infeasible-sample generation.
# Defaults to FT3 contingency data (case118, N-1) — a diverse set of
# feasible samples.  Change to any other preprocessed OPF HDF5 directory.
SRC_DATASET=${SRC_DATASET:-FT3_contingency_HeteroSAGE_data}

# Factor by which load features (Pd, Qd) are multiplied to make instances
# infeasible.  Values >=5 reliably exceed generation capacity for pglib cases.
OVERLOAD_FACTOR=${OVERLOAD_FACTOR:-6.0}

# Maximum feasible samples to use (same number of infeasible samples generated)
MAX_SAMPLES=${MAX_SAMPLES:-5000}

N_TRAIN=${SLURM_JOB_NUM_NODES}

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

export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH
export PYTHONPATH=/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH

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
# Derived paths
# -----------------------------------------------------------------------------
FT_DIR=$HYDRAGNN_ROOT/examples/opf/finetune
DATA_ROOT=$FT_DIR/../dataset
LOG_NAME="FT1_feasibility_${FT_ARCH}_${FT_REGIME}_${SLURM_JOB_ID}"

SRC_DIR=$DATA_ROOT/${SRC_DATASET}.h5
OUT_DIR=$DATA_ROOT/${FT1_DATASET}.h5

echo "============================================================"
echo " FT1 Feasibility Classification on Frontier"
echo "  Arch             : $FT_ARCH  (pretrained: $PRETRAINED_MODEL)"
echo "  Regime           : $FT_REGIME"
echo "  Phases           : $PHASES"
echo "  Source dataset   : $SRC_DATASET.h5"
echo "  FT1 dataset      : $FT1_DATASET.h5"
echo "  Overload factor  : $OVERLOAD_FACTOR"
echo "  Max samples      : $MAX_SAMPLES (per class)"
echo "  Log name         : $LOG_NAME"
echo "  Job ID           : $SLURM_JOB_ID"
echo "============================================================"

cd $FT_DIR

# =============================================================================
# Phase 1: Generate mixed feasible/infeasible dataset
# =============================================================================
if [[ "$PHASES" == *"preonly"* ]]; then
    echo ""
    echo "--- Phase 1: Generating FT1 feasibility dataset ---"

    if [ -d "$OUT_DIR" ]; then
        echo "  Dataset already exists at $OUT_DIR — skipping generation."
        echo "  Delete $OUT_DIR to regenerate."
    else
        # Single-rank generation (no MPI parallelism needed)
        srun -N1 -n1 -c7 \
            python -u generate_infeasible_samples.py \
                --src_dir  $SRC_DIR \
                --out_dir  $OUT_DIR \
                --overload_factor $OVERLOAD_FACTOR \
                --max_samples $MAX_SAMPLES
        echo "--- Phase 1 complete: $(ls $OUT_DIR) ---"
    fi
fi

# =============================================================================
# Phase 2: Fine-tune for binary classification
# =============================================================================
if [[ "$PHASES" == *"train"* ]]; then
    echo ""
    echo "--- Phase 2: FT1 classification fine-tuning ($FT_REGIME) with $FT_ARCH ---"
    srun -N$N_TRAIN -n$((N_TRAIN * 8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_ft1_classify.py \
            --inputfile FT1_feasibility_classification/config_${FT_ARCH}_${FT_REGIME}.json \
            --modelname $LOG_NAME \
            --data_root $DATA_ROOT \
            --pretrained_model_dir $HYDRAGNN_ROOT/examples/opf/pretrained_models \
            --pretrained_model_name $PRETRAINED_MODEL \
            --finetune_regime $FT_REGIME \
            ${NO_PRETRAINED:+--no_pretrained}
    echo "--- Phase 2 complete ---"
fi

echo ""
echo "FT1 job finished. Logs: $HYDRAGNN_ROOT/logs/$LOG_NAME/"
