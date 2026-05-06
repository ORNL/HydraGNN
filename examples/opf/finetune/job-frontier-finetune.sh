#!/bin/bash
# =============================================================================
#  Frontier fine-tuning job template
#
#  Usage: copy or symlink this file into any FT strategy directory, then edit
#  the FT_STRATEGY and FT_ARCH_REGIME variables below, or override from CLI:
#
#    sbatch --export=ALL,FT_STRATEGY=FT1_topology,FT_ARCH=HeteroSAGE,FT_REGIME=full \
#           job-frontier-finetune.sh
#
#  Two-phase workflow:
#   Phase 1 (--preonly)  : preprocessing/serialisation with N_PREONLY nodes
#   Phase 2 (training)   : fine-tuning with N_TRAIN nodes
#
#  To run both phases in one job, set PHASES=preonly,train (default).
#  To run only one phase: PHASES=preonly  or  PHASES=train
# =============================================================================
#SBATCH -A LRN078
#SBATCH -J OPF-FT
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/finetune-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/finetune-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 8

# -----------------------------------------------------------------------------
# Fine-tuning configuration — edit these or override via --export
# -----------------------------------------------------------------------------
FT_STRATEGY=${FT_STRATEGY:-FT2_operating_condition}  # FT2_operating_condition | FT3_contingency | FT4_task_specific  (use job-frontier-FT1-*.sh for FT1)
FT_ARCH=${FT_ARCH:-HeteroSAGE}                    # HeteroSAGE | HeteroHEAT
FT_REGIME=${FT_REGIME:-full}                      # full | partial | head_only
PRETRAINED_MODEL=${PRETRAINED_MODEL:-${FT_ARCH}_best}
PHASES=${PHASES:-preonly,train}                   # preonly,train | preonly | train
# Set NO_PRETRAINED=1 to skip loading pretrained weights (baseline comparison)
NO_PRETRAINED=${NO_PRETRAINED:-0}

# Preprocessing node count (smaller than training — single node is enough for
# fine-tuning datasets which are much smaller than the 3M pretraining corpus).
N_PREONLY=1
# Training node count
N_TRAIN=${SLURM_JOB_NUM_NODES}

# Optional: limit labeled samples for the fine-tuning split
MAX_SAMPLES=${MAX_SAMPLES:-}                      # leave empty to use all samples

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

# ROCm / libfabric tuning (same as pretraining jobs)
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
# Derived paths and names
# -----------------------------------------------------------------------------
FT_DIR=$HYDRAGNN_ROOT/examples/opf/finetune
INPUTFILE=$FT_DIR/${FT_STRATEGY}/config_${FT_ARCH}_${FT_REGIME}.json
_n_tag=${MAX_TRAIN_SAMPLES:+_n${MAX_TRAIN_SAMPLES}}
_scratch_tag=${NO_PRETRAINED:+_scratch}
LOG_NAME="finetune_${FT_STRATEGY}_${FT_ARCH}_${FT_REGIME}${_scratch_tag}${_n_tag}"
DATA_MODELNAME="${FT_STRATEGY}_data"

# Read case/group metadata from the config (informational only)
CASE_NAME=$(python3 -c "import json; c=json.load(open('$INPUTFILE')); print(c.get('_ft_case_name',''))" 2>/dev/null)
NUM_GROUPS=$(python3 -c "import json; c=json.load(open('$INPUTFILE')); print(c.get('_ft_num_groups','1'))" 2>/dev/null)
TOPO_PERTURB=$(python3 -c "import json; c=json.load(open('$INPUTFILE')); print('--topological_perturbations' if c.get('_ft_topological_perturbations') else '')" 2>/dev/null)

echo "============================================================"
echo " OPF Fine-tuning on Frontier"
echo "  Strategy : $FT_STRATEGY"
echo "  Arch     : $FT_ARCH  (pretrained: $PRETRAINED_MODEL)"
echo "  Regime   : $FT_REGIME"
echo "  Phases   : $PHASES"
echo "  Case     : $CASE_NAME  (groups: $NUM_GROUPS)"
echo "  Log name : $LOG_NAME"
echo "  Job ID   : $SLURM_JOB_ID"
echo "============================================================"

cd $FT_DIR

# Build optional --max_samples flag
MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

# Also read max_samples from config if not overridden from environment
if [ -z "$MAX_SAMPLES_FLAG" ]; then
    CFG_MAX=$(python3 -c "import json; c=json.load(open('$INPUTFILE')); v=c.get('_ft_max_samples'); print(v if v is not None else '')" 2>/dev/null)
    if [ -n "$CFG_MAX" ]; then
        MAX_SAMPLES_FLAG="--max_samples $CFG_MAX"
    fi
fi

# =============================================================================
# Phase 1: Preprocessing / serialisation
# =============================================================================
if [[ "$PHASES" == *"preonly"* ]]; then
    echo ""
    echo "--- Phase 1: Preprocessing (--preonly) ---"
    srun -N$N_PREONLY -n$((N_PREONLY * 8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u $HYDRAGNN_ROOT/examples/opf/train_opf_solution_heterogeneous.py \
            --inputfile $INPUTFILE \
            --hdf5 \
            --preonly \
            --case_name $CASE_NAME \
            --num_groups $NUM_GROUPS \
            --modelname $DATA_MODELNAME \
            --data_root $FT_DIR/../dataset \
            $TOPO_PERTURB \
            $MAX_SAMPLES_FLAG
    echo "--- Phase 1 complete ---"
fi

# =============================================================================
# Phase 2: Fine-tuning
# =============================================================================
if [[ "$PHASES" == *"train"* ]]; then
    echo ""
    echo "--- Phase 2: Fine-tuning ($FT_REGIME) with $FT_ARCH ---"
    srun -N$N_TRAIN -n$((N_TRAIN * 8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_finetune.py \
            --inputfile ${FT_STRATEGY}/config_${FT_ARCH}_${FT_REGIME}.json \
            --hdf5 \
            --modelname $LOG_NAME \
            --data_root $FT_DIR/../dataset \
            --pretrained_model_dir $HYDRAGNN_ROOT/examples/opf/pretrained_models \
            --pretrained_model_name $PRETRAINED_MODEL \
            --finetune_regime $FT_REGIME \
            ${NO_PRETRAINED:+--no_pretrained} \
            ${MAX_TRAIN_SAMPLES:+--max_train_samples $MAX_TRAIN_SAMPLES}
    echo "--- Phase 2 complete ---"
fi

echo ""
echo "Fine-tuning job finished. Logs: $HYDRAGNN_ROOT/logs/$LOG_NAME/"
