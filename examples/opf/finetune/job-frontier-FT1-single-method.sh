#!/bin/bash
# =============================================================================
#  Frontier Slurm job — FT1 Feasibility Classification (single method)
#
#  Runs ONE training method for all sample sizes sequentially.
#  Uses only N_PER_RUN nodes (default: 8) — 4x smaller than the bundled job.
#
#  Required env vars:
#    FT_ARCH         HeteroSAGE | HeteroHEAT
#    FT_METHOD       full | partial | head_only | scratch
#
#  Optional env vars:
#    N_PER_RUN       nodes (default: 8)
#    PRETRAINED_MODEL
#    SAMPLE_SIZES    space-separated list
#                    (default: 100 500 1000 2500 5000 10000 25000 50000 100000)
#
#  Usage:
#    sbatch --export=ALL,FT_ARCH=HeteroSAGE,FT_METHOD=full \
#           job-frontier-FT1-single-method.sh
# =============================================================================
#SBATCH -A eng164
#SBATCH -J FT1-single
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT1-single-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT1-single-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FT_ARCH=${FT_ARCH:-HeteroSAGE}
FT_METHOD=${FT_METHOD:-full}
N_PER_RUN=${N_PER_RUN:-1}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-${FT_ARCH}_best}
read -ra SAMPLE_SIZES <<< "${SAMPLE_SIZES:-100 500 1000 2500 5000 10000 25000 50000 100000}"

# scratch baseline uses full regime + --no_pretrained flag
SCRATCH_FLAG=""
REGIME="$FT_METHOD"
if [[ "$FT_METHOD" == "scratch" ]]; then
    REGIME="full"
    SCRATCH_FLAG="--no_pretrained"
fi

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

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1

# Single-node NCCL: AWS OFI / libfabric cxi plugin cannot bind without a remote
# peer. Disable network plugin so NCCL uses intra-node xGMI/SHM only.
if [[ "${N_PER_RUN:-1}" -le 1 ]]; then
    unset NCCL_NET_PLUGIN NCCL_NET NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_SOCKET_IFNAME
    export NCCL_P2P_LEVEL=SYS
fi

# -----------------------------------------------------------------------------
# Derived
# -----------------------------------------------------------------------------
FT_DIR=$HYDRAGNN_ROOT/examples/opf/finetune
DATA_ROOT=$FT_DIR/../dataset
_scratch_tag=""
[[ -n "$SCRATCH_FLAG" ]] && _scratch_tag="_scratch"

echo "============================================================"
echo " FT1 Feasibility Classification — single method"
echo "  Arch          : $FT_ARCH  (pretrained: $PRETRAINED_MODEL)"
echo "  Method        : $FT_METHOD  (regime: $REGIME)"
echo "  Nodes         : $N_PER_RUN"
echo "  Sample sizes  : ${SAMPLE_SIZES[*]}"
echo "  Job ID        : $SLURM_JOB_ID"
echo "============================================================"

cd $FT_DIR

ROUND=0
for N in "${SAMPLE_SIZES[@]}"; do
    ROUND=$((ROUND + 1))
    TOTAL=$((N * 2))
    # NOTE: LOG_NAME is keyed on the *requested* training-sample count N
    # (not TOTAL=N*2 which also includes the held-out class sub-sample),
    # so each sweep point gets its own dir and never overwrites another.
    LOG_NAME="FT1_feasibility_${FT_ARCH}_${REGIME}${_scratch_tag}_n${N}"
    LOGFILE="$HYDRAGNN_ROOT/${LOG_NAME}-${SLURM_JOB_ID}.out"

    echo ""
    echo "── Round $ROUND / ${#SAMPLE_SIZES[@]}: N=$N (total=$TOTAL) → $LOGFILE"

    srun -N${N_PER_RUN} -n$((N_PER_RUN * 8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_ft1_classify.py \
            --inputfile FT1_feasibility_classification/config_${FT_ARCH}_${REGIME}.json \
            --modelname "$LOG_NAME" \
            --data_root $DATA_ROOT \
            --pretrained_model_dir $HYDRAGNN_ROOT/examples/opf/pretrained_models \
            --pretrained_model_name $PRETRAINED_MODEL \
            --finetune_regime $REGIME \
            $SCRATCH_FLAG \
            --max_train_samples $TOTAL \
        > "$LOGFILE" 2>&1

    echo "  Round $ROUND done."
done

echo ""
echo "All rounds finished. Job ID: $SLURM_JOB_ID"
