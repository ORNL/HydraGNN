#!/bin/bash
# =============================================================================
#  submit_ft3_data_efficiency_sweep.sh
#
#  Data-efficiency experiment for FT3 (N-1 contingency OPF regression):
#  compare fine-tuning a pre-trained graph foundation model vs. training from
#  scratch for progressively increasing numbers of training samples.
#
#  Experiment grid:
#    sample_sizes  : SAMPLE_SIZES array below
#    methods       : full FT, partial FT, head_only FT, from-scratch baseline
#    architectures : HeteroSAGE, HeteroHEAT
#
#  Requirements:
#    - FT3 contingency dataset already serialised at
#      examples/opf/dataset/FT3_contingency_data.h5
#    - Pretrained models in examples/opf/pretrained_models/
#
#  Usage (from examples/opf/finetune/):
#    bash submit_ft3_data_efficiency_sweep.sh [--dry-run]
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
SLURM_OUT_DIR="$HYDRAGNN_ROOT"

PROJECT=${PROJECT:-LRN070}
PARTITION=${PARTITION:-batch}
N_NODES=${N_NODES:-8}
WALL_TIME=${WALL_TIME:-04:00:00}

# FT3 has 270k train samples available (full FT3_contingency_data.h5 trainset)
SAMPLE_SIZES=(100 500 1000 2500 5000 10000 25000 50000 100000 270000)

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ─────────────────────────────────────────────────────────────────────────────
# Submission helper
# ─────────────────────────────────────────────────────────────────────────────
_sbatch() {
    local label="$1"; shift
    if $DRY_RUN; then
        echo "[dry-run] sbatch $*"
    else
        local id
        id=$(sbatch "$@" | awk '{print $NF}')
        echo "[submit] $label → job $id"
    fi
}

echo "========================================================"
echo " FT3 data-efficiency sweep"
echo "  Account    : $PROJECT"
echo "  Partition  : $PARTITION"
echo "  Nodes      : $N_NODES"
echo "  Wall time  : $WALL_TIME"
echo "  Sample sizes: ${SAMPLE_SIZES[*]}"
echo "========================================================"

for ARCH in HeteroSAGE HeteroHEAT; do
    for N in "${SAMPLE_SIZES[@]}"; do

        # ── full fine-tuning (all layers) ────────────────────────────────
        LABEL="FT3_${ARCH}_FT_full_n${N}"
        echo "--- $LABEL ---"
        _sbatch "$LABEL" \
            --account=$PROJECT --partition=$PARTITION \
            --job-name="$LABEL" \
            --nodes=$N_NODES --time=$WALL_TIME \
            --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=$ARCH,FT_REGIME=full,PHASES=train,MAX_TRAIN_SAMPLES=$N \
            "$FT_DIR/job-frontier-finetune.sh"

        # ── partial fine-tuning (last conv + head) ───────────────────────
        LABEL="FT3_${ARCH}_FT_partial_n${N}"
        echo "--- $LABEL ---"
        _sbatch "$LABEL" \
            --account=$PROJECT --partition=$PARTITION \
            --job-name="$LABEL" \
            --nodes=$N_NODES --time=$WALL_TIME \
            --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=$ARCH,FT_REGIME=partial,PHASES=train,MAX_TRAIN_SAMPLES=$N \
            "$FT_DIR/job-frontier-finetune.sh"

        # ── head-only fine-tuning (linear probe) ─────────────────────────
        LABEL="FT3_${ARCH}_FT_head_only_n${N}"
        echo "--- $LABEL ---"
        _sbatch "$LABEL" \
            --account=$PROJECT --partition=$PARTITION \
            --job-name="$LABEL" \
            --nodes=$N_NODES --time=$WALL_TIME \
            --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=$ARCH,FT_REGIME=head_only,PHASES=train,MAX_TRAIN_SAMPLES=$N \
            "$FT_DIR/job-frontier-finetune.sh"

        # ── from scratch baseline (random init, full regime) ─────────────
        LABEL="FT3_${ARCH}_scratch_n${N}"
        echo "--- $LABEL ---"
        _sbatch "$LABEL" \
            --account=$PROJECT --partition=$PARTITION \
            --job-name="$LABEL" \
            --nodes=$N_NODES --time=$WALL_TIME \
            --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=$ARCH,FT_REGIME=full,PHASES=train,NO_PRETRAINED=1,MAX_TRAIN_SAMPLES=$N \
            "$FT_DIR/job-frontier-finetune.sh"
    done
done

echo ""
echo "All jobs submitted."
echo "Total jobs: $((${#SAMPLE_SIZES[@]} * 4 * 2))  (${#SAMPLE_SIZES[@]} sizes × 4 methods × 2 archs)"
