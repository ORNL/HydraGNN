#!/bin/bash
# =============================================================================
#  submit_ft1_ft3_debug.sh  —  Debug-queue submission for FT1 + FT3
#
#  Submits fine-tuning jobs for the two best HPO architectures (HeteroSAGE
#  and HeteroHEAT) under FT1 (feasibility classification) and FT3
#  (contingency) strategies using the Frontier debug queue.
#
#  Assumptions:
#    - Preprocessed datasets already exist (PHASES=train only).
#    - Pretrained models are in examples/opf/pretrained_models/.
#
#  Usage (from examples/opf/finetune/):
#    bash submit_ft1_ft3_debug.sh [--dry-run]
#
#  Flags:
#    --dry-run   Print sbatch commands without submitting.
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
SLURM_OUT_DIR="$HYDRAGNN_ROOT"

PROJECT=eng164
PARTITION=debug
N_NODES=2          # debug queue max
WALL_TIME=00:30:00 # debug queue max

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ─────────────────────────────────────────────────────────────────────────────
# Submission helper
# ─────────────────────────────────────────────────────────────────────────────
_sbatch() {
    local label="$1"
    shift
    if $DRY_RUN; then
        echo "[dry-run] sbatch $*"
    else
        local id
        id=$(sbatch "$@" | awk '{print $NF}')
        echo "[submit] $label → job $id"
    fi
}

echo "========================================================"
echo " FT1 + FT3 debug submissions"
echo "  Account   : $PROJECT"
echo "  Partition : $PARTITION"
echo "  Nodes     : $N_NODES"
echo "  Wall time : $WALL_TIME"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# FT1 — Feasibility classification (HeteroSAGE and HeteroHEAT)
# Assumes FT1 feasibility dataset already generated (PHASES=train).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- FT1: HeteroSAGE ---"
_sbatch "FT1_HeteroSAGE" \
    --account=$PROJECT --partition=$PARTITION \
    --job-name=FT1-SAGE-dbg \
    --nodes=$N_NODES --time=$WALL_TIME \
    --output="$SLURM_OUT_DIR/FT1-SAGE-dbg-%j.out" \
    --error="$SLURM_OUT_DIR/FT1-SAGE-dbg-%j.out" \
    --export=ALL,FT_REGIME=full,PHASES=train \
    "$FT_DIR/FT1_feasibility_classification/job-frontier-FT1-HeteroSAGE.sh"

echo "--- FT1: HeteroHEAT ---"
_sbatch "FT1_HeteroHEAT" \
    --account=$PROJECT --partition=$PARTITION \
    --job-name=FT1-HEAT-dbg \
    --nodes=$N_NODES --time=$WALL_TIME \
    --output="$SLURM_OUT_DIR/FT1-HEAT-dbg-%j.out" \
    --error="$SLURM_OUT_DIR/FT1-HEAT-dbg-%j.out" \
    --export=ALL,FT_REGIME=full,PHASES=train \
    "$FT_DIR/FT1_feasibility_classification/job-frontier-FT1-HeteroHEAT.sh"

# ─────────────────────────────────────────────────────────────────────────────
# FT3 — Contingency fine-tuning (HeteroSAGE and HeteroHEAT)
# Assumes FT3 contingency dataset already preprocessed (PHASES=train).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- FT3: HeteroSAGE ---"
_sbatch "FT3_HeteroSAGE" \
    --account=$PROJECT --partition=$PARTITION \
    --job-name=FT3-SAGE-dbg \
    --nodes=$N_NODES --time=$WALL_TIME \
    --output="$SLURM_OUT_DIR/FT3-SAGE-dbg-%j.out" \
    --error="$SLURM_OUT_DIR/FT3-SAGE-dbg-%j.out" \
    --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=HeteroSAGE,FT_REGIME=full,PHASES=train \
    "$FT_DIR/FT3_contingency/job-frontier-HeteroSAGE.sh"

echo "--- FT3: HeteroHEAT ---"
_sbatch "FT3_HeteroHEAT" \
    --account=$PROJECT --partition=$PARTITION \
    --job-name=FT3-HEAT-dbg \
    --nodes=$N_NODES --time=$WALL_TIME \
    --output="$SLURM_OUT_DIR/FT3-HEAT-dbg-%j.out" \
    --error="$SLURM_OUT_DIR/FT3-HEAT-dbg-%j.out" \
    --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=HeteroHEAT,FT_REGIME=full,PHASES=train \
    "$FT_DIR/FT3_contingency/job-frontier-HeteroHEAT.sh"

echo ""
echo "All jobs submitted."
