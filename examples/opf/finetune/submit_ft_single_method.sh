#!/bin/bash
# =============================================================================
#  submit_ft_single_method.sh
#
#  Submits all FT1 + FT3 data-efficiency runs as 16 small jobs:
#    2 archs × 2 tasks × 4 methods = 16 jobs × 8 nodes each
#
#  Each job chains all sample sizes sequentially for one method.
#  Much easier to backfill than 8 × 32-node jobs.
#
#  Usage:
#    bash submit_ft_single_method.sh [--dry-run]
# =============================================================================
set -euo pipefail

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
SLURM_OUT_DIR="$HYDRAGNN_ROOT"

PROJECT=${PROJECT:-eng164}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-normal}
N_PER_RUN=${N_PER_RUN:-8}
WALL_TIME=${WALL_TIME:-02:00:00}

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

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
echo " FT single-method data-efficiency sweep"
echo "  Account   : $PROJECT"
echo "  Partition : $PARTITION"
echo "  QOS       : $QOS"
echo "  Nodes     : $N_PER_RUN per job  (4x smaller than bundled)"
echo "  Wall time : $WALL_TIME"
echo "  Jobs      : 16  (2 archs × 2 tasks × 4 methods)"
echo "========================================================"

for ARCH in HeteroSAGE HeteroHEAT; do
    for METHOD in full partial head_only scratch; do

        # ── FT1 ──────────────────────────────────────────────────────────
        LABEL="FT1-${ARCH}-${METHOD}"
        echo "--- $LABEL ---"
        _sbatch "$LABEL" \
            --account=$PROJECT --partition=$PARTITION --qos=$QOS \
            --job-name="$LABEL" \
            --nodes=$N_PER_RUN --time=$WALL_TIME \
            --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --export=ALL,FT_ARCH=$ARCH,FT_METHOD=$METHOD,N_PER_RUN=$N_PER_RUN \
            "$FT_DIR/job-frontier-FT1-single-method.sh"

        # ── FT3 ──────────────────────────────────────────────────────────
        LABEL="FT3-${ARCH}-${METHOD}"
        echo "--- $LABEL ---"
        _sbatch "$LABEL" \
            --account=$PROJECT --partition=$PARTITION --qos=$QOS \
            --job-name="$LABEL" \
            --nodes=$N_PER_RUN --time=$WALL_TIME \
            --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
            --export=ALL,FT_ARCH=$ARCH,FT_METHOD=$METHOD,N_PER_RUN=$N_PER_RUN \
            "$FT_DIR/job-frontier-FT3-single-method.sh"

    done
done

echo ""
echo "Done. 16 jobs submitted (8 nodes each)."
