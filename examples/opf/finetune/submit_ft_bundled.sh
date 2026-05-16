#!/bin/bash
# =============================================================================
#  submit_ft_bundled.sh  —  Submit all FT1 + FT3 data-efficiency sweeps as
#                           8 bundled Slurm jobs (2 per arch × task).
#
#  Each job chains sample sizes sequentially; within each size the 4 training
#  methods (full / partial / head_only / scratch) run concurrently via srun.
#  Split into small/large bundles to stay within the 2h wall-time cap.
#
#  Jobs submitted (per arch, ×2 for HeteroSAGE + HeteroHEAT):
#    FT1-small   n = 100 500 1000 2500 5000
#    FT1-large   n = 10000 25000 50000 100000
#    FT3-small   n = 100 500 1000 2500 5000
#    FT3-large   n = 10000 25000 50000 100000 270000
#
#  Usage (from examples/opf/finetune/):
#    bash submit_ft_bundled.sh [--dry-run]
# =============================================================================
set -euo pipefail

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
SLURM_OUT_DIR="$HYDRAGNN_ROOT"

PROJECT=${PROJECT:-eng164}
PARTITION=${PARTITION:-batch}
N_PER_RUN=${N_PER_RUN:-8}
N_NODES=$((N_PER_RUN * 4))   # 32 total nodes per job
WALL_TIME=${WALL_TIME:-02:00:00}

FT1_SMALL_SIZES="100 500 1000 2500 5000"
FT1_LARGE_SIZES="10000 25000 50000 100000"
FT3_SMALL_SIZES="100 500 1000 2500 5000"
FT3_LARGE_SIZES="10000 25000 50000 100000 270000"

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
echo " FT bundled data-efficiency sweep"
echo "  Account   : $PROJECT"
echo "  Partition : $PARTITION"
echo "  Nodes     : $N_NODES  (${N_PER_RUN} per method × 4 methods)"
echo "  Wall time : $WALL_TIME"
echo "  Jobs      : 8  (FT1+FT3 × small+large × HeteroSAGE+HeteroHEAT)"
echo "========================================================"

for ARCH in HeteroSAGE HeteroHEAT; do
    # ── FT1 small ────────────────────────────────────────────────────────
    LABEL="FT1-small-${ARCH}"
    echo "--- $LABEL (${FT1_SMALL_SIZES}) ---"
    _sbatch "$LABEL" \
        --account=$PROJECT --partition=$PARTITION \
        --job-name="$LABEL" \
        --nodes=$N_NODES --time=$WALL_TIME \
        --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --export=ALL,FT_ARCH=$ARCH,N_PER_RUN=$N_PER_RUN,SAMPLE_SIZES="$FT1_SMALL_SIZES" \
        "$FT_DIR/job-frontier-FT1-bundled.sh"

    # ── FT1 large ────────────────────────────────────────────────────────
    LABEL="FT1-large-${ARCH}"
    echo "--- $LABEL (${FT1_LARGE_SIZES}) ---"
    _sbatch "$LABEL" \
        --account=$PROJECT --partition=$PARTITION \
        --job-name="$LABEL" \
        --nodes=$N_NODES --time=$WALL_TIME \
        --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --export=ALL,FT_ARCH=$ARCH,N_PER_RUN=$N_PER_RUN,SAMPLE_SIZES="$FT1_LARGE_SIZES" \
        "$FT_DIR/job-frontier-FT1-bundled.sh"

    # ── FT3 small ────────────────────────────────────────────────────────
    LABEL="FT3-small-${ARCH}"
    echo "--- $LABEL (${FT3_SMALL_SIZES}) ---"
    _sbatch "$LABEL" \
        --account=$PROJECT --partition=$PARTITION \
        --job-name="$LABEL" \
        --nodes=$N_NODES --time=$WALL_TIME \
        --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --export=ALL,FT_ARCH=$ARCH,N_PER_RUN=$N_PER_RUN,SAMPLE_SIZES="$FT3_SMALL_SIZES" \
        "$FT_DIR/job-frontier-FT3-bundled.sh"

    # ── FT3 large ────────────────────────────────────────────────────────
    LABEL="FT3-large-${ARCH}"
    echo "--- $LABEL (${FT3_LARGE_SIZES}) ---"
    _sbatch "$LABEL" \
        --account=$PROJECT --partition=$PARTITION \
        --job-name="$LABEL" \
        --nodes=$N_NODES --time=$WALL_TIME \
        --output="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --error="$SLURM_OUT_DIR/${LABEL}-%j.out" \
        --export=ALL,FT_ARCH=$ARCH,N_PER_RUN=$N_PER_RUN,SAMPLE_SIZES="$FT3_LARGE_SIZES" \
        "$FT_DIR/job-frontier-FT3-bundled.sh"
done

echo ""
echo "Done. 8 jobs submitted."
