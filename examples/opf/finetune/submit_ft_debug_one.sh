#!/bin/bash
# =============================================================================
#  submit_ft_debug_one.sh
#
#  Submit exactly one FT bundled job to Frontier debug QOS.
#  Debug is single-job-per-user on this system, so this script supports:
#    - immediate submit (default)
#    - wait-until-slot-opens then submit (--wait-slot)
#
#  Usage:
#    bash submit_ft_debug_one.sh --target FT1-small-HeteroSAGE
#    bash submit_ft_debug_one.sh --target FT3-small-HeteroHEAT --wait-slot
#    bash submit_ft_debug_one.sh --target FT1-large-HeteroSAGE --dry-run
#
#  Valid targets:
#    FT1-small-HeteroSAGE
#    FT1-large-HeteroSAGE
#    FT1-small-HeteroHEAT
#    FT1-large-HeteroHEAT
#    FT3-small-HeteroSAGE
#    FT3-large-HeteroSAGE
#    FT3-small-HeteroHEAT
#    FT3-large-HeteroHEAT
# =============================================================================
set -euo pipefail

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
SLURM_OUT_DIR="$HYDRAGNN_ROOT"

PROJECT=${PROJECT:-eng164}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-debug}
N_PER_RUN=${N_PER_RUN:-8}
N_NODES=$((N_PER_RUN * 4))
WALL_TIME=${WALL_TIME:-02:00:00}

TARGET=""
WAIT_SLOT=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="${2:-}"
            shift 2
            ;;
        --wait-slot)
            WAIT_SLOT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            sed -n '1,42p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    echo "Error: --target is required."
    exit 1
fi

SCRIPT_PATH=""
FT_ARCH=""
SAMPLE_SIZES=""

case "$TARGET" in
    FT1-small-HeteroSAGE)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT1-bundled.sh"
        FT_ARCH="HeteroSAGE"
        SAMPLE_SIZES="100 500 1000 2500 5000"
        ;;
    FT1-large-HeteroSAGE)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT1-bundled.sh"
        FT_ARCH="HeteroSAGE"
        SAMPLE_SIZES="10000 25000 50000 100000"
        ;;
    FT1-small-HeteroHEAT)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT1-bundled.sh"
        FT_ARCH="HeteroHEAT"
        SAMPLE_SIZES="100 500 1000 2500 5000"
        ;;
    FT1-large-HeteroHEAT)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT1-bundled.sh"
        FT_ARCH="HeteroHEAT"
        SAMPLE_SIZES="10000 25000 50000 100000"
        ;;
    FT3-small-HeteroSAGE)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT3-bundled.sh"
        FT_ARCH="HeteroSAGE"
        SAMPLE_SIZES="100 500 1000 2500 5000"
        ;;
    FT3-large-HeteroSAGE)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT3-bundled.sh"
        FT_ARCH="HeteroSAGE"
        SAMPLE_SIZES="10000 25000 50000 100000 270000"
        ;;
    FT3-small-HeteroHEAT)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT3-bundled.sh"
        FT_ARCH="HeteroHEAT"
        SAMPLE_SIZES="100 500 1000 2500 5000"
        ;;
    FT3-large-HeteroHEAT)
        SCRIPT_PATH="$FT_DIR/job-frontier-FT3-bundled.sh"
        FT_ARCH="HeteroHEAT"
        SAMPLE_SIZES="10000 25000 50000 100000 270000"
        ;;
    *)
        echo "Invalid --target: $TARGET"
        exit 1
        ;;
esac

_debug_jobs_active() {
    squeue -u "$USER" -h -o "%q %T" | awk '$1=="debug" && $2 ~ /PENDING|RUNNING|CONFIGURING/{n++} END{print n+0}'
}

if ! $DRY_RUN; then
    if $WAIT_SLOT; then
        echo "Waiting for a free debug QOS slot..."
        while true; do
            ACTIVE=$(_debug_jobs_active)
            if [[ "$ACTIVE" -eq 0 ]]; then
                echo "Debug slot is free."
                break
            fi
            echo "Debug slot busy (active debug jobs: $ACTIVE). Retrying in 30s..."
            sleep 30
        done
    else
        ACTIVE=$(_debug_jobs_active)
        if [[ "$ACTIVE" -gt 0 ]]; then
            echo "Debug slot currently busy (active debug jobs: $ACTIVE)."
            echo "Use --wait-slot to auto-submit when free."
            exit 2
        fi
    fi
fi

CMD=(
    sbatch
    --account="$PROJECT"
    --partition="$PARTITION"
    --qos="$QOS"
    --job-name="$TARGET"
    --nodes="$N_NODES"
    --time="$WALL_TIME"
    --output="$SLURM_OUT_DIR/${TARGET}-%j.out"
    --error="$SLURM_OUT_DIR/${TARGET}-%j.out"
    --export="ALL,FT_ARCH=$FT_ARCH,N_PER_RUN=$N_PER_RUN,SAMPLE_SIZES=$SAMPLE_SIZES"
    "$SCRIPT_PATH"
)

echo "========================================================"
echo " Submitting one debug FT bundle"
echo "  Target    : $TARGET"
echo "  Account   : $PROJECT"
echo "  Partition : $PARTITION"
echo "  QOS       : $QOS"
echo "  Nodes     : $N_NODES"
echo "  Wall time : $WALL_TIME"
echo "  Script    : $SCRIPT_PATH"
echo "========================================================"

if $DRY_RUN; then
    echo -n "[dry-run] "
    printf '%q ' "${CMD[@]}"
    echo
else
    "${CMD[@]}"
fi
