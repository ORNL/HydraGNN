#!/bin/bash
# Submit all 16 single-method FT jobs to debug QOS, chained via --dependency=afterany
# so only one runs at a time.
set -euo pipefail

FT_DIR="$(cd "$(dirname "$0")" && pwd)"
HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

PREV_JOB=""

submit_one() {
    local TASK="$1" ARCH="$2" METHOD="$3"
    local LABEL="${TASK}-${ARCH}-${METHOD}"
    local SCRIPT="${FT_DIR}/job-frontier-${TASK}-single-method.sh"
    local DEP_OPT=""
    [[ -n "$PREV_JOB" ]] && DEP_OPT="--dependency=afterany:${PREV_JOB}"

    if $DRY_RUN; then
        echo "[dry-run] $LABEL  dep=${PREV_JOB:-none}"
        PREV_JOB="DRYRUN_${LABEL}"
        return
    fi

    local ID
    ID=$(sbatch \
        --account=eng164 --partition=batch --qos=debug \
        --job-name="$LABEL" \
        --nodes=8 --time=02:00:00 \
        --output="${HYDRAGNN_ROOT}/${LABEL}-%j.out" \
        --error="${HYDRAGNN_ROOT}/${LABEL}-%j.out" \
        --export=ALL,FT_ARCH=${ARCH},FT_METHOD=${METHOD},N_PER_RUN=8 \
        ${DEP_OPT} \
        "$SCRIPT" | awk '{print $NF}')
    echo "[submit] $LABEL → job $ID ${DEP_OPT}"
    PREV_JOB="$ID"
}

for ARCH in HeteroSAGE HeteroHEAT; do
    for METHOD in full partial head_only scratch; do
        submit_one FT1 "$ARCH" "$METHOD"
        submit_one FT3 "$ARCH" "$METHOD"
    done
done

echo ""
echo "All 16 jobs submitted in debug QOS chain."
