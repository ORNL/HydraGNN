#!/bin/bash
# =============================================================================
#  watch_and_chain_debug.sh
#
#  Submits 16 FT single-method jobs to the debug QOS one at a time.
#  Waits for each job to leave the queue (RUNNING→finished) before
#  submitting the next one. Polls every 30 seconds.
#
#  Usage:
#    nohup bash watch_and_chain_debug.sh > /tmp/ft_debug_chain.log 2>&1 &
#    tail -f /tmp/ft_debug_chain.log
# =============================================================================
set -uo pipefail

FT_DIR="$(cd "$(dirname "$0")" && pwd)"
HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN
POLL_INTERVAL=30

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_job() {
    local JID="$1"
    while squeue -j "$JID" -h 2>/dev/null | grep -q "$JID"; do
        log "  Job $JID still in queue — waiting ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    done
    log "  Job $JID finished."
}

submit_debug() {
    local TASK="$1" ARCH="$2" METHOD="$3"
    local LABEL="${TASK}-${ARCH}-${METHOD}"
    local SCRIPT="${FT_DIR}/job-frontier-${TASK}-single-method.sh"

    # Wait until debug QOS is free (no jobs of ours there)
    while squeue -u "$USER" --qos=debug -h 2>/dev/null | grep -q "$USER"; do
        log "  Debug QOS busy — waiting ${POLL_INTERVAL}s before submitting $LABEL ..."
        sleep $POLL_INTERVAL
    done

    local ID
    ID=$(sbatch \
        --account=eng164 --partition=batch --qos=debug \
        --job-name="$LABEL" \
        --nodes=1 --time=02:00:00 \
        --output="${HYDRAGNN_ROOT}/${LABEL}-%j.out" \
        --error="${HYDRAGNN_ROOT}/${LABEL}-%j.out" \
        --export=ALL,FT_ARCH=${ARCH},FT_METHOD=${METHOD},N_PER_RUN=1 \
        "$SCRIPT" | awk '{print $NF}')
    log "[submit] $LABEL → job $ID"
    wait_for_job "$ID"
}

log "=== FT debug chain starting — 16 jobs ==="

for ARCH in HeteroSAGE HeteroHEAT; do
    for METHOD in full partial head_only scratch; do
        submit_debug FT1 "$ARCH" "$METHOD"
        submit_debug FT3 "$ARCH" "$METHOD"
    done
done

log "=== All 16 FT jobs done! ==="
