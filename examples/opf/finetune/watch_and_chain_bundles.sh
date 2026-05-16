#!/bin/bash
# =============================================================================
#  Bundled per-(arch, method, bundle) submission for OPF FT1 + FT3.
#
#  Each Slurm job runs ALL sample sizes in a "bundle" inside one allocation.
#  Bundles are sized so that the total expected wall-time fits the selected
#  QOS walltime cap (rough estimate: ~0.5 s per train sample).
#
#  Per-cell isolation is preserved: the underlying job scripts already write
#  each (arch, method, N) into its own logs/{LOG_NAME}/ directory, so cells
#  cannot overwrite each other.
#
#  If a bundle is killed mid-way, cells whose results.json already wrote will
#  be skipped on the next run; the existing per-N watcher (or this same
#  bundle watcher) will pick up the unfinished cells automatically.
# =============================================================================
set -u

HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN
FT_DIR=$HYDRAGNN_ROOT/examples/opf/finetune

ACCOUNT=eng164
QOS=debug
PARTITION=batch
TIME=02:00:00
NODES=1

ARCHES=(HeteroSAGE HeteroHEAT)
METHODS=(full partial head_only scratch)

# Bundles, ordered cheapest → most expensive within each group.
# A: 5 cheap N's,  ~120 min worst case
# B: two medium N's,  ~145 min
# C: 50000 alone,  ~90 min
# D: 100000 alone, ~150 min
# E: 270000 alone (FT3 only)
FT1_BUNDLES=(
    "100 500 1000 2500 5000"
    "10000 25000"
    "50000"
    "100000"
)
FT3_BUNDLES=(
    "100 500 1000 2500 5000"
    "10000 25000"
    "50000"
    "100000"
    "270000"
)

LOG=$FT_DIR/ft_bundles_chain.log
: > "$LOG"

# Compute LOG_NAME for one (task, arch, method, n) — must match the rules
# baked into the underlying job scripts.
log_name_for() {
    local task=$1 arch=$2 method=$3 n=$4
    local scratch_tag="" regime="$method"
    if [[ "$method" == "scratch" ]]; then
        scratch_tag="_scratch"
        regime="full"
    fi
    if [[ "$task" == "FT1" ]]; then
        echo "FT1_feasibility_${arch}_${regime}${scratch_tag}_n${n}"
    else
        echo "finetune_FT3_contingency_${arch}_${regime}${scratch_tag}_n${n}"
    fi
}

# Filter out N's that already have a results.json on disk.
remaining_ns() {
    local task=$1 arch=$2 method=$3
    shift 3
    local out="" n logname
    for n in "$@"; do
        logname=$(log_name_for "$task" "$arch" "$method" "$n")
        if [[ ! -f "$FT_DIR/logs/$logname/results.json" ]]; then
            out+="$n "
        fi
    done
    echo "$out" | sed 's/[[:space:]]*$//'
}

submit_bundle() {
    local task=$1 arch=$2 method=$3
    shift 3
    local ns=("$@")
    local sizes_str="${ns[*]}"

    local script jobname
    case "$task" in
        FT1) script=$FT_DIR/job-frontier-FT1-single-method.sh ;;
        FT3) script=$FT_DIR/job-frontier-FT3-single-method.sh ;;
        *) echo "Unknown task $task" >> "$LOG"; return 1 ;;
    esac

    local first="${ns[0]}" last="${ns[-1]}"
    jobname="${task}-${arch}-${method}-bundle-${first}-${last}"
    local out=$HYDRAGNN_ROOT/${jobname}-%j.out

    # Wait for an available slot in the selected QOS.
    while squeue -u "$USER" -h --qos=$QOS -o "%i" 2>/dev/null | grep -q .; do
        echo "[$(date +%H:%M:%S)] $QOS QOS busy — waiting 30s before $jobname" >> "$LOG"
        sleep 30
    done

    local jid
    jid=$(sbatch --parsable \
        --account=$ACCOUNT --partition=$PARTITION --qos=$QOS \
        --job-name="$jobname" \
        --nodes=$NODES --time=$TIME \
        --output="$out" --error="$out" \
        --export=ALL,FT_ARCH=$arch,FT_METHOD=$method,SAMPLE_SIZES="$sizes_str",N_PER_RUN=1 \
        "$script" 2>>"$LOG")
    if [[ -z "$jid" ]]; then
        echo "[$(date +%H:%M:%S)] [submit-FAILED] $jobname  N=[$sizes_str]" >> "$LOG"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] [submit] $jobname → job $jid  N=[$sizes_str]" >> "$LOG"

    # Poll until the job leaves the queue.
    while squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -q .; do
        sleep 30
    done
    echo "[$(date +%H:%M:%S)]   $jobname (job $jid) finished." >> "$LOG"
}

run_task_bundles() {
    local task=$1 arch=$2 method=$3
    shift 3
    local bundles=("$@")
    local b ns
    for b in "${bundles[@]}"; do
        # Filter already-done cells.
        ns=$(remaining_ns "$task" "$arch" "$method" $b)
        if [[ -z "$ns" ]]; then
            echo "[$(date +%H:%M:%S)] [skip-bundle] $task $arch $method [$b] (all cells done)" >> "$LOG"
            continue
        fi
        # shellcheck disable=SC2086
        submit_bundle "$task" "$arch" "$method" $ns
    done
}

for ARCH in "${ARCHES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        run_task_bundles FT1 "$ARCH" "$METHOD" "${FT1_BUNDLES[@]}"
        run_task_bundles FT3 "$ARCH" "$METHOD" "${FT3_BUNDLES[@]}"
    done
done

echo "[$(date +%H:%M:%S)] All bundles submitted and finished." >> "$LOG"
