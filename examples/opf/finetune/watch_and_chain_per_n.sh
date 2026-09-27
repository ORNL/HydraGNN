#!/bin/bash
# =============================================================================
#  Per-(arch, method, N) chained submission for OPF FT1 + FT3.
#  Submits ONE 1-node job at a time, polls for completion, then submits
#  the next.  Each job runs a single sample size for a single (arch, method)
#  combination so that it fits comfortably inside the selected QOS walltime.
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
FT1_NS=(100 500 1000 2500 5000 10000 25000 50000 100000)
FT3_NS=(100 500 1000 2500 5000 10000 25000 50000 100000 270000)

LOG=/lustre/orion/lrn078/proj-shared/HydraGNN/examples/opf/finetune/ft_per_n_chain.log
: > "$LOG"

submit_one() {
    local task=$1 arch=$2 method=$3 n=$4
    local script jobname
    case "$task" in
        FT1) script=$FT_DIR/job-frontier-FT1-single-method.sh ;;
        FT3) script=$FT_DIR/job-frontier-FT3-single-method.sh ;;
        *) echo "Unknown task $task" >> "$LOG"; return 1 ;;
    esac
    jobname="${task}-${arch}-${method}-n${n}"
    local out=$HYDRAGNN_ROOT/${jobname}-%j.out

    # Skip if a results.json already exists for this exact (task, arch, method, n).
    local scratch_tag=""
    [[ "$method" == "scratch" ]] && scratch_tag="_scratch"
    local regime="$method"
    [[ "$method" == "scratch" ]] && regime="full"
    local logname
    if [[ "$task" == "FT1" ]]; then
        logname="FT1_feasibility_${arch}_${regime}${scratch_tag}_n${n}"
    else
        logname="finetune_FT3_contingency_${arch}_${regime}${scratch_tag}_n${n}"
    fi
    if [[ -f "$FT_DIR/logs/$logname/results.json" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] $jobname (results.json present)" >> "$LOG"
        return 0
    fi

    # Wait for selected QOS slot (only 1 submitted job allowed).
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
        --export=ALL,FT_ARCH=$arch,FT_METHOD=$method,SAMPLE_SIZES=$n,N_PER_RUN=1 \
        "$script" 2>>"$LOG")
    if [[ -z "$jid" ]]; then
        echo "[$(date +%H:%M:%S)] [submit-FAILED] $jobname" >> "$LOG"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] [submit] $jobname → job $jid" >> "$LOG"

    while squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -q .; do
        sleep 30
    done
    echo "[$(date +%H:%M:%S)]   $jobname (job $jid) finished." >> "$LOG"
}

for ARCH in "${ARCHES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for N in "${FT1_NS[@]}"; do
            submit_one FT1 "$ARCH" "$METHOD" "$N"
        done
        for N in "${FT3_NS[@]}"; do
            submit_one FT3 "$ARCH" "$METHOD" "$N"
        done
    done
done

echo "[$(date +%H:%M:%S)] All jobs submitted and finished." >> "$LOG"
