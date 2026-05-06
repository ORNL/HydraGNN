#!/bin/bash
# =============================================================================
#  submit_ft1_ft3_jobs.sh  —  Frontier Slurm orchestration for FT1 + FT3
#
#  Submits all experiments as Slurm jobs with proper dependencies so that:
#
#    Stage 0: Preprocess FT3 data (download + serialise, two archs in parallel)
#    Stage 0: Generate FT1 feasibility data (after FT3 SAGE data is ready)
#    Stage 1: Run 16 training jobs (8 FT1 + 8 FT3) in parallel after data is ready
#    Stage 2: Collect results & generate plots after all 16 jobs complete
#
#  Total submitted: 3 (data) + 16 (training) + 1 (collect+plot) = 20 jobs
#
#  Usage (from examples/opf/finetune/):
#    bash submit_ft1_ft3_jobs.sh [--dry-run]
#
#  Flags:
#    --dry-run   Print all sbatch commands but do not submit them.
#
#  Requirements:
#    - HydraGNN pretrained models in examples/opf/pretrained_models/
#    - HYDRAGNN_ROOT, PROJECT, VENV variables set below or in environment
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit for your allocation
# ─────────────────────────────────────────────────────────────────────────────
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
LOG_DIR="$HYDRAGNN_ROOT/logs"
PROJECT=${PROJECT:-LRN078}
PARTITION=${PARTITION:-batch}
N_TRAIN_NODES=${N_TRAIN_NODES:-8}   # nodes per training job
N_DATA_NODES=${N_DATA_NODES:-1}     # nodes for preprocessing
TRAIN_TIME=${TRAIN_TIME:-04:00:00}  # wall time per training job
DATA_TIME=${DATA_TIME:-02:00:00}    # wall time for data jobs
SLURM_OUT_DIR=${SLURM_OUT_DIR:-/lustre/orion/lrn078/proj-shared/HydraGNN}

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ─────────────────────────────────────────────────────────────────────────────
# Submission helper
# ─────────────────────────────────────────────────────────────────────────────
declare -a ALL_TRAINING_JOB_IDS=()

_sbatch() {
    local label="$1"
    shift
    if $DRY_RUN; then
        echo "[dry-run] sbatch $*"
        echo "DRY_$label"
    else
        local id
        id=$(sbatch "$@" | awk '{print $NF}')
        echo "[submit] $label → job $id"
        echo "$id"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0a — Preprocess FT3 data (HeteroSAGE and HeteroHEAT in parallel)
#             Phase 1 of train_opf_solution_heterogeneous.py --preonly
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Stage 0a: FT3 data preprocessing"
echo "========================================================"

JOB_FT3_DATA_SAGE=$(_sbatch "FT3_data_HeteroSAGE" \
    --account=$PROJECT --partition=$PARTITION \
    --job-name=FT3-data-SAGE \
    --nodes=$N_DATA_NODES --time=$DATA_TIME \
    --output="$SLURM_OUT_DIR/FT3-data-SAGE-%j.out" \
    --error="$SLURM_OUT_DIR/FT3-data-SAGE-%j.out" \
    --export=ALL,FT_STRATEGY=FT3_contingency,FT_ARCH=HeteroSAGE,FT_REGIME=full,PHASES=preonly \
    "$FT_DIR/FT3_contingency/job-frontier-HeteroSAGE.sh")

# FT3 HeteroHEAT reuses the same dataset as HeteroSAGE — no separate data job needed.
JOB_FT3_DATA_HEAT=$JOB_FT3_DATA_SAGE

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0b — Generate FT1 feasibility dataset
#             Depends on FT3_SAGE data (used as the source of feasible samples)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Stage 0b: FT1 feasibility data generation"
echo "========================================================"

# Inline job script for generate_infeasible_samples.py
read -r -d '' FT1_DATA_SCRIPT <<'SCRIPT' || true
#!/bin/bash
#SBATCH -A __PROJECT__
#SBATCH -J FT1-data
#SBATCH -o __SLURM_OUT__/FT1-data-%j.out
#SBATCH -e __SLURM_OUT__/FT1-data-%j.out
#SBATCH -t __DATA_TIME__
#SBATCH -p __PARTITION__
#SBATCH -N 1

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate __VENV__

HYDRAGNN_ROOT=__HYDRAGNN_ROOT__
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
DATASET_DIR="$HYDRAGNN_ROOT/examples/opf/dataset"

export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH

SRC_DIR="$DATASET_DIR/FT3_contingency_data"
OUT_DIR="$DATASET_DIR/FT1_feasibility_data"

echo "[FT1 data] Generating infeasible samples..."
echo "  src  : $SRC_DIR"
echo "  out  : $OUT_DIR"

python -u "$FT_DIR/generate_infeasible_samples.py" \
    --src_dir "$SRC_DIR" \
    --out_dir "$OUT_DIR" \
    --overload_factor 6.0 \
    --max_samples 5000 \
    --seed 42

echo "[FT1 data] Done."
SCRIPT

# Materialise the inline script to a temp file
FT1_DATA_SCRIPT_PATH="$FT_DIR/_tmp_ft1_data_job.sh"
echo "$FT1_DATA_SCRIPT" \
    | sed "s|__PROJECT__|$PROJECT|g" \
    | sed "s|__SLURM_OUT__|$SLURM_OUT_DIR|g" \
    | sed "s|__DATA_TIME__|$DATA_TIME|g" \
    | sed "s|__PARTITION__|$PARTITION|g" \
    | sed "s|__HYDRAGNN_ROOT__|$HYDRAGNN_ROOT|g" \
    | sed "s|__VENV__|/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv|g" \
    > "$FT1_DATA_SCRIPT_PATH"
chmod +x "$FT1_DATA_SCRIPT_PATH"

JOB_FT1_DATA=$(_sbatch "FT1_data" \
    --dependency="afterok:$JOB_FT3_DATA_SAGE" \
    "$FT1_DATA_SCRIPT_PATH")

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Training jobs
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Stage 1: Training jobs (FT1 + FT3, all regimes)"
echo "========================================================"

# Helper: submit a single FT1 training job
_submit_ft1() {
    local arch="$1"
    local regime="$2"
    local no_pretrained="${3:-false}"
    local label="FT1_${arch}_${regime}"
    local dep="afterok:$JOB_FT1_DATA"
    local extra_flags=""

    if [[ "$no_pretrained" == "true" ]]; then
        label="FT1_${arch}_baseline"
        extra_flags="--export=ALL,FT_REGIME=full,NO_PRETRAINED=1"
    else
        extra_flags="--export=ALL,FT_REGIME=$regime"
    fi

    local job_script="$FT_DIR/FT1_feasibility_classification/job-frontier-FT1-${arch}.sh"

    local id
    id=$(_sbatch "$label" \
        --account=$PROJECT --partition=$PARTITION \
        --job-name="$label" \
        --nodes=$N_TRAIN_NODES --time=$TRAIN_TIME \
        --output="$SLURM_OUT_DIR/${label}-%j.out" \
        --error="$SLURM_OUT_DIR/${label}-%j.out" \
        --dependency="$dep" \
        $extra_flags \
        "$job_script")
    ALL_TRAINING_JOB_IDS+=("$id")
}

# Helper: submit a single FT3 training job
_submit_ft3() {
    local arch="$1"
    local regime="$2"
    local no_pretrained="${3:-false}"
    local label="FT3_${arch}_${regime}"
    local dep="afterok:$JOB_FT3_DATA_SAGE"
    local export_str="FT_STRATEGY=FT3_contingency,FT_ARCH=$arch,FT_REGIME=$regime,PHASES=train"

    if [[ "$no_pretrained" == "true" ]]; then
        label="FT3_${arch}_baseline"
        export_str="${export_str},NO_PRETRAINED=1"
    fi

    local job_script
    if [[ "$arch" == "HeteroSAGE" ]]; then
        job_script="$FT_DIR/FT3_contingency/job-frontier-HeteroSAGE.sh"
    else
        job_script="$FT_DIR/FT3_contingency/job-frontier-HeteroHEAT.sh"
    fi

    local id
    id=$(_sbatch "$label" \
        --account=$PROJECT --partition=$PARTITION \
        --job-name="$label" \
        --nodes=$N_TRAIN_NODES --time=$TRAIN_TIME \
        --output="$SLURM_OUT_DIR/${label}-%j.out" \
        --error="$SLURM_OUT_DIR/${label}-%j.out" \
        --dependency="$dep" \
        --export="ALL,$export_str" \
        "$job_script")
    ALL_TRAINING_JOB_IDS+=("$id")
}

# ── FT1 × {HeteroSAGE, HeteroHEAT} × {head_only, partial, full, baseline} ──
for ARCH in HeteroSAGE HeteroHEAT; do
    for REGIME in head_only partial full; do
        _submit_ft1 "$ARCH" "$REGIME" false
    done
    _submit_ft1 "$ARCH" "full" true    # baseline (random init)
done

# ── FT3 × {HeteroSAGE, HeteroHEAT} × {head_only, partial, full, baseline} ──
for ARCH in HeteroSAGE HeteroHEAT; do
    for REGIME in head_only partial full; do
        _submit_ft3 "$ARCH" "$REGIME" false
    done
    _submit_ft3 "$ARCH" "full" true    # baseline (random init)
done

echo ""
echo "Submitted ${#ALL_TRAINING_JOB_IDS[@]} training jobs:"
printf '  %s\n' "${ALL_TRAINING_JOB_IDS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Collect results and generate plots
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Stage 2: collect_results.py + plot_ft_results.py"
echo "========================================================"

# Build afterok dependency on ALL training jobs
DEP_ALL_TRAIN=$(IFS=:; echo "afterok:${ALL_TRAINING_JOB_IDS[*]}")

read -r -d '' COLLECT_SCRIPT <<'SCRIPT' || true
#!/bin/bash
#SBATCH -A __PROJECT__
#SBATCH -J FT-collect
#SBATCH -o __SLURM_OUT__/FT-collect-%j.out
#SBATCH -e __SLURM_OUT__/FT-collect-%j.out
#SBATCH -t 00:30:00
#SBATCH -p __PARTITION__
#SBATCH -N 1

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate __VENV__

HYDRAGNN_ROOT=__HYDRAGNN_ROOT__
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
LOGS_ROOT="$HYDRAGNN_ROOT/logs"
OUT_DIR="$FT_DIR/results"

export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH

echo "[Stage 2] Collecting results..."
python -u "$FT_DIR/collect_results.py" \
    --logs_root "$LOGS_ROOT" \
    --out_dir   "$OUT_DIR"

echo "[Stage 2] Generating plots..."
python -u "$FT_DIR/plot_ft_results.py" \
    --summary "$OUT_DIR/ft1_ft3_summary.json" \
    --out_dir "$OUT_DIR/figures"

echo "[Stage 2] Done.  Figures in: $OUT_DIR/figures/"
SCRIPT

COLLECT_SCRIPT_PATH="$FT_DIR/_tmp_collect_job.sh"
echo "$COLLECT_SCRIPT" \
    | sed "s|__PROJECT__|$PROJECT|g" \
    | sed "s|__SLURM_OUT__|$SLURM_OUT_DIR|g" \
    | sed "s|__PARTITION__|$PARTITION|g" \
    | sed "s|__HYDRAGNN_ROOT__|$HYDRAGNN_ROOT|g" \
    | sed "s|__VENV__|/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv|g" \
    > "$COLLECT_SCRIPT_PATH"
chmod +x "$COLLECT_SCRIPT_PATH"

JOB_COLLECT=$(_sbatch "collect_plot" \
    --dependency="$DEP_ALL_TRAIN" \
    "$COLLECT_SCRIPT_PATH")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Submission complete"
echo "========================================================"
echo "  FT3 data prep  : $JOB_FT3_DATA_SAGE (SAGE), $JOB_FT3_DATA_HEAT (HEAT)"
echo "  FT1 data gen   : $JOB_FT1_DATA"
echo "  Training jobs  : ${ALL_TRAINING_JOB_IDS[*]}"
echo "  Collect+plot   : $JOB_COLLECT"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $SLURM_OUT_DIR/FT-collect-<JOB_ID>.out"
echo ""
echo "After completion, results are in:"
echo "  $HYDRAGNN_ROOT/examples/opf/finetune/results/"

# Cleanup temp scripts (only if we actually submitted)
if ! $DRY_RUN; then
    rm -f "$FT1_DATA_SCRIPT_PATH" "$COLLECT_SCRIPT_PATH"
fi
