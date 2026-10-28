#!/bin/bash
#SBATCH -A LRN078
#SBATCH -J OPF-HEAT-AB
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/job-opf-heat-ab-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/job-opf-heat-ab-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 16

set -euo pipefail

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN
INPUTFILE=${INPUTFILE:-logs/opf_hpo_4249563_0.28/config.json}
MODELNAME=${MODELNAME:-OPF_Solution_Hetero}
BASELINE_LOG_NAME=${BASELINE_LOG_NAME:-opf_heat_hpo_best_baseline}
PHYSICS_LOG_NAME=${PHYSICS_LOG_NAME:-opf_heat_hpo_best_physics}

# Domain-loss weights for the physics-informed run.
# Use two separate index variables to avoid fragile string-splitting.
DOMAIN_SMOOTHNESS_WEIGHT=${DOMAIN_SMOOTHNESS_WEIGHT:-0.001}
DOMAIN_TRANSFORMER_SMOOTHNESS_WEIGHT=${DOMAIN_TRANSFORMER_SMOOTHNESS_WEIGHT:-0.001}
DOMAIN_VOLTAGE_BOUND_WEIGHT=${DOMAIN_VOLTAGE_BOUND_WEIGHT:-0.01}
VMIN_IDX=${VMIN_IDX:-2}
VMAX_IDX=${VMAX_IDX:-3}
# voltage_output_index=1: Vm is at index 1 in bus targets [Va(0), Vm(1)]
DOMAIN_VOLTAGE_OUTPUT_INDEX=${DOMAIN_VOLTAGE_OUTPUT_INDEX:-1}
# va_output_index=0: Va is at index 0 in bus targets
DOMAIN_VA_OUTPUT_INDEX=${DOMAIN_VA_OUTPUT_INDEX:-0}
# angle difference limit and DC thermal limit penalties (new deeper physics terms)
DOMAIN_ANGLE_DIFF_WEIGHT=${DOMAIN_ANGLE_DIFF_WEIGHT:-0.001}
DOMAIN_LINE_FLOW_WEIGHT=${DOMAIN_LINE_FLOW_WEIGHT:-0.001}

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate /lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

export PYTHONPATH=$HYDRAGNN_ROOT:${PYTHONPATH:-}
export PYTHONPATH=/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH

module unload darshan-runtime

export PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PLUGIN_PATH}/lib
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_NET_PLUGIN=${PLUGIN_PATH}/lib/librccl-net.so
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0

cd $HYDRAGNN_ROOT/examples/opf

which python
python -c "import torch; print(torch.__version__, torch.__file__)"
python -c "import numpy; print(numpy.__version__)"

run_train() {
    local log_name=$1
    shift
    echo ""
    echo "====================================================================="
    echo "  STARTING: INPUTFILE=$INPUTFILE  MODELNAME=$MODELNAME  LOG=$log_name"
    echo "  $(date)"
    echo "====================================================================="
    srun --export=ALL,HYDRAGNN_DIAG=1,HYDRAGNN_DIAG_RANK=0 \
        -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 \
        --gpus-per-task=1 --gpu-bind=closest \
        python -u train_opf_solution_heterogeneous.py \
        --hdf5 \
        --inputfile "$INPUTFILE" \
        --modelname "$MODELNAME" \
        --log "$log_name" \
        "$@"
    echo ""
    echo "====================================================================="
    echo "  COMPLETED: $log_name  $(date)"
    echo "====================================================================="
    # Sanity-check: verify the run produced a log file.
    local runlog="logs/$log_name/run.log"
    if [[ -f "$runlog" ]]; then
        echo "  Log found: $runlog"
    else
        echo "  WARNING: expected log not found at $runlog"
    fi
}

# ── Run 1: Baseline (standard loss) ──────────────────────────────────────────
run_train "$BASELINE_LOG_NAME" --disable_domain_loss

# ── Run 2: Physics-informed (domain loss enabled) ────────────────────────────
run_train "$PHYSICS_LOG_NAME" \
    --enable_domain_loss \
    --domain_loss_smoothness_weight "$DOMAIN_SMOOTHNESS_WEIGHT" \
    --domain_loss_transformer_smoothness_weight "$DOMAIN_TRANSFORMER_SMOOTHNESS_WEIGHT" \
    --domain_loss_voltage_bound_weight "$DOMAIN_VOLTAGE_BOUND_WEIGHT" \
    --domain_loss_voltage_bound_feature_indices "$VMIN_IDX" "$VMAX_IDX" \
    --domain_loss_voltage_output_index "$DOMAIN_VOLTAGE_OUTPUT_INDEX" \
    --domain_loss_va_output_index "$DOMAIN_VA_OUTPUT_INDEX" \
    --domain_loss_angle_diff_weight "$DOMAIN_ANGLE_DIFF_WEIGHT" \
    --domain_loss_line_flow_weight "$DOMAIN_LINE_FLOW_WEIGHT"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "====================================================================="
echo "  BOTH RUNS COMPLETE  $(date)"
echo "  Baseline log : logs/$BASELINE_LOG_NAME/run.log"
echo "  Physics log  : logs/$PHYSICS_LOG_NAME/run.log"
echo ""
echo "  To compare results run:"
echo "    python3 compare_heat_runs.py \\"
echo "        logs/$BASELINE_LOG_NAME/run.log \\"
echo "        logs/$PHYSICS_LOG_NAME/run.log"
echo "  For JSON output add --json"
echo "====================================================================="
