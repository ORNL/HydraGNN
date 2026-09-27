#!/bin/bash
# =============================================================================
#  preprocess_ft1_ft3_loginnode.sh  —  CPU-only data preprocessing for FT1+FT3
#
#  Runs all preprocessing steps on the login node (no GPU, no srun):
#    Step 1: Serialize FT3 contingency data (shared by HeteroSAGE and HeteroHEAT)
#    Step 2: Generate FT1 feasibility dataset from FT3 data
#
#  Usage (from examples/opf/finetune/):
#    bash preprocess_ft1_ft3_loginnode.sh
# =============================================================================
set -euo pipefail

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn078/proj-shared/HydraGNN}
FT_DIR="$HYDRAGNN_ROOT/examples/opf/finetune"
DATASET_DIR="$HYDRAGNN_ROOT/examples/opf/dataset"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate /lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

export PYTHONPATH=$HYDRAGNN_ROOT:${PYTHONPATH:-}
export PYTHONPATH=/lustre/orion/lrn078/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:${PYTHONPATH:-}

cd "$FT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: FT3 contingency data — HeteroSAGE
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Step 1: FT3 contingency preprocessing (HeteroSAGE)"
echo "========================================================"
python -u "$HYDRAGNN_ROOT/examples/opf/train_opf_solution_heterogeneous.py" \
    --inputfile "$FT_DIR/FT3_contingency/config_HeteroSAGE_full.json" \
    --hdf5 \
    --preonly \
    --case_name pglib_opf_case118_ieee \
    --num_groups 20 \
    --modelname FT3_contingency_data \
    --data_root "$DATASET_DIR" \
    --topological_perturbations
echo "Step 1 done."

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: FT1 feasibility dataset (from FT3 SAGE data)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Step 2: FT1 feasibility data generation"
echo "========================================================"
python -u "$FT_DIR/generate_infeasible_samples.py" \
    --src_dir "$DATASET_DIR/FT3_contingency_data" \
    --out_dir "$DATASET_DIR/FT1_feasibility_data" \
    --overload_factor 6.0 \
    --max_samples 5000 \
    --seed 42
echo "Step 2 done."

echo ""
echo "========================================================"
echo " All preprocessing complete."
echo " FT3 SAGE : $DATASET_DIR/FT3_contingency_data"
echo " FT3 HEAT : (reuses FT3_contingency_data)"
echo " FT1 data : $DATASET_DIR/FT1_feasibility_data"
echo "========================================================"
