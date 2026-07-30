export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

function cmd() {
    echo "$@"
    time $@
}

# ROCm 7.13 variant of export_variables_and_run_interactive_job.sh.
# Uses the AMD ROCm 7.13 PyTorch wheel (bundled RCCL 2.28.3, which fixes the
# HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION MI250X collective crash seen on the old
# ROCm 7.1.x / 6.4 RCCL) plus the validated lumina-sdk RCCL runtime config.
HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN}"
VENV_PATH="${VENV_PATH:-/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm713/hydragnn_venv_rocm713}"

# Load ROCm 7.13 module stack + conda environment
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm713.sh
source activate "${VENV_PATH}"

# export python path to HydraGNN
export PYTHONPATH=$PWD:$PYTHONPATH

# export python path to use ADIOS2 v.2.10.2 (installed into the venv site-packages)
export PYTHONPATH=${VENV_PATH}/lib/python3.11/site-packages/:$PYTHONPATH

which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

module unload darshan-runtime
module list

echo $LD_LIBRARY_PATH | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

# ---------------------------------------------------------------------------
# ROCm / RCCL runtime config (validated on the ROCm 7.13 stack in lumina-sdk)
# ---------------------------------------------------------------------------
export ROCM_HOME=${ROCM_PATH}
export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID:-$$}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Disable the aws-ofi-nccl/rccl plugin. On the ROCm 7.13 stack it fails CXI
# domain creation (RC -38 ENOSYS); RCCL then falls back to its built-in
# transports (intra-node xGMI/SHM, inter-node TCP over the HSN NICs).
# NOTE: this intentionally drops the old ROCm 6.x/7.1 workarounds that were
# needed with the aws-ofi plugin — NCCL_P2P_DISABLE, NCCL_P2P_LEVEL,
# NCCL_PROTO=Simple, and the LD_LIBRARY_PATH injection of the ROCm 6.3.1 plugin.
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_PLUGIN=none
export NCCL_CROSS_NIC=1
export TORCH_NCCL_HIGH_PRIORITY=1

# Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA
env | grep ^NCCL

export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=1
[ -z $BATCH_SIZE ] && BATCH_SIZE=20 ## 320 (Perlmutter) 160 (Frontier)
export BATCH_SIZE=$BATCH_SIZE
export NUM_EPOCH=4

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2

# Dataset ordering matches gfm_deephyper_multi_all_mpnn.py multi_model_list
export datadir0=Alexandria
export datadir1=ANI1x
export datadir2=MPTrj
export datadir3=OC2020
export datadir4=OC2022
export datadir5=ODAC23
export datadir6=OMat24
export datadir7=OMol25
export datadir8=OC25
export datadir9=OPoly2026
export datadir10=Nabla2DFT
export datadir11=QCML
export datadir12=QM7-X
export datadir13=transition1x
export datadir14=OMol25-non-neutral

# (A) Setup omnistat sampling environment
ml use /sw/frontier/amdsw/modulefiles/
ml omnistat-wrapper
export OMNISTAT_CONFIG=$HYDRAGNN_ROOT/omnistat.hydragnn-external-fp64.config

# (B) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 1

## HYDRAGNN_USE_FSDP: 1 (enabled), 0 (disabled)
export HYDRAGNN_USE_FSDP=1
## HYDRAGNN_FSDP_VERSION: 1 (FSDP1), 2 (FSDP2/composable)
export HYDRAGNN_FSDP_VERSION=1
export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD

[ -z $HIDDEN_DIM ] && HIDDEN_DIM=30

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-FSDP$HYDRAGNN_USE_FSDP --everyone \
    --inputfile=gfm_mlip.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH)) \
    --multi --ddstore --multi_model_list=$datadir2,$datadir3,$datadir4,$datadir5 --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=fp64 \
    --mpnn_type=SchNet \
    --hidden_dim=$HIDDEN_DIM \
    --num_conv_layers=4 \
    --num_headlayers=2 \
    --dim_headlayers=10
