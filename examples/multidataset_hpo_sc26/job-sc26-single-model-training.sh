#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 16
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -C nvme

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

function cmd() {
    echo "$@"
    time $@
}

function setup_bb()
{
    # Move a copy of the env to the NVMe on each node
    if [ -d /mnt/bb/${USER} ]; then
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
	    SRCDIR=/lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN
        for FILENAME in HydraGNN-Installation-Frontier.tar; do
            echo "Checking ${FILENAME}"
            if [ ! -f /mnt/bb/${USER}/${FILENAME} ]; then
                echo "Copying ${FILENAME} to each local NVME"
                time sbcast -pfv ${SRCDIR}/${FILENAME} /mnt/bb/${USER}/${FILENAME}
                if [ ! "$?" == "0" ]; then
                    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
                    # your application may pick up partially complete shared library files, which would give you confusing errors.
                    echo "SBCAST failed!"
                    break
                fi
            fi
            echo "Untar ${FILENAME}"
            time srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u tar -xf /mnt/bb/${USER}/${FILENAME} -C  /mnt/bb/${USER}/
        done
        echo "NVME is ready to use"
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
    fi
}

HYDRAGNN_ROOT=/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2026/HydraGNN
# HYDRAGNN_ROOT=/lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN

# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm640.sh

# source activate /lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN/HydraGNN-Installation-Frontier/hydragnn_venv
source activate /lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2026/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm6.4/hydragnn_venv


# setup_bb
# if [ -d /mnt/bb/${USER}/HydraGNN-Installation-Frontier ]; then    
#     export PYTHONPATH=/mnt/bb/${USER}/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH
# fi

#export python path to HydragNN
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo "===== LD_LIBRARY_PATH ====="
echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

export HYDRAGNN_TRACE_LEVEL=0
export HYDRAGNN_MAX_NUM_BATCH=1000
export TASK_PARALLEL=0
export HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT=0
export BATCH_SIZE=40
export NUM_EPOCH=50

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=1

# Dataset ordering matches gfm_deephyper_multi_all_mpnn.py multi_model_list
export datadir0=Alexandria
export datadir1=ANI1x
export datadir2=MPTrj
export datadir3=OC2020
export datadir4=OC2022
export datadir5=OC25
export datadir6=ODAC23
export datadir7=OMat24
export datadir8=OMol25
export datadir9=OMol25-neutral
export datadir10=OMol25-non-neutral
export datadir11=OPoly2026
export datadir12=Nabla2DFT
export datadir13=QCML
export datadir14=QM7X
export datadir15=transition1x


# (A) Setup omnistat sampling environment
ml use /sw/frontier/amdsw/modulefiles/
ml omnistat-wrapper
export OMNISTAT_CONFIG=$HYDRAGNN_ROOT/omnistat.hydragnn-external-fp64.config

# (B) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 15

## Verification mode for this script: FSDP2 + SHARD_GRAD_OP, task-parallel optional
export HYDRAGNN_USE_FSDP=1
export HYDRAGNN_FSDP_VERSION=2
export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
TASK_PARALLEL_ARG=""
if [ "$TASK_PARALLEL" = "1" ]; then
    TASK_PARALLEL_ARG="--task_parallel"
fi

## Getting strange omnistat counts with the following line. Otherswise, FSDP runs fine.

MULTI_MODEL_LIST=$datadir0
# MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15

DATASET=datadir$K
cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest -l --kill-on-bad-exit=1 \
python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
    --log=multidataset_hpo-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-FSDP$HYDRAGNN_USE_FSDP-V$HYDRAGNN_FSDP_VERSION-TP$TASK_PARALLEL --everyone \
    --inputfile=gfm_mlip.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH)) \
    --multi --ddstore --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    $TASK_PARALLEL_ARG \
    --precision=fp64 \
    --mpnn_type=EGNN \
    --num_conv_layers=2 \
    --hidden_dim=1000 \
    --num_headlayers=2 \
    --dim_headlayers=300

# (C) End of job: stop data collection
${OMNISTAT_WRAPPER} usermode --stop
