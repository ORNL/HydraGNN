#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -N 8626
#SBATCH -C nvme
#SBATCH --exclude=frontier00318,frontier05378,frontier05387
##SBATCH --signal=SIGUSR1@180

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
# export MPICH_OFI_NIC_POLICY=GPU
export MPICH_OFI_NIC_POLICY=NUMA
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi
export PYTHONNOUSERSITE=1

function check_node()
{
    [ ! -d .node.status ] && mkdir .node.status
    ssh $1 hostname 2> /dev/null
    [ $? -eq 0 ] && touch .node.status/$1
}

function check_badnodes()
{
    ## Check bad nodes
    for NODE in `scontrol show hostnames`; do
        check_node $NODE &
    done
    wait
    ## Debugging
    ls .node.status/* | tail -n 2 | xargs rm -f

    BAD_NODELIST=""
    for NODE in `scontrol show hostnames`; do
        [ ! -f .node.status/$NODE ] && BAD_NODELIST="$NODE,$BAD_NODELIST"
    done
    [ ! -z $BAD_NODELIST ] && [ ${BAD_NODELIST: -1} == "," ] && BAD_NODELIST=${BAD_NODELIST::-1}
    export HYDRAGNN_EXCLUDE_NODELIST=$BAD_NODELIST
    echo "HYDRAGNN_EXCLUDE_NODELIST: $HYDRAGNN_EXCLUDE_NODELIST"
}

function setup_bb()
{
    ENVNAME=hydragnn-py3.12-rocm5.7.1-mpich8.1.26
    # Move a copy of the env to the NVMe on each node
    if [ -d /mnt/bb/${USER} ]; then
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
        echo "copying hydragnn env to each node in the job"
        if [ ! -f /mnt/bb/${USER}/${ENVNAME}.tar.gz ]; then
            time sbcast -pf /lustre/orion/world-shared/cph161/HydraGNN-gb24-comm/bb/${ENVNAME}.tar.gz /mnt/bb/${USER}/${ENVNAME}.tar.gz
            if [ ! "$?" == "0" ]; then
                # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
                # your application may pick up partially complete shared library files, which would give you confusing errors.
                echo "SBCAST failed!"
                exit 1
            fi
        fi

        # Untar the environment file (only need 1 task per node to do this)
        if [ ! -d /mnt/bb/${USER}/${ENVNAME} ]; then
            time srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u tar -xzf /mnt/bb/${USER}/${ENVNAME}.tar.gz -C  /mnt/bb/${USER}/
            if [ ! "$?" == "0" ]; then
                echo "srun untar failed!"
                exit 1
            fi
        fi

        export PATH=/mnt/bb/${USER}/${ENVNAME}/bin:$PATH
        export PYTHONPATH=/mnt/bb/${USER}/${ENVNAME}/lib/python3.12/site-packages:$PYTHONPATH
        export LD_LIBRARY_PATH=/mnt/bb/${USER}/${ENVNAME}/lib:$LD_LIBRARY_PATH

        # Check: should have torch in NVME
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u python -u -c "import torch; print(torch.__file__)"
    fi
}

function dosummarize()
{
    echo 'OMNISTAT closing ...'
    # (3) Summarize data collection results
    for i in ${DEEPHYPER_LOG_DIR}/trial_map_*; do
        trial=$(echo $i | awk -F_ '{print $(NF-1)}')
        step=$(echo $i | awk -F_ '{print $(NF)}')
        ${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 15 --step $step \
            --pdf omnistat-${SLURM_JOB_ID}-$trial.pdf > omnistat-${SLURM_JOB_ID}-$trial.txt
    done

    ${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 15 --pdf omnistat-${SLURM_JOB_ID}.pdf > omnistat-${SLURM_JOB_ID}.txt 

    # (4) Tear-down data collection
    ${OMNISTAT_WRAPPER} usermode --stop
    echo 'OMNISTAT done.'
}

## Run "dosummarize" before terminating
# trap 'dosummarize' SIGUSR1

# source module-to-load-frontier-py312-rocm6.1.3-mpich8.1.26.sh

#ml omniperf/1.0.10
# export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier_rocm613/install/lib/python3.12/site-packages/:$PYTHONPATH

export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=/lustre/orion/world-shared/cph161/jyc/frontier/sw/aws-ofi-rccl/devel-rocm${CRAY_ROCM_VERSION}/lib:$LD_LIBRARY_PATH

WDIR=examples/multidataset_hpo_NN${SLURM_JOB_NUM_NODES}_${SLURM_JOB_ID}
echo "workdir: $WDIR"
cp -r examples/multidataset_hpo ${WDIR}
cd ${WDIR}

## Setup NVME
setup_bb

#export MPLCONFIGDIR=/lustre/orion/cph161/world-shared/mlupopa/

# HPO DeepHyper Configuration 
export NNODES=$SLURM_JOB_NUM_NODES # e.g., 100 total nodes
export NNODES_PER_TRIAL=128
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 )) # e.g., 800 total GPUs
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL )) # e.g., 32 GPUs per training
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL )) # e.g., 25 total DH ranks
export OMP_NUM_THREADS=4 # e.g., 8 threads per rank
[ $NTOTGPUS -lt $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!! Not enough GPUs. Exit" && exit

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID 
mkdir -p $DEEPHYPER_LOG_DIR
export DEEPHYPER_DB_HOST=$HOST
# Start Redis server (shared memory between search processes)
# TODO: install Redis and set the `redis.conf` path here
#export REDIS_CONF=...
#pushd $DEEPHYPER_LOG_DIR
#redis-server $REDIS_CONF &
#popd

# (1a) Setup omnistat sampling environment
export OMNISTAT_WRAPPER=/autofs/nccs-svm1_sw/crusher/amdsw/omnistat/1.0.0-RC1/misc/omnistat-ornl
# (1b) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 15 | tee omnistat_start.log

## Check bad nodes
BAD_NODELIST=""
# BAD_NODELIST=`grep "Missing exporter" omnistat_start.log | awk '{print $4}' | tr '\n' ','`
[ -f omnistat_failed_hosts.${SLURM_JOB_ID}.out ] && BAD_NODELIST=`cat omnistat_failed_hosts.${SLURM_JOB_ID}.out | tr '\n' ','`
[ ! -z $BAD_NODELIST ] && [ ${BAD_NODELIST: -1} == "," ] && BAD_NODELIST=${BAD_NODELIST::-1}
export HYDRAGNN_EXCLUDE_NODELIST=$BAD_NODELIST
echo "HYDRAGNN_EXCLUDE_NODELIST: $HYDRAGNN_EXCLUDE_NODELIST"
NUM_EXCLUDE_NODES=0
[ ! -z $HYDRAGNN_EXCLUDE_NODELIST ] && NUM_EXCLUDE_NODES=`echo $HYDRAGNN_EXCLUDE_NODELIST | tr ',' '\n' | wc -l`
[ $NTOTGPUS -lt $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS + $NUM_EXCLUDE_NODES)) ] && echo "ERROR!! Not enough GPUs. The num. of excluded nodes: $NUM_EXCLUDE_NODES" && exit

## Checking ENVs
which python
python -c "import torch; print(torch.__file__)"
python -c "import torch; print(torch.__version__)"
echo PYTHONPATH=$PYTHONPATH

# (2) Run HPO
python gfm_deephyper_multi.py

# dosummarize
# # (3) Summarize data collection results
# for i in ${DEEPHYPER_LOG_DIR}/trial_map_*; do
#     trial=$(echo $i | awk -F_ '{print $(NF-1)}')
#     step=$(echo $i | awk -F_ '{print $(NF)}')
#     ${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 15 --step $step \
#         --pdf omnistat-${SLURM_JOB_ID}-$trial.pdf > omnistat-${SLURM_JOB_ID}-$trial.txt
# done

# ${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 15 --pdf omnistat-${SLURM_JOB_ID}.pdf > omnistat-${SLURM_JOB_ID}.txt 

# # (4) Tear-down data collection
# ${OMNISTAT_WRAPPER} usermode --stop

echo "Done."
