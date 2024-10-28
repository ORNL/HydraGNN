#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 2560
#SBATCH -C nvme
##SBATCH --exclude=frontier00318,frontier05378,frontier05387
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

WDIR=examples/multidataset_hpo_NN8626_2484338_gfm_long_v3_${SLURM_JOB_ID}_NN${SLURM_JOB_NUM_NODES}
# WDIR=examples/multidataset_hpo_NN8626_2484338_gfm_long_v3_2686216_NN2560
if [ ! -d $WDIR ]; then
    ## For restarting
    mkdir -p $WDIR
    echo "workdir: $WDIR"
    cd ${WDIR}
    cp ../multidataset_hpo_NN8626_2484338/gfm.py .
    cp -r ../multidataset_hpo_NN8626_2484338/dataset .
else
    echo "workdir: $WDIR"
    cd ${WDIR}
fi

## Setup NVME
setup_bb

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR

# (1a) Setup omnistat sampling environment
export OMNISTAT_WRAPPER=/autofs/nccs-svm1_sw/crusher/amdsw/omnistat/1.0.0-RC1/misc/omnistat-ornl
# (1b) Enable data collectors and polling (1 sec interval)
# ${OMNISTAT_WRAPPER} usermode --start --interval 15 | tee omnistat_start.log

## Checking ENVs
which python
python -c "import torch; print(torch.__file__)"
python -c "import torch; print(torch.__version__)"
echo PYTHONPATH=$PYTHONPATH

# (2) Run HPO Long
[ -z $TASK_ID_LIST ] && TASK_ID_LIST="229 156 147 260 165 78 137 1 175 171 181 67 179 351 167"

for TASK_ID in $TASK_ID_LIST; do

    ## Setup
    if [ ! -d logs/gfm_0.$TASK_ID ]; then
        mkdir -p logs/gfm_0.$TASK_ID
        cp ../multidataset_hpo_NN8626_2484338/output-0.$TASK_ID.txt .
        cp ../multidataset_hpo_NN8626_2484338/logs/gfm_0.$TASK_ID/config.json logs/gfm_0.$TASK_ID/

        yq -i '.NeuralNetwork.Training.num_epoch = 10000' logs/gfm_0.$TASK_ID/config.json
        yq -i '.NeuralNetwork.Training.continue = 0' logs/gfm_0.$TASK_ID/config.json
        yq -i '.NeuralNetwork.Training.EarlyStopping = true' logs/gfm_0.$TASK_ID/config.json
        yq -i '.NeuralNetwork.Training.patience = 10' logs/gfm_0.$TASK_ID/config.json
        yq -i '.NeuralNetwork.Training.Checkpoint = true' logs/gfm_0.$TASK_ID/config.json
        yq -i '.NeuralNetwork.Training.checkpoint_warmup = 1' logs/gfm_0.$TASK_ID/config.json
    else
        ## Resume
        CHECKPOINT_FILE=`readlink logs/gfm_0.$TASK_ID/gfm_0.$TASK_ID.pk`
        if [ ! -z $CHECKPOINT_FILE ]; then
            EPOCH_LAST=`echo echo ${CHECKPOINT_FILE%.*} | cut -d'_' -f4`
            EPOCH_START=$((EPOCH_LAST+1))
            echo "EPOCH_START: $EPOCH_START"
            yq -i ".NeuralNetwork.Training.epoch_start = $EPOCH_START" logs/gfm_0.$TASK_ID/config.json
        fi
    fi

    CMD=`head -n 1 output-0.${TASK_ID}.txt`
    CMD=`echo $CMD | sed 's/Command = //g'`

    OPT=${CMD}
    OPT=`echo $OPT | sed 's/OC2020-20M-v2/OC2020-v2/g'`
    OPT=`echo $OPT | sed 's/-v2/-v3/g'`
    OPT=`echo $OPT | sed 's/.* \(--model_type.*\); \/autofs.*/\1/g'`
    OPT=`echo $OPT | sed 's/--num_epoch=10/--num_epoch=10000/g'`
    echo $OPT

    echo srun -N 128 -n 1024 -u --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task 4 --threads-per-core 1 --cpu-bind threads --gpus-per-task=1 --gpu-bind=closest \
        --export=ALL,HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1,HYDRAGNN_AGGR_BACKEND=mpi \
        --output ${DEEPHYPER_LOG_DIR}/output_${SLURM_JOB_ID}_0.${TASK_ID}.txt --error ${DEEPHYPER_LOG_DIR}/error_${SLURM_JOB_ID}_0.${TASK_ID}.txt \
        bash -c "touch ${DEEPHYPER_LOG_DIR}/trial_map_0.${TASK_ID}_\$SLURM_STEP_ID; ${OMNISTAT_WRAPPER} rms; python -u gfm.py --inputfile=logs/gfm_0.${TASK_ID}/config.json $OPT; ${OMNISTAT_WRAPPER} rms --nostep;"

    srun -N 128 -n 1024 -u --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task 4 --threads-per-core 1 --cpu-bind threads --gpus-per-task=1 --gpu-bind=closest \
        --export=ALL,HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1,HYDRAGNN_AGGR_BACKEND=mpi \
        --output ${DEEPHYPER_LOG_DIR}/output_${SLURM_JOB_ID}_0.${TASK_ID}.txt --error ${DEEPHYPER_LOG_DIR}/error_${SLURM_JOB_ID}_0.${TASK_ID}.txt \
        bash -c "touch ${DEEPHYPER_LOG_DIR}/trial_map_0.${TASK_ID}_\$SLURM_STEP_ID; ${OMNISTAT_WRAPPER} rms; python -u gfm.py --inputfile=logs/gfm_0.${TASK_ID}/config.json $OPT; ${OMNISTAT_WRAPPER} rms --nostep;" &

    sleep 60
done
wait

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
