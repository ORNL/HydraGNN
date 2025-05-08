#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN-multibranch
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 5
##SBATCH -S 1

 
# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm624
 
#export python path to use ADIOS2 v.2.9.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

which python
python -c "import numpy; print(numpy.__version__)"

module unload darshan-runtime
module list


echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0

## Getting error without these after 20 nodes
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export FI_MR_CACHE_MONITOR=disabled

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

## Score-P envs
export SCOREP_ENABLE_PROFILING=false
export SCOREP_ENABLE_TRACING=true
export SCOREP_TOTAL_MEMORY=512M
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$SLURM_JOB_ID
# SCOREP_OPT="-m scorep --verbose --keep-files --noinstrumenter --mpp=mpi"
SCOREP_OPT=""

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py  --multi --ddstore --multi_model_list=ANI1x-v3,MPTrj-v3,OC2020-20M-v3,OC2022-v3,qm7x-v3

export datadir0=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/ANI1x-v3.bp
export datadir1=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/qm7x-v3.bp
export datadir2=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/MPTrj-v3.bp
export datadir3=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/Alexandria-v3.bp
export datadir4=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/transition1x-v3.bp


#export datadir4=/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/open_catalyst_2020
#export datadir5=/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/omat24

export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=5
[ -z $BATCH_SIZE ] && BATCH_SIZE=160 ## 320 (Perlmutter) 160 (Frontier)
export BATCH_SIZE=$BATCH_SIZE

# # (1a) Setup omnistat sampling environment
# export OMNISTAT_WRAPPER=/autofs/nccs-svm1_sw/crusher/amdsw/omnistat/1.3.0/misc/omnistat-ornl
# export OMNISTAT_VICSERVER_DATADIR=/lustre/orion/${SLURM_JOB_ACCOUNT}/world-shared/omnistat/${SLURM_JOB_ID}
# # (1b) Enable data collectors and polling (1 sec interval)
# ${OMNISTAT_WRAPPER} usermode --start --interval 15 | tee omnistat_start.log

export SCOREP_EXPERIMENT_DIRECTORY=scorep-$SLURM_JOB_ID-TP0

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
python -u $SCOREP_OPT ./examples/multibranch/train.py --log=GFM_taskparallel_weak-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-BS$BATCH_SIZE-TP0-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
--inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
--multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --batch_size=$BATCH_SIZE --num_epoch=4 \
--oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))

sleep 5
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$SLURM_JOB_ID-TP1

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest \
python -u $SCOREP_OPT ./examples/multibranch/train.py --log=GFM_taskparallel_weak-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-BS$BATCH_SIZE-TP1-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
--inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
--multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --batch_size=$BATCH_SIZE --num_epoch=4 \
--task_parallel --use_devicemesh --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))

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
