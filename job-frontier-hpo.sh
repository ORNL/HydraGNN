#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 10
#SBATCH -q debug

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi

source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-frontier-rocm613.sh
source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm613
#ml omniperf/1.0.10
export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier_rocm613/install/lib/python3.12/site-packages/:$PYTHONPATH

export PYTHONPATH=$PWD:$PYTHONPATH
cd  examples/multidataset_hpo/

export MPLCONFIGDIR=/lustre/orion/cph161/world-shared/mlupopa/

# HPO DeepHyper Configuration 
export NNODES=$SLURM_JOB_NUM_NODES # e.g., 100 total nodes
export NNODES_PER_TRIAL=8
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 )) # e.g., 800 total GPUs
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL )) # e.g., 32 GPUs per training
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL )) # e.g., 25 total DH ranks
export OMP_NUM_THREADS=4 # e.g., 8 threads per rank
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!!" 

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
${OMNISTAT_WRAPPER} usermode --start --interval 1

# (2) Run HPO
python gfm_deephyper_multi.py

# (3) Summarize data collection results
for i in ${DEEPHYPER_LOG_DIR}/trial_map_*; do
	trial=$(echo $i | awk -F_ '{print $(NF-1)}')
	step=$(echo $i | awk -F_ '{print $(NF)}')
	${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 1 --step $step \
		--pdf omnistat-${SLURM_JOB_ID}-$trial.pdf > omnistat-${SLURM_JOB_ID}-$trial.txt
done

${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 1 --pdf omnistat-${SLURM_JOB_ID}.pdf > omnistat-${SLURM_JOB_ID}.txt 

# (4) Tear-down data collection
${OMNISTAT_WRAPPER} usermode --stop

echo "Done."
