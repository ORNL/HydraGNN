#!/bin/bash
#SBATCH -A m4452_g
#SBATCH -J FNET-HPO
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 4

# Perlmutter user guide: https://docs.nersc.gov/systems/perlmutter/
#
# Scalable hyperparameter optimization for the FNET temporal anomaly-detection
# example. Pre-process the dataset *once* before submitting this job:
#
#   python fnet_temporal_anomaly_detection.py --preonly \
#       --data_root <path/to/FNETDATAforOrnl> --date 2024-06-01 \
#       --cache_dir $PWD/dataset --format pickle

set -x

export MIOPEN_DISABLE_CACHE=1

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=4/' $HOSTS > $HOSTFILE

# ---------------------------------------------------------------------------
# Cluster configuration (Perlmutter: 4 GPUs per node).
# ---------------------------------------------------------------------------
export NNODES=$SLURM_JOB_NUM_NODES
export NNODES_PER_TRIAL=1
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 4 ))
export NGPUS_PER_TRIAL=$(( 4 * $NNODES_PER_TRIAL ))
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL ))
export OMP_NUM_THREADS=4
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!!"

# ---------------------------------------------------------------------------
# DeepHyper logging
# ---------------------------------------------------------------------------
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR

# ---------------------------------------------------------------------------
# FNET-specific configuration (forwarded to every trial subprocess).
# ---------------------------------------------------------------------------
export FNET_CACHE_DIR=${FNET_CACHE_DIR:-$PWD/dataset}
export FNET_DATE=${FNET_DATE:-2024-06-01}
export FNET_FORMAT=${FNET_FORMAT:-pickle}
export NUM_EPOCH=${NUM_EPOCH:-30}
export BATCH_SIZE=${BATCH_SIZE:-16}

# Required for read_node_list() helper used by the HPO driver.
export HYDRAGNN_SYSTEM=perlmutter

# NOTE: the per-trial srun in fnet_temporal_anomaly_detection_deephyper.py uses
# `--ntasks-per-node=8 --gpus-per-node=8` (Frontier defaults). On Perlmutter
# you should edit the `prefix` block in fnet_temporal_anomaly_detection_deephyper.py
# to use `--ntasks-per-node=4 --gpus-per-node=4` (no `--gpu-bind=closest`)
# before launching this script. The launcher below is otherwise identical.

sleep 5

python fnet_temporal_anomaly_detection_deephyper.py --max_evals 100
