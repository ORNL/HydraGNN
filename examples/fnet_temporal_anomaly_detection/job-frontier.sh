#!/bin/bash

#SBATCH -A CPH161
#SBATCH -J FNET-HPO
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 4

# Frontier User Guide: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
#
# Scalable hyperparameter optimization for the FNET temporal anomaly-detection
# example. Pre-process the dataset *once* before submitting this job:
#
#   python fnet_temporal_anomaly_detection.py --preonly \
#       --data_root <path/to/FNETDATAforOrnl> --date 2024-06-01 \
#       --cache_dir $PWD/dataset --format pickle
#
# Then submit this script. Each DeepHyper trial picks a node subset off the
# queue and launches `fnet_temporal_anomaly_detection.py` via `srun`.

set -x

export MIOPEN_DISABLE_CACHE=1

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# ---------------------------------------------------------------------------
# Cluster configuration: 1 trial per 1 node (8 GPUs), all nodes concurrent.
# Adjust NNODES_PER_TRIAL up if a single node is too small for your settings.
# ---------------------------------------------------------------------------
export NNODES=$SLURM_JOB_NUM_NODES
export NNODES_PER_TRIAL=1
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 ))
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL ))
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
# Edit FNET_CACHE_DIR / FNET_DATE / FNET_FORMAT to match your pre-processed
# cache produced by `fnet_temporal_anomaly_detection.py --preonly`.
# ---------------------------------------------------------------------------
export FNET_CACHE_DIR=${FNET_CACHE_DIR:-$PWD/dataset}
export FNET_DATE=${FNET_DATE:-2024-06-01}
export FNET_FORMAT=${FNET_FORMAT:-pickle}
export NUM_EPOCH=${NUM_EPOCH:-30}
export BATCH_SIZE=${BATCH_SIZE:-16}

# Required for read_node_list() helper used by the HPO driver.
export HYDRAGNN_SYSTEM=frontier

sleep 5

# Launch the DeepHyper search (1 driver process; trials run via srun).
python fnet_temporal_anomaly_detection_deephyper.py --max_evals 100
