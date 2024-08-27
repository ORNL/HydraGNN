#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN-postprocess
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -q debug

usage() { echo "Usage: sbatch job-frontier-hpo-postprocess-omnistat.sh -j <target job ID> [-p <path to deephyper log dir>]" 1>&2; exit 1; }

while getopts ":j:p:" o; do
    case "${o}" in
        j)
            TARGET_SLURM_JOB_ID=${OPTARG}
            ;;
        p)
            DEEPHYPER_LOG_DIR=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${TARGET_SLURM_JOB_ID}" ]; then
  usage
fi

if [ -z "${DEEPHYPER_LOG_DIR}" ]; then
  DEEPHYPER_LOG_DIR="logs/deephyper-experiment"-$TARGET_SLURM_JOB_ID
fi

if [ ! -d "${DEEPHYPER_LOG_DIR}" ]; then
  echo "DeepHyper log dir path is invalid."
  usage
fi

echo "TARGET_SLURM_JOB_ID = ${TARGET_SLURM_JOB_ID}"
echo "DEEPHYPER_LOG_DIR = ${DEEPHYPER_LOG_DIR}"

# omnistat variables
export OMNISTAT_WRAPPER=/autofs/nccs-svm1_sw/crusher/amdsw/omnistat/1.0.0-RC1/misc/omnistat-ornl
export OMNISTAT_PROMSERVER_DATADIR=/lustre/orion/${SLURM_JOB_ACCOUNT}/world-shared/omnistat/${TARGET_SLURM_JOB_ID}

# (1) Begin omnistat server
${OMNISTAT_WRAPPER} usermode --startserver

# (2) Summarize data collection results
for i in ${DEEPHYPER_LOG_DIR}/trial_map_*; do
	trial=$(echo $i | awk -F_ '{print $(NF-1)}')
	step=$(echo $i | awk -F_ '{print $(NF)}')
	${OMNISTAT_WRAPPER} query --job ${TARGET_SLURM_JOB_ID} --interval 1 --step $step \
		--pdf omnistat-${TARGET_SLURM_JOB_ID}-$trial.pdf > omnistat-${TARGET_SLURM_JOB_ID}-$trial.txt
done

${OMNISTAT_WRAPPER} query --job ${TARGET_SLURM_JOB_ID} --interval 1 --pdf omnistat-${TARGET_SLURM_JOB_ID}.pdf > omnistat-${TARGET_SLURM_JOB_ID}.txt 

# (3) Tear-down omnistat server
${OMNISTAT_WRAPPER} usermode --stopserver

echo "Done."
