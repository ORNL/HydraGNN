#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN-postprocess
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -q debug

# target SLURM job ID and DeepHyper log dir
# CHANGEME: set SLURM job ID for which you want to do post-processing of its omnistat data
export TARGET_SLURM_JOB_ID=206276

# CHANGEME: set DeepHyper log dir corresponding to the above SLURM job
export DEEPHYPER_LOG_DIR="logs/deephyper-experiment"-$TARGET_SLURM_JOB_ID 

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
