#!/bin/bash
# Convenience launcher for FT4_task_specific with HeteroSAGE.
# Usage: sbatch FT4_task_specific/job-frontier-HeteroSAGE.sh
#   Override regime via SBATCH --export, e.g.:
#     sbatch --export=ALL,FT_REGIME=partial FT4_task_specific/job-frontier-HeteroSAGE.sh
#SBATCH -A LRN078
#SBATCH -J OPF-FT4-HeteroSAGE
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT4_task_specific-HeteroSAGE-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT4_task_specific-HeteroSAGE-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 8

export FT_STRATEGY=FT4_task_specific
export FT_ARCH=HeteroSAGE
export FT_REGIME=${FT_REGIME:-full}
export PRETRAINED_MODEL=HeteroSAGE_best

bash $(dirname $0)/../job-frontier-finetune.sh
