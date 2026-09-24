#!/bin/bash
# Convenience launcher for FT3_contingency with HeteroSAGE.
# Usage: sbatch FT3_contingency/job-frontier-HeteroSAGE.sh
#   Override regime via SBATCH --export, e.g.:
#     sbatch --export=ALL,FT_REGIME=partial FT3_contingency/job-frontier-HeteroSAGE.sh
#SBATCH -A LRN078
#SBATCH -J OPF-FT3-HeteroSAGE
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT3_contingency-HeteroSAGE-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT3_contingency-HeteroSAGE-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 8

export FT_STRATEGY=FT3_contingency
export FT_ARCH=HeteroSAGE
export FT_REGIME=${FT_REGIME:-full}
export PRETRAINED_MODEL=HeteroSAGE_best

bash $(dirname $0)/../job-frontier-finetune.sh
