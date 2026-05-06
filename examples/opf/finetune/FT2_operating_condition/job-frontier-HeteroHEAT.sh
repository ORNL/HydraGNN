#!/bin/bash
# Convenience launcher for FT2_operating_condition with HeteroHEAT.
# Usage: sbatch FT2_operating_condition/job-frontier-HeteroHEAT.sh
#   Override regime via SBATCH --export, e.g.:
#     sbatch --export=ALL,FT_REGIME=partial FT2_operating_condition/job-frontier-HeteroHEAT.sh
#SBATCH -A LRN078
#SBATCH -J OPF-FT2-HeteroHEAT
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT2_operating_condition-HeteroHEAT-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT2_operating_condition-HeteroHEAT-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 8

export FT_STRATEGY=FT2_operating_condition
export FT_ARCH=HeteroHEAT
export FT_REGIME=${FT_REGIME:-full}
export PRETRAINED_MODEL=HeteroHEAT_best

bash $(dirname $0)/../job-frontier-finetune.sh
