#!/bin/bash
# Convenience launcher for FT1_topology with HeteroHEAT.
# Usage: sbatch FT1_topology/job-frontier-HeteroHEAT.sh
#   Override regime via SBATCH --export, e.g.:
#     sbatch --export=ALL,FT_REGIME=partial FT1_topology/job-frontier-HeteroHEAT.sh
#SBATCH -A LRN078
#SBATCH -J OPF-FT1-HeteroHEAT
#SBATCH -o /lustre/orion/lrn078/proj-shared/HydraGNN/FT1_topology-HeteroHEAT-%j.out
#SBATCH -e /lustre/orion/lrn078/proj-shared/HydraGNN/FT1_topology-HeteroHEAT-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 8

export FT_STRATEGY=FT1_topology
export FT_ARCH=HeteroHEAT
export FT_REGIME=${FT_REGIME:-full}
export PRETRAINED_MODEL=HeteroHEAT_best

bash $(dirname $0)/../job-frontier-finetune.sh
