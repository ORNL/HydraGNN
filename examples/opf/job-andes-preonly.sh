#!/bin/bash
#SBATCH -A LRN078
#SBATCH -J HydraGNN
#SBATCH -o job-hydragnn-grid-%j.out
#SBATCH -e job-hydragnn-grid-%j.out
#SBATCH -t 32:00:00
#SBATCH -p batch 
#SBATCH -N 1 #16 
##SBATCH -S 1
 
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
 
# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-andes.sh
source activate /lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Andes/hydragnn_venv
 
#export python path to use ADIOS2 v.2.10.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Andes/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH
 
HYDRAGNN_ROOT=/lustre/orion/lrn078/proj-shared/HydraGNN

#export python path to HydragNN
export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH
 
which python
python -c "import numpy; print(numpy.__version__)"
# Core runtime controls
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

cd $HYDRAGNN_ROOT/examples/opf

# 2) Rebuild processed files once (single rank; safest)
srun -N1 -n1 -c7 python -u train_opf_solution_heterogeneous.py --preonly --adios --num_groups 20 --case_name pglib_opf_case118_ieee pglib_opf_case14_ieee pglib_opf_case2000_goc pglib_opf_case30_ieee pglib_opf_case500_goc pglib_opf_case57_ieee pglib_opf_case6470_rte pglib_opf_case4661_sdet pglib_opf_case10000_goc pglib_opf_case13659_pegase
