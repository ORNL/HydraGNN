#!/bin/bash
# Debug reserve command: salloc -A LRN070 -J HydraGNN -t 01:00:00 -q debug -N 1
# source debug-setup.sh
 
# Load conda environemnt
module reset
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
# source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm631.sh
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda deactivate
conda activate hydragnn_rocm624
# conda activate hydragnn_rocm631
 
#export python path to use ADIOS2 v.2.9.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH
# #export python path to use ADIOS2 for ROCM 6.31 (was v.2.10.2 but v.2.9.2 for now because of problem)
# export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm631/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

export PYTHONPATH=/lustre/orion/lrn070/world-shared/rylieweaver/Direct-vs-Autodiff-ForceComp/HydraGNN:$PYTHONPATH

which python
python -c "import numpy; print(numpy.__version__)"

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA


#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py  --multi --ddstore --multi_model_list=ANI1x-v3,MPTrj-v3,OC2020-20M-v3,OC2022-v3,qm7x-v3

export datadir0=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/ANI1x-v3.bp
export datadir1=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/qm7x-v3.bp
export datadir2=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/MPTrj-v3.bp
export datadir3=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/Alexandria-v3.bp
export datadir4=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/transition1x-v3.bp


#export datadir4=/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/open_catalyst_2020
#export datadir5=/lustre/orion/lrn070/world-shared/mlupopa/Supercomputing2025/HydraGNN/examples/omat24

#TRAINING
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py --log=SC25_GFM_multibranch_login_EGNN --inputfile=multibranch_GFM260_SC25_EGNN.json --num_samples=10000 --multi --ddstore --multi_model_list=$datadir1
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py --log=SC25_GFM_multibranch_physics-informed_login_EGNN --inputfile=multibranch_GFM260_SC25_physics-informed_EGNN.json --num_samples=10000 --multi --ddstore --multi_model_list=$datadir1

#INFERENCE
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES * 8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/inference.py --log="/lustre/orion/lrn070/world-shared/rylieweaver/Direct-vs-Autodiff-ForceComp/HydraGNN/logs/SC25_GFM_multibranch_login_EGNN" --multi --ddstore --multi_model_list="/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/qm7x-v3.bp" --num_samples=10000
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES * 8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/inference_physics-informed.py --log="/lustre/orion/lrn070/world-shared/rylieweaver/Direct-vs-Autodiff-ForceComp/HydraGNN/logs/SC25_GFM_multibranch_physics-informed_login_EGNN" --multi --ddstore --multi_model_list="/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/qm7x-v3.bp" --num_samples=10000
