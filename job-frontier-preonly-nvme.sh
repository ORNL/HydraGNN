#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 5
#SBATCH -q debug
#SBATCH -S 0
#SBATCH -C nvme

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi

source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-frontier.sh

source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn

export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier/install/lib/python3.8/site-packages/:$PYTHONPATH

export PYTHONPATH=$PWD:$PYTHONPATH
cd  examples/mptrj/

# SBCAST file from Orion to NVMe -- NOTE: ``-C nvme`` is required to use the NVMe drive
sbcast -pf dataset/MPtrj_2022.9_full.json /mnt/bb/$USER/MPtrj_2022.9_full.json
if [ ! "$?" == "0" ]; then
    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
    # your application may pick up partially complete shared library files, which would give you confusing errors.
    echo "SBCAST failed!"
    exit 1
fi

echo
# Showing the file on the current node -- this will be the same on all other nodes in the allocation
echo "*****SBCAST FILE ON CURRENT NODE******"
ls /mnt/bb/$USER/
echo "**************************************"

srun -n$((SLURM_JOB_NUM_NODES*4)) python -u train.py --preonly --pickle --tmpfs "/mnt/bb/$USER/" 
#python -u train.py --preonly 
