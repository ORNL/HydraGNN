#!/bin/bash -l
#PBS -N HydraGNN
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=flare
###PBS -q debug-scaling
#PBS -q prod
#PBS -A HydraGNN

export NNODES=`wc -l < $PBS_NODEFILE`
export NPROCS_PER_NODE=12 # Number of MPI ranks to spawn per node
export NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
export NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

export NTOTPROCS=$(( NNODES * NPROCS_PER_NODE ))

source /lus/flare/projects/HydraGNN/mlupopa/export_connection.sh

module load frameworks

export TORCH_LIB=$(python -c "import torch; print(torch.__file__)" | sed 's/__init__.py/lib/')
export TORCH_VERSION=`python -c "import torch; print(torch.__version__)" | sed 's/^\([0-9.]*\).*/\1/'`

export LD_LIBRARY_PATH=${TORCH_LIB}:$LD_LIBRARY_PATH

source /lus/flare/projects/HydraGNN/mlupopa/hydragnn_venv/bin/activate

export PYTHONPATH=/lus/flare/projects/HydraGNN/mlupopa/ADIOS/adios2-install/lus/flare/projects/HydraGNN/mlupopa/hydragnn_venv/lib/python3.10/site-packages/:$PYTHONPATH

which python
python -c "import numpy; print(numpy.__version__)"

cd /lus/flare/projects/HydraGNN/mlupopa/HydraGNN

echo $PWD
export PYTHONPATH=$PWD:$PYTHONPATH

echo $LD_LIBRARY_PATH  | tr ':' '\n'

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

export datadir0=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/ANI1x-v3.bp
export datadir1=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/qm7x-v3.bp
export datadir2=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/MPTrj-v3.bp
export datadir3=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/Alexandria-v3.bp
export datadir4=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/transition1x-v3.bp

export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=5
[ -z $EFFECTIVE_BATCH_SIZE ] && EFFECTIVE_BATCH_SIZE=6400 
export EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE
export BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NTOTPROCS))

mpiexec -n ${NTOTPROCS} --ppn ${NPROCS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
python -u ./examples/multibranch/train.py --log=GFM_taskparallel_strong-$PBS_JOBID-NN$NNODES-BS$BATCH_SIZE-TP0-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
--inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
--multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --batch_size=$BATCH_SIZE --num_epoch=4 \
--oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))

sleep 5

mpiexec -n ${NTOTPROCS} --ppn ${NPROCS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
python -u ./examples/multibranch/train.py --log=GFM_taskparallel_strong-$PBS_JOBID-NN$NNODES-BS$BATCH_SIZE-TP1-DD$HYDRAGNN_DDSTORE_METHOD-NW$HYDRAGNN_NUM_WORKERS --everyone \
--inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
--multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --batch_size=$BATCH_SIZE --num_epoch=4 \
--task_parallel --use_devicemesh --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH))
