#!/bin/bash -l
#PBS -N HydraGNN
#PBS -l select=32
#PBS -l place=scatter
#PBS -l walltime=6:00:00
#PBS -l filesystems=flare
#PBS -q prod
#PBS -A HydraGNN
#PBS -m abe
#PBS -M mehtakv@ornl.gov


echo -e "====== JOB SCRIPT ======"
cat "$0"
echo -e "====== END OF JOB SCRIPT ======"

export NNODES=`wc -l < $PBS_NODEFILE`
export NPROCS_PER_NODE=12 # Number of MPI ranks to spawn per node
export NDEPTH=7 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
export NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)
export NTOTPROCS=$(( NNODES * NPROCS_PER_NODE ))

module load frameworks

export TORCH_LIB=$(python -c "import torch; print(torch.__file__)" | sed 's/__init__.py/lib/')
export TORCH_VERSION=`python -c "import torch; print(torch.__version__)" | sed 's/^\([0-9.]*\).*/\1/'`
export LD_LIBRARY_PATH=${TORCH_LIB}:$LD_LIBRARY_PATH
source /lus/flare/projects/HydraGNN/mlupopa/hydragnn_venv/bin/activate
export PYTHONPATH=/lus/flare/projects/HydraGNN/mlupopa/ADIOS/adios2-install/lus/flare/projects/HydraGNN/mlupopa/hydragnn_venv/lib/python3.10/site-packages/:$PYTHONPATH

cd /lus/flare/projects/HydraGNN/kmehta/HydraGNN-max-fork

echo $PWD
export PYTHONPATH=$PWD:$PYTHONPATH

export HYDRAGNN_MAX_NUM_BATCH=5
export HYDRAGNN_VALTEST=0	# disables validation and testing

# export datadir0=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/ANI1x-v3.bp
# export datadir1=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/qm7x-v3.bp
# export datadir2=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/MPTrj-v3.bp
# export datadir3=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/Alexandria-v3.bp
# export datadir4=/lus/flare/projects/HydraGNN/mlupopa/HydraGNN/examples/multibranch/dataset/transition1x-v3.bp

# Copy data to /tmp
set -x
mpiexec -n $NNODES -ppn 1 rm -rf /tmp/datasets
time mpiexec -n $NNODES -ppn 1 cp -r /lus/flare/projects/HydraGNN/kmehta/datasets /tmp/.
set +x
if [ $? -ne 0 ]; then
	echo "Could not load data to /tmp. Exiting"
	exit 1
fi

export datadir0=/tmp/datasets/ANI1x-v3.bp
export datadir1=/tmp/datasets/qm7x-v3.bp
export datadir2=/tmp/datasets/MPTrj-v3.bp
export datadir3=/tmp/datasets/Alexandria-v3.bp
export datadir4=/tmp/datasets/transition1x-v3.bp


for batch_size in 800 1600 3200 6400 12800 25600 51200 102400 204800 409600 819200 1638400 3276800 6553600; do
	for NTOTPROCS in 5 10 20 40 80 1; do

		export EFFECTIVE_BATCH_SIZE=$batch_size
		export BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NTOTPROCS))
		LOGNAME=GFM-strong-NRANK${NTOTPROCS}-EBS${EFFECTIVE_BATCH_SIZE}-LBS${BATCH_SIZE}
		GPSUMMARY="/lus/flare/projects/HydraGNN/kmehta/HydraGNN-max-fork/logs/${LOGNAME}/gp_timing.summary"

		if [ "$BATCH_SIZE" -lt 32 ] || [ "$BATCH_SIZE" -gt 4096 ]; then
			continue
		fi

		# this test was already performed
		if [ -e $GPSUMMARY ]; then
			echo "Skipping $GPSUMMARY as it is already done."
			continue
		fi

		echo "Node count: $nnodes, num_processes: $NTOTPROCS, effective batch size: $EFFECTIVE_BATCH_SIZE, local batch size: $BATCH_SIZE"
		
		echo "Starting experiment at `date`"
		CMD="timeout --signal=TERM --kill-after=10s 10m mpiexec -n ${NTOTPROCS} --ppn ${NPROCS_PER_NODE} --depth=7 --cpu-bind depth python -u ./examples/multibranch/train.py --log=$LOGNAME --everyone --inputfile=multibranch_GFM260.json --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) --multi --ddstore --multi_model_list=${datadir0},${datadir1},${datadir2},${datadir3},${datadir4} --batch_size=${BATCH_SIZE} --num_epoch=2"
		set -x
		$CMD
		set +x
		echo "Finished experiment at `date`"
		sleep 1
	done
done

echo -e "DONE"
sleep 1

