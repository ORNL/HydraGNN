from collections import defaultdict
import numpy as np

import os

GPU_ENERGY_COUNTERS = None
GPU_ENERGY_TRACERS = None

CPU_ENERGY_COUNTERS = None
CPU_ENERGY_TRACERS = None

MEMORY_ENERGY_COUNTERS = None
MEMORY_ENERGY_TRACERS = None

ENERGY_CALLS = None

if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
    ## Summit
    LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
elif os.getenv("SLURM_LOCALID"):
    ## CADES
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
else:
    LOCAL_RANK = 0

TRACK_NAME = ["forward", "backward", "dataload", "train","initLoad"]

def get_craypmGPUCounter(rank):
    path = "/sys/cray/pm_counters/"
    
    ## This should be rank//2 when running on frontier
    ## since there are 2 GCD for each GPU.

    with open(f"{path}accel{rank//2}_energy") as f:
    #with open(f"{path}accel{rank}_energy") as f:
        data = f.read()
    energyCounter = int(data.split()[0])
    return energyCounter

def get_craypmCPUCounter():
    if LOCAL_RANK == 0:
        
        path = "/sys/cray/pm_counters/"

        with open(f"{path}cpu_energy") as f:
            data = f.read()
        energyCounter = int(data.split()[0])
        return energyCounter
    else:
        return 0

def get_craypmMemoryCounter():
    if LOCAL_RANK == 0:
        path = "/sys/cray/pm_counters/"

        with open(f"{path}memory_energy") as f:
            data = f.read()
        energyCounter = int(data.split()[0])
        return energyCounter
    else:
        return 0

def initialize():
    
    global GPU_ENERGY_COUNTERS, GPU_ENERGY_TRACERS, ENERGY_CALLS
    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS
    global MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS

    
    GPU_ENERGY_COUNTERS = {}
    GPU_ENERGY_TRACERS = defaultdict(int)
    
    CPU_ENERGY_COUNTERS = {}
    CPU_ENERGY_TRACERS = defaultdict(int)
    
    MEMORY_ENERGY_COUNTERS = {}
    MEMORY_ENERGY_TRACERS = defaultdict(int)
    ENERGY_CALLS = defaultdict(int)
    
def start(name):

    global GPU_ENERGY_COUNTERS, GPU_ENERGY_TRACERS, ENERGY_CALLS
    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS
    global MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS
    if name in TRACK_NAME:
        GPU_ENERGY_COUNTERS[name] = get_craypmGPUCounter(LOCAL_RANK)
        CPU_ENERGY_COUNTERS[name] = get_craypmCPUCounter()
        MEMORY_ENERGY_COUNTERS[name] = get_craypmMemoryCounter()
        ENERGY_CALLS[name] +=1
    pass
    
def stop(name):

    global GPU_ENERGY_COUNTERS, GPU_ENERGY_TRACERS
    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS
    global MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS
    if name in TRACK_NAME:
        GPU_ENERGY_COUNTERS[name] = get_craypmGPUCounter(LOCAL_RANK) - GPU_ENERGY_COUNTERS[name]
        GPU_ENERGY_TRACERS[name] += GPU_ENERGY_COUNTERS[name]

        CPU_ENERGY_COUNTERS[name] = get_craypmCPUCounter() - CPU_ENERGY_COUNTERS[name]
        CPU_ENERGY_TRACERS[name] += CPU_ENERGY_COUNTERS[name]

        MEMORY_ENERGY_COUNTERS[name] = get_craypmMemoryCounter() - MEMORY_ENERGY_COUNTERS[name]
        MEMORY_ENERGY_TRACERS[name] += MEMORY_ENERGY_COUNTERS[name]
            
    pass

def enable():
    pass

def disable():
    #nvmlShutdown()
    pass


def reset():
    global GPU_ENERGY_COUNTERS, GPU_ENERGY_TRACERS
    GPU_ENERGY_COUNTERS = {}
    ENERGY_CALLS = defaultdict(int)
    GPU_ENERGY_TRACERS = defaultdict(int)
    

def print_device():
    pass

def pr_file(file_path,rank):
    if not os.path.isdir(file_path):
        os.makedirs(file_path, exist_ok = True)
        
    with open(f"{file_path}/cray_gpu_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total\n")
        for k,v in GPU_ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")

    with open(f"{file_path}/cray_cpu_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total\n")
        for k,v in CPU_ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")

    with open(f"{file_path}/cray_memory_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total\n")
        for k,v in MEMORY_ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")
