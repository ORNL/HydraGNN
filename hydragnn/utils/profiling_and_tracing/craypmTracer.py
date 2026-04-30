from collections import defaultdict
import numpy as np
from mpi4py import MPI

import os

CPU_ENERGY_COUNTERS = None
CPU_ENERGY_TRACERS = None
MEMORY_ENERGY_COUNTERS = None
MEMORY_ENERGY_TRACERS = None
ENERGY_CALLS = None


WORLD_SIZE = MPI.COMM_WORLD.Get_size()
WORLD_RANK = MPI.COMM_WORLD.Get_rank()    

#TRACK_NAME = ["forward", "enc_forward", "branch0_forward", "backward", "dataload", "zero_grad", "get_head_indices", "opt_step","train"]

def get_craypmEnergyCounter():
    path = "/sys/cray/pm_counters/"
    
    ## This should be rank//2 when running on frontier
    ## since there are 2 GCD for each GPU.

    with open(f"{path}cpu_energy") as f:
    #with open(f"{path}accel{rank}_energy") as f:
        data = f.read()
    cpuEnergyCounter = int(data.split()[0])

    with open(f"{path}memory_energy") as f:
    #with open(f"{path}accel{rank}_energy") as f:
        data = f.read()
    memoryEnergyCounter = int(data.split()[0])
    
    return cpuEnergyCounter, memoryEnergyCounter


def initialize():
    
    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS, MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS, ENERGY_CALLS

    CPU_ENERGY_COUNTERS = {}
    CPU_ENERGY_TRACERS = defaultdict(int)
    MEMORY_ENERGY_COUNTERS = {}
    MEMORY_ENERGY_TRACERS = defaultdict(int)
    ENERGY_CALLS = defaultdict(int)
    
def start(name):

    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS, MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS, ENERGY_CALLS
    if name !="get" and (WORLD_RANK%8==0):
        CPU_ENERGY_COUNTERS[name], MEMORY_ENERGY_COUNTERS[name] = get_craypmEnergyCounter()
        ENERGY_CALLS[name] +=1
    pass
    
def stop(name):
    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS,MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS
    if name !="get" and (WORLD_RANK%8==0):

        cpu_end_energy, memory_end_energy = get_craypmEnergyCounter()

        #if cpu_end_energy > CPU_ENERGY_COUNTERS[name]:
        CPU_ENERGY_COUNTERS[name] =  cpu_end_energy - CPU_ENERGY_COUNTERS[name]
            
        #if memory_end_energy > MEMORY_ENERGY_COUNTERS[name]:
        MEMORY_ENERGY_COUNTERS[name] =  memory_end_energy - MEMORY_ENERGY_COUNTERS[name]
            
        CPU_ENERGY_TRACERS[name] += CPU_ENERGY_COUNTERS[name]
        MEMORY_ENERGY_TRACERS[name] += MEMORY_ENERGY_COUNTERS[name]
            
    pass

def enable():
    pass

def disable():
    #nvmlShutdown()
    pass


def reset():
    global CPU_ENERGY_COUNTERS, CPU_ENERGY_TRACERS,MEMORY_ENERGY_COUNTERS, MEMORY_ENERGY_TRACERS
    CPU_ENERGY_COUNTERS = {}
    MEMORY_ENERGY_COUNTERS = {}
    ENERGY_CALLS = defaultdict(int)
    CPU_ENERGY_TRACERS = defaultdict(int)
    MEMORY_ENERGY_TRACERS = defaultdict(int)

def print_device():
    pass

def pr_file(file_path):
    print(f"pr_file:{WORLD_RANK},{WORLD_RANK%8},{WORLD_RANK//8}")
    if WORLD_RANK%8==0:
        dirname = os.path.dirname(file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok = True)

        with open(f"{file_path}_cpu_energy.n{int(WORLD_RANK//8)}", mode="w", encoding="utf-8") as file:
            file.write("name,ncalls,total\n")
            for k,v in CPU_ENERGY_TRACERS.items():
                file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")

        with open(f"{file_path}_memory_energy.n{int(WORLD_RANK//8)}", mode="w", encoding="utf-8") as file:
            file.write("name,ncalls,total\n")
            for k,v in MEMORY_ENERGY_TRACERS.items():
                file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")
