from collections import defaultdict
import numpy as np
import time
from mpi4py import MPI
import os

DEVICE_COUNT = None
DEVICE_HANDLER = None
DEVICE_UUID = None
DEVICE_NAME = None

ENERGY_COUNTERS = None
ENERGY_TRACERS = None
ENERGY_CALLS = None

TRACK_NAME = ["forward", "backward", "dataload", "zero_grad", "get_head_indices", "opt_step", "train"]

def initialize():

    global ENERGY_COUNTERS, ENERGY_TRACERS, ENERGY_CALLS
    print(f"Initialized for Time Handler")

    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(list)
    ENERGY_CALLS = defaultdict(int)
    
def start(name):
    
    global ENERGY_COUNTERS, ENERGY_CALLS
    if name in TRACK_NAME:
        ENERGY_COUNTERS[name] = time.time()
        ENERGY_CALLS[name] += 1
    pass
    
def stop(name):

    global ENERGY_COUNTERS, ENERGY_TRACERS
    if name in TRACK_NAME:
        ENERGY_COUNTERS[name] = time.time() - ENERGY_COUNTERS[name]
        #ENERGY_TRACERS[name] += ENERGY_COUNTERS[name]
        ENERGY_TRACERS[name].append(ENERGY_COUNTERS[name])
    pass

def enable():
    pass

def disable():
    #nvmlShutdown()
    pass


def reset():
    global ENERGY_COUNTERS, ENERGY_TRACERS, ENERGY_CALLS
    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(list)
    ENERGY_CALLS = defaultdict(int)
    

def print_device():
   pass

def pr_file(file_path,rank):
    if not os.path.isdir(file_path):
        os.makedirs(file_path, exist_ok = True)
        
    with open(f"{file_path}/time_dump_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total,mean,std\n")
        for k,v in ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{np.sum(v)},{np.mean(v)},{np.std(v)}\n")

            
