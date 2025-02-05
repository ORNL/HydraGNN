from collections import defaultdict
import numpy as np

from mpi4py import MPI
from pynvml import *
import os

DEVICE_COUNT = None
DEVICE_HANDLER = None
DEVICE_UUID = None
DEVICE_NAME = None

ENERGY_COUNTERS = None
ENERGY_TRACERS = None
ENERGY_CALLS = None

TRACK_NAME = ["forward", "backward", "dataload", "zero_grad", "get_head_indices", "opt_step"]

def initialize():
    global DEVICE_COUNT, DEVICE_HANDLER, DEVICE_UUID, DEVICE_NAME
    global ENERGY_COUNTERS, ENERGY_TRACERS, ENERGY_CALLS
    nvmlInit()    
    DEVICE_COUNT = nvmlDeviceGetCount()
    DEVICE_HANDLER = nvmlDeviceGetHandleByIndex(0)
    DEVICE_UUID = nvmlDeviceGetUUID(DEVICE_HANDLER)
    DEVICE_NAME = nvmlDeviceGetName(DEVICE_HANDLER)
    print(f"Initialized for NVML Handler for {DEVICE_NAME}:{DEVICE_UUID}")

    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(int)
    ENERGY_CALLS = defaultdict(int)
    
def start(name):
    
    global ENERGY_COUNTERS, ENERGY_CALLS
    if name in TRACK_NAME:
        ENERGY_COUNTERS[name] = nvmlDeviceGetTotalEnergyConsumption(DEVICE_HANDLER)
        ENERGY_CALLS[name] += 1
    pass
    
def stop(name):

    global ENERGY_COUNTERS, ENERGY_TRACERS
    if name in TRACK_NAME:
        ENERGY_COUNTERS[name] = nvmlDeviceGetTotalEnergyConsumption(DEVICE_HANDLER) - ENERGY_COUNTERS[name]
        ENERGY_TRACERS[name] += ENERGY_COUNTERS[name]
    pass

def enable():
    pass

def disable():
    #nvmlShutdown()
    pass


def reset():
    global ENERGY_COUNTERS, ENERGY_TRACERS, ENERGY_CALLS
    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(int)
    ENERGY_CALLS = defaultdict(int)
    

def print_device():
   pass

def pr_file(file_path,rank):
    if not os.path.isdir(file_path):
        os.makedirs(file_path, exist_ok = True)
        
    with open(f"{file_path}/nvml_dump_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total\n")
        for k,v in ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")

            
    nvmlShutdown()
