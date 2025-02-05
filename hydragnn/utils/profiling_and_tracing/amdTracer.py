
from amdsmi import *
from collections import defaultdict
import numpy as np
import os
import time



if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
    ## Summit
    LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
elif os.getenv("SLURM_LOCALID"):
    ## CADES
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
else:
    LOCAL_RANK = 0

DEVICE_COUNT = None
DEVICE_HANDLER = None
DEVICE_ID = None

ENERGY_COUNTERS = None
ENERGY_TRACERS = None
ENERGY_CALLS = None

TRACK_NAME = ["forward", "backward", "dataload", "train","initLoad"]

def initialize():
    global DEVICE_COUNT, DEVICE_HANDLER, DEVICE_ID, DEVICE_NAME
    global ENERGY_COUNTERS, ENERGY_TRACERS, LOCAL_RANK, ENERGY_CALLS

    amdsmi_init()

    devices = amdsmi_get_processor_handles()
    DEVICE_HANDLER = devices[LOCAL_RANK]
    DEVICE_ID = amdsmi_get_gpu_asic_info(DEVICE_HANDLER)['asic_serial']

    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(int)
    ENERGY_CALLS = defaultdict(int)

def start(name):
    global ENERGY_COUNTERS, ENERGY_CALLS
    if name in TRACK_NAME:
        energy_dict = amdsmi_get_energy_count(DEVICE_HANDLER)
        ENERGY_COUNTERS[name] = energy_dict['power'] * energy_dict['counter_resolution']
        ENERGY_CALLS[name] += 1
    pass

def stop(name):
    
    global ENERGY_COUNTERS, ENERGY_TRACERS
    if name in TRACK_NAME:
        energy_dict = amdsmi_get_energy_count(DEVICE_HANDLER)
        ENERGY_COUNTERS[name] = (energy_dict['power'] * energy_dict['counter_resolution']) - ENERGY_COUNTERS[name]        
        ENERGY_TRACERS[name] += (ENERGY_COUNTERS[name]*1e-6)
    pass

def disable():
    pass

def print_device():
    pass

    
def pr_file(file_path,rank):
    if not os.path.isdir(file_path):
        os.makedirs(file_path, exist_ok=True)
        
    with open(f"{file_path}/amd_dump_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total\n")
        for k,v in ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")            
    amdsmi_shut_down()

