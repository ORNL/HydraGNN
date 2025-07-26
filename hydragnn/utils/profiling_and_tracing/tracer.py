"""
This is a tracer package to act as a wrapper to execute gptl and/or scorep.
"""

from __future__ import absolute_import
from functools import wraps
from contextlib import contextmanager

from abc import ABC, abstractmethod
import torch
from mpi4py import MPI

from collections import defaultdict
import numpy as np
from hydragnn.utils.distributed import get_comm_size_and_rank


class Tracer(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def start(self, name):
        pass

    @abstractmethod
    def stop(self, name):
        pass

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def disable(self):
        pass


try:
    import gptl4py as gp

    class GPTLTracer(Tracer):
        def __init__(self, **kwargs):
            gp.initialize()

        def start(self, name):
            gp.start(name)

        def stop(self, name):
            gp.stop(name)

        def enable(self):
            gp.enable()

        def disable(self):
            gp.disable()

        def reset(self):
            gp.reset()


except:
    pass


try:
    import scorep.user as sp

    class SCOREPTracer(Tracer):
        def __init__(self, **kwargs):
            pass

        def start(self, name):
            sp.region_begin(name)

        def stop(self, name):
            sp.region_end(name)

        def enable(self):
            sp.enable_recording()

        def disable(self):
            sp.disable_recording()

        def reset(self):
            pass


except:
    pass

try:
    from pynvml import *

    class NVMLTracer:
        def __init__(self, **kwargs):
            nvmlInit()

            deviceCount = nvmlDeviceGetCount()
            if os.getenv("ROCR_VISIBLE_DEVICES"):
                self.device = int(os.getenv("ROCR_VISIBLE_DEVICES").split(",")[0])
            elif os.getenv("CUDA_VISIBLE_DEVICES"):
                self.device = int(os.getenv("CUDA_VISIBLE_DEVICES").split(",")[0])
            else:
                self.device = 0
            self.d_handle = nvmlDeviceGetHandleByIndex(self.device)
            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)
            self.enabled = True
            print(f"NVMLTracer initalized: rank={self.rank}, device={self.device}")

        def start(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = nvmlDeviceGetTotalEnergyConsumption(
                self.d_handle
            )

        def stop(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = (
                nvmlDeviceGetTotalEnergyConsumption(self.d_handle)
                - self.energyCounters[name]
            )
            self.energyTracer[name].append(self.energyCounters[name])

        def enable(self):
            self.enabled = True

        def disable(self):
            self.enabled = False

        def reset(self):
            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)

        def pr_file(self, file_path):
            dirname = os.path.dirname(file_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            with open(file_path, mode="w", encoding="utf-8") as file:
                file.write("name, ncalls, mean, total, median, std_dev, max, min\n")
                for k, v in self.energyTracer.items():
                    mean_energy = np.mean(v)
                    total_energy = np.sum(v)
                    median_energy = np.median(v)
                    stdDev = np.std(v)
                    max_energy = np.max(v)
                    min_energy = np.min(v)

                    file.write(
                        f"{k}, {len(v)}, {mean_energy}, {total_energy}, {median_energy}, {stdDev}, {max_energy}, {min_energy}\n"
                    )


except:
    pass

try:
    ## Ref: https://rocm.docs.amd.com/projects/rocm_smi_lib/en/develop/tutorials/python_tutorials.html
    import os
    import sys

    ROCM_PATH = os.getenv("ROCM_PATH", "/opt/rocm")
    sys.path.append(f"{ROCM_PATH}/libexec/rocm_smi/")

    import rocm_smi
    from rsmiBindings import *
    from ctypes import *

    def safe_float(s):
        try:
            return float(s)
        except ValueError:
            return np.nan

    def get_energy(rocmsmi, device):
        """Accumulated Energy (uJ)"""
        try:
            power = c_uint64()
            timestamp = c_uint64()
            counter_resolution = c_float()
            ret = rocmsmi.rsmi_dev_energy_count_get(
                device, byref(power), byref(counter_resolution), byref(timestamp)
            )
            return power.value * counter_resolution.value
        except Exception as e:
            print(f"An error occurred: {e}")
            return np.nan

    class ROCMTracer:
        def __init__(self, **kwargs):
            rocm_smi.initializeRsmi()
            self.rocmsmi = initRsmiBindings()
            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)
            self.enabled = True
            self.rank = MPI.COMM_WORLD.Get_rank()

            if os.getenv("ROCR_VISIBLE_DEVICES"):
                self.device = int(os.getenv("ROCR_VISIBLE_DEVICES").split(",")[0])
            elif os.getenv("CUDA_VISIBLE_DEVICES"):
                self.device = int(os.getenv("CUDA_VISIBLE_DEVICES").split(",")[0])
            else:
                self.device = rocm_smi.listDevices()[0]
            print(f"ROCMTracer initalized: rank={self.rank}, device={self.device}")

        def start(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = get_energy(self.rocmsmi, self.device)

        def stop(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = (
                get_energy(self.rocmsmi, self.device) - self.energyCounters[name]
            )
            self.energyTracer[name].append(self.energyCounters[name])

        def enable(self):
            self.enabled = True

        def disable(self):
            self.enabled = False

        def reset(self):
            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)

        def pr_file(self, file_path):
            dirname = os.path.dirname(file_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            with open(file_path, mode="w", encoding="utf-8") as file:
                file.write("name, ncalls, mean, total, median, std_dev, max, min\n")
                for k, v in self.energyTracer.items():
                    mean_energy = np.mean(v)
                    total_energy = np.sum(v)
                    median_energy = np.median(v)
                    stdDev = np.std(v)
                    max_energy = np.max(v)
                    min_energy = np.min(v)

                    file.write(
                        f"{k}, {len(v)}, {mean_energy}, {total_energy}, {median_energy}, {stdDev}, {max_energy}, {min_energy}\n"
                    )


except:
    print("ROCMTracer not available, please check ROCM installation.")
    pass

__tracer_list__ = dict()


def has(name):
    return name in __tracer_list__


def initialize(
    trlist=["GPTLTracer", "SCOREPTracer", "NVMLTracer", "ROCMTracer"],
    verbose=False,
    **kwargs,
):
    for trname in trlist:
        try:
            tr = globals()[trname](**kwargs)
            __tracer_list__[trname] = tr
        except Exception as e:
            if verbose:
                print("tracer loading error:", trname, e)
            pass


def start(name, cudasync=False, sync=False):
    if cudasync and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
    if sync:
        MPI.COMM_WORLD.Barrier()
    for tr in __tracer_list__.values():
        tr.start(name)


def stop(name, cudasync=False, sync=False):
    if cudasync and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
    if sync:
        MPI.COMM_WORLD.Barrier()
    for tr in __tracer_list__.values():
        tr.stop(name)


def enable():
    for tr in __tracer_list__.values():
        tr.enable()


def disable():
    for tr in __tracer_list__.values():
        tr.disable()


def reset():
    for tr in __tracer_list__.values():
        tr.reset()


def save(log_name):
    _, rank = get_comm_size_and_rank()
    if has("GPTLTracer"):
        import gptl4py as gp

        gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))

    if has("NVMLTracer"):
        nv = __tracer_list__["NVMLTracer"]
        nv.pr_file(os.path.join("logs", log_name, "nvml_energy.p%d" % rank))

    if has("ROCMTracer"):
        rc = __tracer_list__["ROCMTracer"]
        rc.pr_file(os.path.join("logs", log_name, "rocm_energy.p%d" % rank))


def profile(x_or_func=None, *decorator_args, **decorator_kws):
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kws):
            if "x_or_func" not in locals() or callable(x_or_func) or x_or_func is None:
                x = func.__name__
            else:
                x = x_or_func
            start(x)
            out = func(*args, **kws)
            stop(x)
            return out

        return wrapper

    return _decorator(x_or_func) if callable(x_or_func) else _decorator


@contextmanager
def timer(x):
    start(x)
    yield
    stop(x)
