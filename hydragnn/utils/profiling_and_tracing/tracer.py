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
from hydragnn.utils.distributed import get_comm_size_and_rank, get_local_rank
import os


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
            self.hist = dict()
            self.last = dict()
            self.enabled = True

        def start(self, name):
            if not self.enabled:
                return
            gp.start(name)

        def stop(self, name):
            if not self.enabled:
                return
            gp.stop(name)

            count, wallclock = gp.query_raw(name)
            if name not in self.hist:
                self.hist[name] = list()
                self.last[name] = 0.0
            self.hist[name].append((count, wallclock - self.last[name]))
            self.last[name] = wallclock

        def enable(self):
            self.enabled = True
            gp.enable()

        def disable(self):
            self.enabled = False
            gp.disable()

        def reset(self):
            gp.reset()
            self.hist = dict()
            self.last = dict()


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
            if os.getenv("CUDA_VISIBLE_DEVICES"):
                device_list = [
                    int(x) for x in os.getenv("CUDA_VISIBLE_DEVICES").split(",")
                ]
            else:
                device_list = [0]

            local_rank = get_local_rank()
            self.device = (
                device_list[local_rank] if len(device_list) > 1 else device_list[0]
            )

            self.d_handle = nvmlDeviceGetHandleByIndex(self.device)
            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)
            self.enabled = True
            print(f"NVMLTracer initalized: rank={self.rank}, device={self.device}")

        ## nvmlDeviceGetTotalEnergyConsumption returns in mJ. Use uJ
        def start(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = (
                nvmlDeviceGetTotalEnergyConsumption(self.d_handle) * 1_000
            )

        def stop(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = (
                nvmlDeviceGetTotalEnergyConsumption(self.d_handle) * 1_000
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

    class ROCMTracer:
        def __init__(self, **kwargs):
            rocm_smi.initializeRsmi()
            self.rocmsmi = initRsmiBindings()
            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)
            self.enabled = True
            self.rank = MPI.COMM_WORLD.Get_rank()

            if os.getenv("ROCR_VISIBLE_DEVICES"):
                device_list = [
                    int(x) for x in os.getenv("ROCR_VISIBLE_DEVICES").split(",")
                ]
            else:
                device_list = rocm_smi.listDevices()

            local_rank = get_local_rank()
            self.device = (
                device_list[local_rank] if len(device_list) > 1 else device_list[0]
            )
            print(f"ROCMTracer initalized: rank={self.rank}, device={self.device}")

        def get_energy(self):
            """Accumulated Energy (uJ)"""
            try:
                power = c_uint64()
                timestamp = c_uint64()
                counter_resolution = c_float()
                ret = self.rocmsmi.rsmi_dev_energy_count_get(
                    self.device,
                    byref(power),
                    byref(counter_resolution),
                    byref(timestamp),
                )
                return power.value * counter_resolution.value
            except:
                return np.nan

        def start(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = self.get_energy()

        def stop(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = self.get_energy() - self.energyCounters[name]
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

    class XPUTracer:
        def __init__(self, **kwargs):
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.device = torch.xpu.current_device()
            group = self.device // 2
            counter_id = group * 3 + self.device % 2 + 1
            counter = f"/sys/class/hwmon/hwmon{counter_id}/energy1_input"
            self.f = open(counter, "r")

            self.energyCounters = dict()
            self.energyTracer = defaultdict(list)
            self.enabled = True

            print(f"XPUTracer initalized: rank={self.rank}, device={self.device}")

        def get_energy_read(self):
            ## Cumulative energy used (uJ)
            self.f.seek(0)
            energy_uj = float(self.f.read().strip())
            return energy_uj

        def start(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = self.get_energy_read()

        def stop(self, name):
            if not self.enabled:
                return
            self.energyCounters[name] = (
                self.get_energy_read() - self.energyCounters[name]
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


__tracer_list__ = dict()


def has(name):
    return name in __tracer_list__


def initialize(
    trlist=["GPTLTracer", "SCOREPTracer", "NVMLTracer", "ROCMTracer", "XPUTracer"],
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
    elif cudasync and hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
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
    elif cudasync and hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
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

        tx = __tracer_list__["GPTLTracer"]

        gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))

        with open(os.path.join("logs", log_name, f"gp_full.p{rank}"), "w") as f:
            f.write("rank,label,count,wallclock\n")
            for key, value_list in tx.hist.items():
                for count, wallclock in value_list:
                    f.write(f"{rank},{key},{count},{wallclock}\n")

    if has("NVMLTracer"):
        tx = __tracer_list__["NVMLTracer"]
        tx.pr_file(os.path.join("logs", log_name, "nvml_energy.p%d" % rank))

    if has("ROCMTracer"):
        tx = __tracer_list__["ROCMTracer"]
        tx.pr_file(os.path.join("logs", log_name, "rocm_energy.p%d" % rank))

    if has("XPUTracer"):
        tx = __tracer_list__["XPUTracer"]
        tx.pr_file(os.path.join("logs", log_name, "xpu_energy.p%d" % rank))


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
