import time
import torch
from utils.distributed import get_comm_size_and_rank
from utils.print_utils import print_distributed


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    timers_local = dict()
    timers_min = dict()
    timers_max = dict()
    timers_avg = dict()
    number_calls = dict()

    def __init__(self, name: str):
        self.start_time = None
        self.elapsed_time = None
        self.tmin = None
        self.tmax = None
        self.tavg = None
        self.running = False
        self.calls = 0
        self.name = name

        self.world_size, self.world_rank = get_comm_size_and_rank()

        self.timers_local.setdefault(name, 0.0)
        self.timers_min.setdefault(name, 0.0)
        self.timers_max.setdefault(name, 0.0)
        self.timers_avg.setdefault(name, 0.0)
        self.number_calls.setdefault(name, 0)

    def start(self):
        if self.start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self.running = True
        self.calls = self.calls + 1
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        self.elapsed_time = time.perf_counter() - self.start_time
        self.start_time = None

        self.tmin = torch.Tensor([self.elapsed_time])
        self.tmax = torch.Tensor([self.elapsed_time])
        self.tavg = torch.Tensor([self.elapsed_time])

        world_size, world_rank = get_comm_size_and_rank()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.tmin, op=torch.distributed.ReduceOp.MIN)
            self.tmin = self.tmin.item()
            torch.distributed.all_reduce(self.tmax, op=torch.distributed.ReduceOp.MAX)
            self.tmax = self.tmax.item()
            torch.distributed.all_reduce(self.tavg, op=torch.distributed.ReduceOp.SUM)
        self.tavg = self.tavg.item() / world_size

        if self.name:
            self.timers_local[self.name] += self.elapsed_time
            self.timers_min[self.name] += self.tmin
            self.timers_max[self.name] += self.tmax
            self.timers_avg[self.name] += self.tavg
            self.number_calls[self.name] += self.calls

        self.running = False

    def reset(self):
        self.start_time = None
        self.elapsed_time = None
        self.running = False
        self.tmin = None
        self.tmax = None
        self.tavg = None
        self.calls = 0


def print_timers(verbosity):

    world_size, world_rank = get_comm_size_and_rank()

    # With proper lever of verbosity >=1, the local timers will have different values per process
    print_distributed(
        verbosity, f"Process {world_rank} - Local timers: ", Timer.timers_local
    )

    # The statistics are the result of global collective operations, so we only print them once
    if verbosity >= 1:
        print_distributed(1, "Minimum timers: ")
        print_distributed(1, Timer.timers_min)
        print_distributed(1, "Maximum timers: ")
        print_distributed(1, Timer.timers_max)
        print_distributed(1, "Average timers: ")
        print_distributed(1, Timer.timers_avg)
        print_distributed(1, "Number of calls to timers: ")
        print_distributed(1, Timer.number_calls)
