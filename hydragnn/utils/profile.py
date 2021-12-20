import torch
import contextlib
from unittest.mock import MagicMock
from torch.profiler import profile, record_function, ProfilerActivity


class Profiler(torch.profiler.profile):
    def __init__(self, prefix, enable=False, target_epoch=0):
        self.prefix = prefix
        self.enable = enable
        self.target_epoch = target_epoch
        self.current_epoch = -1

        super(Profiler, self).__init__(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=self.trace_handler,
            record_shapes=True,
            with_stack=True,
        )

        self.null_profiler = contextlib.nullcontext(MagicMock(name="step"))

    def setup(self, config):
        def tryget(key, default):
            val = default
            try:
                val = config[key]
            except:
                pass
            return val

        self.enable = tryget("enable", 0) == 1
        self.target_epoch = tryget("target_epoch", 0)

    def trace_handler(self, p):
        print(
            "Total number of profiled events: %d at epoch %d"
            % (len(p.events()), self.target_epoch)
        )
        torch.profiler.tensorboard_trace_handler(self.prefix)(p)

    def set_current_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def __enter__(self):
        if self.enable and (self.target_epoch == self.current_epoch):
            return super(Profiler, self).__enter__()
        else:
            return self.null_profiler.__enter__()

    def __exit__(self, type, value, traceback):
        if self.enable and (self.target_epoch == self.current_epoch):
            super(Profiler, self).__exit__(type, value, traceback)

    def step(self):
        if self.enable:
            super(Profiler, self).step()

    def reset(self):
        super(Profiler, self).step_num = 0
