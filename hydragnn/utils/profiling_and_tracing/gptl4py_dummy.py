##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
""" 
This is a dummy package for gptl4py (https://github.com/jychoi-hpc/gptl4py).
When gptl4py is not available, use as follows:
```
try:
    import gptl4py as gp
except ImportError:
    import gptl4py_dummy as gp
```
"""

from __future__ import absolute_import
from functools import wraps
from contextlib import contextmanager


def initialize():
    pass


def finalize():
    pass


def start(name):
    pass


def stop(name):
    pass


def pr_file(filename):
    pass


def pr_summary_file(filename):
    pass


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
def nvtx_timer(x):
    start(x)
    yield
    stop(x)
