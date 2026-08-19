##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging


def get_flops_profile(model, input_data, verbose: bool = False):
    try:
        from flops_profiler.profiler import FlopsProfiler
    except Exception as e:
        logging.error(
            "To use this feature you need to install the flops profiler, pip install pip install flops-profiler"
        )
        raise e
    prof = FlopsProfiler(model)
    prof.start_profile()
    model(input_data)
    prof.stop_profile()
    flops = prof.get_total_flops()
    if verbose:
        prof.print_model_profile(profile_step=1)
    return flops
