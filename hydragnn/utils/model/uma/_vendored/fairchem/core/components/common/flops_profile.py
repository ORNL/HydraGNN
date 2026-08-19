##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# Copyright (c) Meta Platforms, Inc. and affiliates.                         #
#                                                                            #
# Portions derived from FAIR-Chem are distributed under the MIT License;     #
# HydraGNN modifications are distributed under the BSD 3-clause license.     #
# Original upstream copyright and license notices are preserved below.       #
#                                                                            #
# SPDX-License-Identifier: MIT AND BSD-3-Clause                              #
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
