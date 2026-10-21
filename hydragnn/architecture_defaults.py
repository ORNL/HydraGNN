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

"""Flat ``Architecture`` defaults derived from :mod:`hydragnn.model_settings`."""

from hydragnn.model_settings import AllScAIPSettings, UMASettings

MODEL_SPECIFIC_ARCHITECTURE_DEFAULTS = {
    **AllScAIPSettings().as_kwargs(),
    **UMASettings().as_kwargs(),
}
