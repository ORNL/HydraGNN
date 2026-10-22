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
from . import _compat

_compat.install()

from . import config, data, distributed
from . import domain_losses, preprocess, models, train, postprocess, utils
from ._version import __version__

# Attribute access (``hydragnn.utils.distributed.setup_ddp``) bypasses the finder.
utils.datasets = data
utils.distributed = distributed
utils.input_config_parsing = config
