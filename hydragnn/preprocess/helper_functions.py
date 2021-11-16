##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import numpy as np

import torch
import torch.distributed as dist


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))
