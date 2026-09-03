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

import os
import subprocess
import sys

import pytest


@pytest.mark.mpi_skip()
def test_atomicdescriptors(tmp_path):
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "hydragnn/utils/descriptors_and_embeddings/atomicdescriptors.py",
    )
    subprocess.run([sys.executable, file_path], cwd=tmp_path, check=True)
