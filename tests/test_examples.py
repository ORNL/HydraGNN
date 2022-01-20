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

import os
import pytest

import subprocess


@pytest.mark.parametrize("example", ["qm9", "md17"])
@pytest.mark.mpi_skip()
def pytest_examples(example):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", example)
    file_path = os.path.join(path, example + ".py")
    return_code = subprocess.call(["python", file_path])

    # Check the file ran without error.
    assert return_code == 0
