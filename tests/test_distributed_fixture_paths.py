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

from tests.deterministic_graph_data import synchronize_dataset_paths


class _NonRootComm:
    def __init__(self, root_paths):
        self.root_paths = root_paths
        self.broadcast_value = object()

    def Get_rank(self):
        return 1

    def bcast(self, value, root):
        self.broadcast_value = value
        assert root == 0
        return self.root_paths


def test_nonroot_uses_rank_zero_fixture_paths():
    root_paths = {"total": "serialized_dataset/unit_test_multihead.pkl"}
    config = {"Dataset": {"path": {"total": "dataset/unit_test_multihead"}}}
    comm = _NonRootComm(root_paths)

    synchronize_dataset_paths(config, comm)

    assert comm.broadcast_value is None
    assert config["Dataset"]["path"] == root_paths
