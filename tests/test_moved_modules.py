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

import importlib
import importlib.util
import pickle
import sys

import pytest

import hydragnn
from hydragnn._compat import MOVED_MODULES, new_module_name

# Modules whose import needs optional dependencies are checked by name only.
_IMPORTABLE = [
    "hydragnn.utils.datasets",
    "hydragnn.utils.datasets.pickledataset",
    "hydragnn.utils.distributed",
    "hydragnn.utils.input_config_parsing",
    "hydragnn.utils.lsms",
]


@pytest.mark.parametrize("old", sorted(MOVED_MODULES))
def test_new_module_exists(old):
    assert importlib.util.find_spec(new_module_name(old)) is not None


@pytest.mark.parametrize("old", _IMPORTABLE)
def test_old_path_aliases_new_module(old):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match="has moved to"):
        module = importlib.import_module(old)
    assert module is importlib.import_module(new_module_name(old))


def test_most_specific_prefix_wins():
    assert (
        new_module_name("hydragnn.utils.model.uma._vendored.fairchem.core")
        == "hydragnn.third_party.uma.fairchem.core"
    )
    assert new_module_name("hydragnn.utils.model.model") is None


def test_attribute_access():
    assert hydragnn.utils.distributed is hydragnn.distributed
    assert hydragnn.utils.datasets is hydragnn.data
    assert hydragnn.utils.input_config_parsing is hydragnn.config


def test_old_pickle_loads():
    from hydragnn.data.pickledataset import SimplePickleDataset

    payload = pickle.dumps(SimplePickleDataset, protocol=0).replace(
        b"hydragnn.data.pickledataset", b"hydragnn.utils.datasets.pickledataset"
    )
    assert b"hydragnn.utils.datasets.pickledataset" in payload
    assert pickle.loads(payload) is SimplePickleDataset
