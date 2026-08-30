##############################################################################
# Copyright (c) 2022, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import copy
import os, json
import pytest

import hydragnn
from hydragnn.utils.input_config_parsing import (
    get_log_name_config,
    sanitize_filename_component,
)


def test_input_config_parsing_is_exposed_through_utils():
    assert hydragnn.utils.input_config_parsing.update_config is not None


@pytest.mark.parametrize("config_file", ["lsms/lsms.json"])
@pytest.mark.mpi_skip()
def test_config(config_file):

    config_file = os.path.join("examples", config_file)
    with open(config_file, "r") as f:
        config = json.load(f)

    expected = {
        "Dataset": [
            "name",
            "path",
            "format",
        ],
        "NeuralNetwork": ["Architecture", "Training"],
        "Variables": ["inputs", "outputs"],
    }

    for category_name, required_keys in expected.items():
        assert category_name in config, "Missing required input category"
        for key in required_keys:
            assert (
                key in config[category_name]
            ), f"Missing required input {category_name}.{key}"


def test_log_name_sanitizes_named_variables():
    with open(os.path.join("tests", "inputs", "ci.json"), "r") as f:
        config = json.load(f)

    config = copy.deepcopy(config)
    config["Variables"]["inputs"] = [
        {"name": "../atomic species/unsafe", "level": "node", "dim": 1},
        {"name": "a" * 100, "level": "graph", "dim": 1},
    ]
    log_name = get_log_name_config(config)
    variable_part = log_name.split("-node_ft-", 1)[1].split("-task_weights-", 1)[0]

    assert "/" not in variable_part
    assert ".." not in variable_part
    assert len(variable_part.split("-", 2)[1]) <= 48
    assert not variable_part.endswith("-")


def test_filename_component_sanitizes_configured_variable_name():
    original = "../atomic species\\unsafe"
    sanitized = sanitize_filename_component(original)

    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert ".." not in sanitized
    assert sanitized != original
    assert len(sanitized) <= 48
    assert sanitized == sanitize_filename_component(original)
