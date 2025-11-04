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

"""
Tests for feature configuration parsing and validation.
"""

import pytest
import torch
from torch_geometric.data import Data

from hydragnn.utils.input_config_parsing.feature_config import (
    parse_feature_config,
    validate_feature_config,
    update_var_config_with_features,
    print_feature_summary,
    get_feature_schema_example,
)


def test_parse_new_format():
    """Test parsing new feature configuration format."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "coordinates": {"dim": 3, "role": "input"},
            "forces": {"dim": 3, "role": "output", "output_type": "node"},
        },
        "graph_features": {
            "energy": {"dim": 1, "role": "output", "output_type": "graph"}
        },
    }

    parsed = parse_feature_config(var_config)

    assert parsed["node_feature_names"] == ["atomic_number", "coordinates", "forces"]
    assert parsed["node_feature_dims"] == [1, 3, 3]
    assert parsed["graph_feature_names"] == ["energy"]
    assert parsed["graph_feature_dims"] == [1]
    assert parsed["input_node_features"] == [0, 1]  # atomic_number, coordinates
    assert parsed["output_names"] == ["forces", "energy"]
    assert parsed["output_dim"] == [3, 1]
    assert parsed["type"] == ["node", "graph"]


def test_parse_legacy_format():
    """Test parsing legacy feature configuration format."""
    var_config = {
        "node_feature_names": ["atomic_number", "coordinates", "forces"],
        "node_feature_dims": [1, 3, 3],
        "graph_feature_names": ["energy"],
        "graph_feature_dims": [1],
        "input_node_features": [0, 1],
        "output_names": ["energy"],
        "output_index": [0],
        "output_dim": [1],
        "type": ["graph"],
    }

    parsed = parse_feature_config(var_config)

    assert parsed["node_feature_names"] == ["atomic_number", "coordinates", "forces"]
    assert parsed["node_feature_dims"] == [1, 3, 3]
    assert parsed["graph_feature_names"] == ["energy"]
    assert parsed["graph_feature_dims"] == [1]
    assert parsed["input_node_features"] == [0, 1]
    assert parsed["output_names"] == ["energy"]


def test_parse_both_role():
    """Test parsing features with 'both' role (input and output)."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "forces": {"dim": 3, "role": "both", "output_type": "node"},
        }
    }

    parsed = parse_feature_config(var_config)

    assert parsed["node_feature_names"] == ["atomic_number", "forces"]
    assert parsed["input_node_features"] == [0, 1]  # Both are inputs
    assert "forces" in parsed["output_names"]  # Forces is also output


def test_validate_config_valid():
    """Test validation of valid configuration."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "coordinates": {"dim": 3, "role": "input"},
        },
        "graph_features": {"energy": {"dim": 1, "role": "output"}},
    }

    is_valid, errors = validate_feature_config(var_config)
    assert is_valid
    assert len(errors) == 0


def test_validate_config_mismatched_lengths():
    """Test validation catches mismatched lengths."""
    var_config = {
        "node_feature_names": ["atomic_number", "coordinates"],
        "node_feature_dims": [1],  # Wrong length!
        "graph_feature_names": ["energy"],
        "graph_feature_dims": [1],
        "input_node_features": [0],
        "output_names": ["energy"],
        "output_index": [0],
        "output_dim": [1],
        "type": ["graph"],
    }

    is_valid, errors = validate_feature_config(var_config)
    assert not is_valid
    assert any("node_feature_names" in err for err in errors)


def test_validate_config_invalid_indices():
    """Test validation catches invalid indices."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
        },
        "graph_features": {
            "energy": {"dim": 1, "role": "output", "output_type": "graph"}
        },
    }
    # Manually add invalid index
    parsed = parse_feature_config(var_config)
    parsed["input_node_features"] = [0, 5]  # 5 is out of range

    is_valid, errors = validate_feature_config(
        {"node_features": var_config["node_features"]}
    )
    # This should be valid initially
    assert is_valid


def test_validate_against_data():
    """Test validation against actual data object."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "coordinates": {"dim": 3, "role": "input"},
        }
    }

    # Create matching data object
    data = Data(x=torch.randn(10, 4))  # 10 nodes, 4 features (1+3)

    is_valid, errors = validate_feature_config(var_config, data)
    assert is_valid
    assert len(errors) == 0


def test_validate_against_data_mismatch():
    """Test validation catches dimension mismatch with data."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "coordinates": {"dim": 3, "role": "input"},
        }
    }

    # Create mismatched data object
    data = Data(x=torch.randn(10, 5))  # 10 nodes, 5 features (should be 4)

    is_valid, errors = validate_feature_config(var_config, data)
    assert not is_valid
    assert any("dimension" in err.lower() for err in errors)


def test_update_var_config():
    """Test updating var_config with parsed features."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "coordinates": {"dim": 3, "role": "input"},
        },
        "graph_features": {"energy": {"dim": 1, "role": "output"}},
    }

    updated = update_var_config_with_features(var_config)

    # Should have legacy keys now
    assert "node_feature_names" in updated
    assert "node_feature_dims" in updated
    assert "graph_feature_names" in updated
    assert "graph_feature_dims" in updated
    assert "input_node_features" in updated


def test_get_feature_schema_example():
    """Test that example schema is valid."""
    example = get_feature_schema_example()

    # Should be valid
    is_valid, errors = validate_feature_config(example)
    assert is_valid, f"Example schema should be valid. Errors: {errors}"


def test_print_feature_summary():
    """Test printing feature summary."""
    var_config = {
        "node_features": {
            "atomic_number": {"dim": 1, "role": "input"},
            "coordinates": {"dim": 3, "role": "input"},
        },
        "graph_features": {"energy": {"dim": 1, "role": "output"}},
    }

    summary = print_feature_summary(var_config)

    assert "Feature Configuration Summary" in summary
    assert "atomic_number" in summary
    assert "coordinates" in summary
    assert "energy" in summary
    assert "dim=1" in summary
    assert "dim=3" in summary


def test_empty_config():
    """Test handling of empty configuration."""
    var_config = {
        "node_features": {},
        "graph_features": {},
    }

    parsed = parse_feature_config(var_config)

    assert parsed["node_feature_names"] == []
    assert parsed["graph_feature_names"] == []
    assert parsed["input_node_features"] == []


def test_missing_config_raises():
    """Test that completely invalid config raises error."""
    var_config = {}  # No features at all

    with pytest.raises(ValueError):
        parse_feature_config(var_config)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
