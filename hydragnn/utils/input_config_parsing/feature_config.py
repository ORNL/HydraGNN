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
Feature configuration parsing and validation utilities.

This module provides utilities to parse and validate feature configurations
from the Variables_of_interest section in config files. It supports both:
1. Legacy format: Using separate lists for feature names and dimensions
2. New format: Using structured dictionaries with explicit feature roles

Example new format:
{
    "node_features": {
        "atomic_number": {"dim": 1, "role": "input"},
        "cartesian_coordinates": {"dim": 3, "role": "input"},
        "forces": {"dim": 3, "role": "output", "output_type": "node"}
    },
    "graph_features": {
        "energy": {"dim": 1, "role": "output", "output_type": "graph"}
    }
}

Example legacy format:
{
    "node_feature_names": ["atomic_number", "cartesian_coordinates", "forces"],
    "node_feature_dims": [1, 3, 3],
    "graph_feature_names": ["energy"],
    "graph_feature_dims": [1],
    "input_node_features": [0, 1],
    "output_names": ["energy"],
    "output_index": [0],
    "output_dim": [1],
    "type": ["graph"]
}
"""

from typing import Dict, List, Tuple, Optional
import warnings


def parse_feature_config(var_config: dict) -> dict:
    """
    Parse Variables_of_interest config and extract feature information.

    Supports both legacy and new formats:
    - Legacy: Uses graph_feature_names, node_feature_names (if present)
    - New: Uses node_features, graph_features dicts

    Args:
        var_config: Variables_of_interest section from config

    Returns:
        Standardized dict with:
        - node_feature_names: List[str]
        - node_feature_dims: List[int]
        - graph_feature_names: List[str]
        - graph_feature_dims: List[int]
        - input_node_features: List[int] (indices)
        - output_names: List[str]
        - output_index: List[int]
        - output_dim: List[int]
        - type: List[str] ('graph' or 'node')

    Raises:
        ValueError: If config format is invalid
    """
    result = {}

    # Check if using new format
    if "node_features" in var_config or "graph_features" in var_config:
        result = _parse_new_format(var_config)
    # Check if using legacy format
    elif "node_feature_names" in var_config or "graph_feature_names" in var_config:
        result = _parse_legacy_format(var_config)
    else:
        raise ValueError(
            "Config must contain either 'node_features'/'graph_features' (new format) "
            "or 'node_feature_names'/'graph_feature_names' (legacy format)"
        )

    return result


def _parse_new_format(var_config: dict) -> dict:
    """
    Parse new feature configuration format.

    The new format uses dictionaries with explicit feature roles:
    - role: 'input', 'output', or 'both'
    - dim: integer dimension
    - output_type: 'graph' or 'node' (for output features)

    Args:
        var_config: Variables_of_interest section

    Returns:
        Standardized feature dict
    """
    result = {
        "node_feature_names": [],
        "node_feature_dims": [],
        "graph_feature_names": [],
        "graph_feature_dims": [],
        "input_node_features": [],
        "output_names": [],
        "output_index": [],
        "output_dim": [],
        "type": [],
    }

    # Parse node features
    if "node_features" in var_config:
        node_idx = 0
        for feat_name, feat_config in var_config["node_features"].items():
            result["node_feature_names"].append(feat_name)
            result["node_feature_dims"].append(feat_config["dim"])

            # Track input features
            role = feat_config.get("role", "input")
            if role in ["input", "both"]:
                result["input_node_features"].append(node_idx)

            # Track output features
            if role in ["output", "both"]:
                result["output_names"].append(feat_name)
                result["output_index"].append(node_idx)
                result["output_dim"].append(feat_config["dim"])
                result["type"].append(feat_config.get("output_type", "node"))

            node_idx += 1

    # Parse graph features
    if "graph_features" in var_config:
        graph_idx = 0
        for feat_name, feat_config in var_config["graph_features"].items():
            result["graph_feature_names"].append(feat_name)
            result["graph_feature_dims"].append(feat_config["dim"])

            # Graph features are typically outputs (unless role explicitly set)
            role = feat_config.get("role", "output")
            if role in ["output", "both"]:
                result["output_names"].append(feat_name)
                result["output_index"].append(graph_idx)
                result["output_dim"].append(feat_config["dim"])
                result["type"].append(feat_config.get("output_type", "graph"))

            graph_idx += 1

    return result


def _parse_legacy_format(var_config: dict) -> dict:
    """
    Parse legacy feature configuration format.

    The legacy format uses separate lists for names, dimensions, and indices.

    Args:
        var_config: Variables_of_interest section

    Returns:
        Standardized feature dict
    """
    result = {
        "node_feature_names": var_config.get("node_feature_names", []),
        "node_feature_dims": var_config.get("node_feature_dims", []),
        "graph_feature_names": var_config.get("graph_feature_names", []),
        "graph_feature_dims": var_config.get("graph_feature_dims", []),
        "input_node_features": var_config.get("input_node_features", []),
        "output_names": var_config.get("output_names", []),
        "output_index": var_config.get("output_index", []),
        "output_dim": var_config.get("output_dim", []),
        "type": var_config.get("type", []),
    }

    return result


def validate_feature_config(
    var_config: dict, data_object=None
) -> Tuple[bool, List[str]]:
    """
    Validate feature configuration for consistency.

    Checks:
    1. Feature names and dimensions lists have matching lengths
    2. Output configuration is consistent
    3. Indices are within valid ranges
    4. If data_object provided, validates against actual data dimensions

    Args:
        var_config: Variables_of_interest section from config
        data_object: Optional PyG Data object to validate against

    Returns:
        (is_valid, list_of_errors)
        - is_valid: True if no errors found
        - list_of_errors: List of error messages (empty if valid)
    """
    errors = []

    # Parse config
    try:
        parsed = parse_feature_config(var_config)
    except Exception as e:
        return False, [f"Failed to parse config: {str(e)}"]

    # Check lengths match
    if len(parsed["node_feature_names"]) != len(parsed["node_feature_dims"]):
        errors.append(
            f"node_feature_names length ({len(parsed['node_feature_names'])}) "
            f"!= node_feature_dims length ({len(parsed['node_feature_dims'])})"
        )

    if len(parsed["graph_feature_names"]) != len(parsed["graph_feature_dims"]):
        errors.append(
            f"graph_feature_names length ({len(parsed['graph_feature_names'])}) "
            f"!= graph_feature_dims length ({len(parsed['graph_feature_dims'])})"
        )

    # Check output configuration consistency
    output_lengths = [
        len(parsed["output_names"]),
        len(parsed["output_index"]),
        len(parsed["output_dim"]),
        len(parsed["type"]),
    ]
    if len(set(output_lengths)) > 1:
        errors.append(
            f"Output config lengths don't match: "
            f"names={output_lengths[0]}, "
            f"index={output_lengths[1]}, "
            f"dim={output_lengths[2]}, "
            f"type={output_lengths[3]}"
        )

    # Check that input_node_features indices are valid
    max_node_idx = len(parsed["node_feature_names"]) - 1
    if max_node_idx >= 0:  # Only check if we have node features
        for idx in parsed["input_node_features"]:
            if idx > max_node_idx:
                errors.append(
                    f"input_node_features contains index {idx}, "
                    f"but only {len(parsed['node_feature_names'])} node features defined"
                )

    # Check that output_index values are valid
    for i, (output_type, output_idx) in enumerate(
        zip(parsed["type"], parsed["output_index"])
    ):
        if output_type == "graph":
            max_idx = len(parsed["graph_feature_names"]) - 1
            if output_idx > max_idx:
                errors.append(
                    f"output_index[{i}]={output_idx} (type='graph'), "
                    f"but only {len(parsed['graph_feature_names'])} graph features defined"
                )
        elif output_type == "node":
            max_idx = len(parsed["node_feature_names"]) - 1
            if output_idx > max_idx:
                errors.append(
                    f"output_index[{i}]={output_idx} (type='node'), "
                    f"but only {len(parsed['node_feature_names'])} node features defined"
                )

    # Validate against data object if provided
    if data_object is not None:
        data_errors = _validate_against_data(parsed, data_object)
        errors.extend(data_errors)

    return len(errors) == 0, errors


def _validate_against_data(parsed: dict, data_object) -> List[str]:
    """
    Validate parsed config against actual data object.

    Args:
        parsed: Parsed feature configuration
        data_object: PyG Data object

    Returns:
        List of validation error messages
    """
    errors = []

    # Check x tensor dimensions
    if hasattr(data_object, "x") and data_object.x is not None:
        expected_x_dim = sum(parsed["node_feature_dims"])
        actual_x_dim = data_object.x.shape[1] if len(data_object.x.shape) > 1 else 1
        if expected_x_dim != actual_x_dim:
            errors.append(
                f"Expected x dimension {expected_x_dim} based on node_feature_dims "
                f"{parsed['node_feature_dims']}, but got {actual_x_dim} in data"
            )

    # Check that features referenced in input_node_features actually exist in data
    if hasattr(data_object, "x") and parsed["input_node_features"]:
        # Verify we can extract the features
        try:
            start_idx = 0
            for feat_idx in parsed["input_node_features"]:
                feat_dim = parsed["node_feature_dims"][feat_idx]
                end_idx = start_idx + feat_dim
                # Just checking we can slice - don't need the actual data
                if hasattr(data_object.x, "shape"):
                    if end_idx > data_object.x.shape[1]:
                        errors.append(
                            f"Feature {feat_idx} ({parsed['node_feature_names'][feat_idx]}) "
                            f"would require columns up to {end_idx}, "
                            f"but x only has {data_object.x.shape[1]} columns"
                        )
                start_idx = end_idx
        except Exception as e:
            errors.append(f"Error validating input features against data: {str(e)}")

    return errors


def update_var_config_with_features(var_config: dict) -> dict:
    """
    Update var_config dict with parsed feature information.

    This ensures backward compatibility by adding legacy keys if they don't exist.
    If using new format (node_features/graph_features), this will populate the
    legacy keys (node_feature_names, etc.) automatically.

    Args:
        var_config: Variables_of_interest section (will be modified in-place)

    Returns:
        Updated var_config (same object, modified in-place)
    """
    parsed = parse_feature_config(var_config)

    # Update with parsed values (ensuring legacy keys exist)
    for key, value in parsed.items():
        if key not in var_config:
            var_config[key] = value

    return var_config


def get_feature_schema_example() -> dict:
    """
    Return an example of the new feature configuration format.

    Returns:
        Example dict showing the new feature configuration format
    """
    return {
        "node_features": {
            "atomic_number": {
                "dim": 1,
                "role": "input",
                "description": "Atomic number (Z)",
            },
            "cartesian_coordinates": {
                "dim": 3,
                "role": "input",
                "description": "3D position in Cartesian coordinates",
            },
            "forces": {
                "dim": 3,
                "role": "output",
                "output_type": "node",
                "description": "Atomic forces",
            },
        },
        "graph_features": {
            "energy": {
                "dim": 1,
                "role": "output",
                "output_type": "graph",
                "description": "Total energy",
            }
        },
    }


def print_feature_summary(var_config: dict) -> str:
    """
    Generate a human-readable summary of feature configuration.

    Args:
        var_config: Variables_of_interest section

    Returns:
        Formatted string summarizing the features
    """
    try:
        parsed = parse_feature_config(var_config)
    except Exception as e:
        return f"Error parsing config: {str(e)}"

    lines = []
    lines.append("=" * 60)
    lines.append("Feature Configuration Summary")
    lines.append("=" * 60)

    # Node features
    if parsed["node_feature_names"]:
        lines.append("\nNode Features:")
        lines.append("-" * 60)
        for i, (name, dim) in enumerate(
            zip(parsed["node_feature_names"], parsed["node_feature_dims"])
        ):
            is_input = i in parsed["input_node_features"]
            role = "input" if is_input else "internal"
            lines.append(f"  [{i}] {name:30s} dim={dim:2d}  role={role}")

    # Graph features
    if parsed["graph_feature_names"]:
        lines.append("\nGraph Features:")
        lines.append("-" * 60)
        for i, (name, dim) in enumerate(
            zip(parsed["graph_feature_names"], parsed["graph_feature_dims"])
        ):
            lines.append(f"  [{i}] {name:30s} dim={dim:2d}")

    # Outputs
    if parsed["output_names"]:
        lines.append("\nOutput Configuration:")
        lines.append("-" * 60)
        for name, idx, dim, otype in zip(
            parsed["output_names"],
            parsed["output_index"],
            parsed["output_dim"],
            parsed["type"],
        ):
            lines.append(f"  {name:30s} index={idx:2d}  dim={dim:2d}  type={otype}")

    lines.append("=" * 60)
    return "\n".join(lines)
