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
"""Strict named input and output variables for PyG ``Data`` objects."""

from dataclasses import dataclass
from typing import Literal

import torch

VariableLevel = Literal["node", "edge", "graph"]
VariableRole = Literal["feature", "position"]
_LEVELS = frozenset(("node", "edge", "graph"))
_ROLES = frozenset(("feature", "position"))
_ENCODING_TYPES = frozenset(("embedding", "one_hot"))
_DERIVED_TENSOR_NAMES = frozenset(
    (
        "x",
        "edge_attr",
        "graph_attr",
        "edge_index",
        "batch",
        "y",
        "y_loc",
        "node_output",
        "edge_output",
        "graph_output",
    )
)


@dataclass(frozen=True)
class InputEncoding:
    """Optional transformation applied to one named input variable."""

    type: Literal["embedding", "one_hot"]
    num_categories: int
    embedding_dim: int | None = None
    min_value: int = 0


@dataclass(frozen=True)
class VariableSpec:
    """The public contract for one tensor attribute on a graph sample."""

    name: str
    level: VariableLevel
    dim: int
    role: VariableRole = "feature"
    encoding: InputEncoding | None = None


@dataclass(frozen=True)
class VariableSchema:
    """Named model inputs and prediction targets."""

    inputs: tuple[VariableSpec, ...]
    outputs: tuple[VariableSpec, ...]


def _parse_group(raw_variables, group: str) -> tuple[VariableSpec, ...]:
    raw_specs = raw_variables.get(group)
    if not isinstance(raw_specs, list):
        raise TypeError(f"Variables.{group} must be a JSON array")

    parsed = []
    for index, raw in enumerate(raw_specs):
        path = f"Variables.{group}[{index}]"
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must be a JSON object")
        extra = set(raw) - {"name", "level", "dim", "role", "encoding"}
        missing = {"name", "level", "dim"} - set(raw)
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"{path} has unknown keys: {', '.join(sorted(extra))}")

        name = raw["name"]
        level = raw["level"]
        dim = raw["dim"]
        role = raw.get("role", "feature")
        raw_encoding = raw.get("encoding")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        if level not in _LEVELS:
            raise ValueError(f"{path}.level must be one of {sorted(_LEVELS)}")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{path}.dim must be a positive integer")
        if role not in _ROLES:
            raise ValueError(f"{path}.role must be one of {sorted(_ROLES)}")
        if role == "position":
            if group != "inputs":
                raise ValueError(f"{path}.role 'position' is valid only for inputs")
            if name != "pos" or level != "node" or dim != 3:
                raise ValueError(
                    f"{path} with role 'position' must have name 'pos', "
                    "level 'node', and dim 3"
                )
        encoding = None
        if raw_encoding is not None:
            if group != "inputs" or level != "node" or role != "feature" or dim != 1:
                raise ValueError(
                    f"{path}.encoding is supported only for scalar node input features"
                )
            if not isinstance(raw_encoding, dict):
                raise TypeError(f"{path}.encoding must be a JSON object")
            encoding_extra = set(raw_encoding) - {
                "type",
                "num_categories",
                "embedding_dim",
                "min_value",
            }
            if encoding_extra:
                raise ValueError(
                    f"{path}.encoding has unknown keys: "
                    + ", ".join(sorted(encoding_extra))
                )
            encoding_type = raw_encoding.get("type")
            num_categories = raw_encoding.get("num_categories")
            embedding_dim = raw_encoding.get("embedding_dim")
            min_value = raw_encoding.get("min_value", 0)
            if encoding_type not in _ENCODING_TYPES:
                raise ValueError(
                    f"{path}.encoding.type must be one of {sorted(_ENCODING_TYPES)}"
                )
            if (
                isinstance(num_categories, bool)
                or not isinstance(num_categories, int)
                or num_categories <= 0
            ):
                raise ValueError(f"{path}.encoding.num_categories must be positive")
            if encoding_type == "embedding":
                if (
                    isinstance(embedding_dim, bool)
                    or not isinstance(embedding_dim, int)
                    or embedding_dim <= 0
                ):
                    raise ValueError(f"{path}.encoding.embedding_dim must be positive")
            elif embedding_dim is not None:
                raise ValueError(
                    f"{path}.encoding.embedding_dim is valid only for embedding"
                )
            if isinstance(min_value, bool) or not isinstance(min_value, int):
                raise ValueError(f"{path}.encoding.min_value must be an integer")
            encoding = InputEncoding(
                type=encoding_type,
                num_categories=num_categories,
                embedding_dim=embedding_dim,
                min_value=min_value,
            )
        if group == "inputs" and name == "pos" and role != "position":
            raise ValueError(f"{path} named 'pos' must declare role 'position'")
        parsed.append(
            VariableSpec(name=name, level=level, dim=dim, role=role, encoding=encoding)
        )
    return tuple(parsed)


def parse_variable_schema(raw_variables: dict) -> VariableSchema:
    """Parse and validate the top-level ``Variables`` JSON section."""
    if not isinstance(raw_variables, dict):
        raise TypeError("Variables must be a JSON object")
    extra = set(raw_variables) - {"inputs", "outputs"}
    if extra:
        raise ValueError("Variables has unknown keys: " + ", ".join(sorted(extra)))
    schema = VariableSchema(
        inputs=_parse_group(raw_variables, "inputs"),
        outputs=_parse_group(raw_variables, "outputs"),
    )
    for group, specs in (("inputs", schema.inputs), ("outputs", schema.outputs)):
        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Variable names within {group} must be unique: "
                + ", ".join(duplicates)
            )
    positions = [spec for spec in schema.inputs if spec.role == "position"]
    if len(positions) > 1:
        raise ValueError("Variables.inputs may contain only one position variable")
    if not any(
        spec.level == "node" and spec.role == "feature" for spec in schema.inputs
    ):
        raise ValueError("Variables.inputs must contain at least one node feature")
    reserved = sorted(
        {
            spec.name
            for spec in (*schema.inputs, *schema.outputs)
            if spec.name in _DERIVED_TENSOR_NAMES
        }
    )
    if reserved:
        raise ValueError(
            "Variable names conflict with HydraGNN derived/internal tensors: "
            + ", ".join(reserved)
        )
    return schema


def get_variable_schema(config: dict) -> VariableSchema:
    """Return the named schema from the top-level configuration."""
    if "Variables" not in config:
        raise ValueError("The top-level Variables section is required")
    return parse_variable_schema(config["Variables"])


def _expected_rows(data, level: VariableLevel) -> int:
    if level == "node":
        if data.num_nodes is None:
            raise ValueError(
                "Cannot validate node variables because num_nodes is unknown"
            )
        return int(data.num_nodes)
    if level == "edge":
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError(
                "Cannot validate edge variables because edge_index is missing"
            )
        if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, E); got {tuple(data.edge_index.shape)}"
            )
        return int(data.edge_index.shape[1])
    return 1


def validate_variable(data, spec: VariableSpec) -> torch.Tensor:
    """Return a named tensor after checking its exact per-sample shape."""
    if not hasattr(data, spec.name):
        raise ValueError(
            f"Data is missing configured {spec.level} attribute '{spec.name}'"
        )
    value = getattr(data, spec.name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Data.{spec.name} must be a torch.Tensor")
    expected = (_expected_rows(data, spec.level), spec.dim)
    valid_shape = value.ndim == 2 and tuple(value.shape) == expected
    if spec.encoding is not None:
        valid_shape = valid_shape or tuple(value.shape) == (expected[0],)
    if not valid_shape:
        raise ValueError(
            f"Data.{spec.name} must have shape {expected} for a {spec.level} "
            f"variable; got {tuple(value.shape)}"
        )
    return value


def prepare_data_from_schema(data, schema: VariableSchema):
    """Validate named attributes and compile them for existing model internals.

    Named tensors remain on ``data``. Feature inputs at node, edge, and graph
    level are concatenated into ``x``, ``edge_attr``, and ``graph_attr``
    respectively. Geometric positions are validated but remain exclusively in
    ``data.pos`` and are never concatenated into ``data.x``.
    Outputs of each level are concatenated along dimension 1 into
    ``node_output``, ``edge_output``, or ``graph_output``. They are also
    flattened into the internal ``y``/``y_loc`` representation while
    preserving one output-head boundary per configured attribute.
    """
    by_level = {level: [] for level in _LEVELS}
    for spec in schema.inputs:
        value = validate_variable(data, spec)
        if spec.role == "feature":
            if spec.encoding is not None:
                value = value.reshape(-1, 1).float()
            by_level[spec.level].append(value)

    if by_level["node"]:
        data.x = torch.cat(by_level["node"], dim=-1)
    else:
        data.x = torch.empty((data.num_nodes, 0), device=data.edge_index.device)
    if by_level["edge"]:
        data.edge_attr = torch.cat(by_level["edge"], dim=-1)
    elif "edge_attr" in data:
        del data["edge_attr"]
    if by_level["graph"]:
        data.graph_attr = torch.cat(by_level["graph"], dim=-1)
    elif "graph_attr" in data:
        del data["graph_attr"]

    output_by_level = {level: [] for level in _LEVELS}
    outputs = []
    locations = [0]
    for spec in schema.outputs:
        value = validate_variable(data, spec)
        output_by_level[spec.level].append(value)
        flattened = value.reshape(-1, 1)
        outputs.append(flattened)
        locations.append(locations[-1] + flattened.numel())
    if outputs:
        data.y = torch.cat(outputs, dim=0)
        data.y_loc = torch.tensor([locations], dtype=torch.int64, device=data.y.device)
    else:
        for name in ("y", "y_loc"):
            if name in data:
                del data[name]
    for level, values in output_by_level.items():
        name = f"{level}_output"
        if values:
            setattr(data, name, torch.cat(values, dim=1))
        elif name in data:
            del data[name]
    return data


def schema_dimensions(schema: VariableSchema, level: VariableLevel, group: str) -> int:
    """Return the concatenated feature dimension for a level and group."""
    specs = getattr(schema, group)
    return sum(
        spec.dim for spec in specs if spec.level == level and spec.role == "feature"
    )


def encoded_schema_dimensions(
    schema: VariableSchema, level: VariableLevel, group: str
) -> int:
    """Return feature width after applying configured categorical encoders."""
    specs = getattr(schema, group)
    total = 0
    for spec in specs:
        if spec.level != level or spec.role != "feature":
            continue
        if spec.encoding is None:
            total += spec.dim
        elif spec.encoding.type == "embedding":
            total += spec.encoding.embedding_dim
        else:
            total += spec.encoding.num_categories
    return total
