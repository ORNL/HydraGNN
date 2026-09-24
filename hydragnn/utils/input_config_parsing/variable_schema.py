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
GraphType = Literal["homogeneous", "heterogeneous"]
_LEVELS = frozenset(("node", "edge", "graph"))
_ROLES = frozenset(("feature", "position"))
_GRAPH_TYPES = frozenset(("homogeneous", "heterogeneous"))
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
    # Disambiguates which heterogeneous node type this variable belongs to;
    # unused (and must be omitted) for homogeneous graphs.
    node_type: str | None = None


@dataclass(frozen=True)
class VariableSchema:
    """Named model inputs and prediction targets."""

    graph_type: GraphType
    node_types: tuple[str, ...]
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
        extra = set(raw) - {"name", "level", "dim", "role", "encoding", "node_type"}
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
        node_type = raw.get("node_type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        if level not in _LEVELS:
            raise ValueError(f"{path}.level must be one of {sorted(_LEVELS)}")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{path}.dim must be a positive integer")
        if role not in _ROLES:
            raise ValueError(f"{path}.role must be one of {sorted(_ROLES)}")
        if node_type is not None:
            if level != "node":
                raise ValueError(f"{path}.node_type is valid only when level is 'node'")
            if not isinstance(node_type, str) or not node_type.strip():
                raise ValueError(f"{path}.node_type must be a non-empty string")
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
            VariableSpec(
                name=name,
                level=level,
                dim=dim,
                role=role,
                encoding=encoding,
                node_type=node_type,
            )
        )
    return tuple(parsed)


def parse_variable_schema(raw_variables: dict) -> VariableSchema:
    """Parse and validate the top-level ``Variables`` JSON section."""
    if not isinstance(raw_variables, dict):
        raise TypeError("Variables must be a JSON object")
    extra = set(raw_variables) - {"graph_type", "node_types", "inputs", "outputs"}
    if extra:
        raise ValueError("Variables has unknown keys: " + ", ".join(sorted(extra)))
    graph_type = raw_variables.get("graph_type")
    if graph_type not in _GRAPH_TYPES:
        raise ValueError(f"Variables.graph_type must be one of {sorted(_GRAPH_TYPES)}")
    raw_node_types = raw_variables.get("node_types")
    if graph_type == "homogeneous":
        if raw_node_types is not None:
            raise ValueError(
                "Variables.node_types is valid only for heterogeneous graphs"
            )
        node_types = ()
    else:
        if not isinstance(raw_node_types, list) or not raw_node_types:
            raise ValueError(
                "Heterogeneous Variables.node_types must be a non-empty array"
            )
        if any(
            not isinstance(value, str) or not value.strip() for value in raw_node_types
        ):
            raise ValueError("Variables.node_types entries must be non-empty strings")
        if len(set(raw_node_types)) != len(raw_node_types):
            raise ValueError("Variables.node_types entries must be unique")
        node_types = tuple(raw_node_types)
    schema = VariableSchema(
        graph_type=graph_type,
        node_types=node_types,
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
    node_specs = [
        (group, spec)
        for group, specs in (("inputs", schema.inputs), ("outputs", schema.outputs))
        for spec in specs
        if spec.level == "node"
    ]
    if graph_type == "homogeneous":
        qualified = [
            f"Variables.{group}.{spec.name}"
            for group, spec in node_specs
            if spec.node_type is not None
        ]
        if qualified:
            raise ValueError(
                "Homogeneous graph variables must omit node_type: "
                + ", ".join(qualified)
            )
    else:
        unqualified = [
            f"Variables.{group}.{spec.name}"
            for group, spec in node_specs
            if spec.node_type is None
        ]
        if unqualified:
            raise ValueError(
                "Heterogeneous graph node variables must declare node_type: "
                + ", ".join(unqualified)
            )
        unknown = [
            f"Variables.{group}.{spec.name}={spec.node_type}"
            for group, spec in node_specs
            if spec.node_type not in node_types
        ]
        if unknown:
            raise ValueError(
                "Variable node_type values are absent from Variables.node_types: "
                + ", ".join(unknown)
            )
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


def node_type_feature_dims(schema: VariableSchema, group: str = "inputs") -> dict:
    """Sum feature dimensions for node-type-qualified variables."""
    specs = getattr(schema, group)
    dims: dict = {}
    for spec in specs:
        if spec.level != "node" or spec.role != "feature":
            continue
        if spec.node_type is None:
            raise ValueError(
                f"Variables.{group}.{spec.name} must declare node_type before "
                "computing heterogeneous feature dimensions"
            )
        dims[spec.node_type] = dims.get(spec.node_type, 0) + spec.dim
    return dims


def validate_node_type_contract(
    schema: VariableSchema, node_types: tuple[str, ...] | None
) -> None:
    """Validate the declared graph structure against a loaded dataset."""
    if node_types is None:
        if schema.graph_type != "homogeneous":
            raise ValueError(
                "Variables.graph_type is 'heterogeneous', but the dataset is homogeneous"
            )
        return

    if schema.graph_type != "heterogeneous":
        raise ValueError(
            "Variables.graph_type is 'homogeneous', but the dataset is heterogeneous"
        )
    if set(schema.node_types) != set(node_types):
        raise ValueError(
            f"Variables.node_types {sorted(schema.node_types)} do not match dataset "
            f"node types {sorted(node_types)}"
        )
