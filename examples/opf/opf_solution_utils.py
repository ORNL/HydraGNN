"""Shared utilities for OPF heterogeneous solution workflows."""

import logging
import torch


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def ensure_node_y_loc(data):
    if not hasattr(data, "y") or data.y is None:
        raise RuntimeError("Missing node targets (data.y) for OPF sample.")
    if data.y.dim() == 1:
        data.y = data.y.unsqueeze(-1)
    num_nodes = int(data.y.shape[0])
    target_dim = int(data.y.shape[1])
    data.y_num_nodes = torch.tensor(
        [num_nodes], dtype=torch.int64, device=data.y.device
    )
    data.y_loc = torch.tensor(
        [[0, num_nodes * target_dim]],
        dtype=torch.int64,
        device=data.y.device,
    )


def resolve_node_target_type(data, requested: str) -> str:
    if hasattr(data, "node_types"):
        if requested in data.node_types:
            return requested
        if hasattr(data, "_node_type_names") and requested in data._node_type_names:
            idx = data._node_type_names.index(requested)
            if idx < len(data.node_types):
                return data.node_types[idx]
        for name in data.node_types:
            if str(name).lower() == requested.lower():
                return name
        if len(data.node_types) > 0:
            return data.node_types[0]
    if hasattr(data, "_node_type_names") and requested in data._node_type_names:
        return requested
    return requested


class HeteroFromHomogeneousDataset:
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        hetero = data.to_heterogeneous()
        if hasattr(data, "y"):
            hetero.y = data.y
        if hasattr(data, "graph_attr"):
            hetero.graph_attr = data.graph_attr
        return hetero


class NodeTargetDatasetAdapter:
    def __init__(self, base, node_target_type: str):
        self.base = base
        self.node_target_type = node_target_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        if (
            not hasattr(data, "node_types")
            or self.node_target_type not in data.node_types
        ):
            raise RuntimeError(
                f"Node type '{self.node_target_type}' not found in OPF sample."
            )
        if (
            not hasattr(data[self.node_target_type], "y")
            or data[self.node_target_type].y is None
        ):
            raise RuntimeError(
                f"No targets found for node type '{self.node_target_type}' in OPF sample."
            )
        data.y = data[self.node_target_type].y
        ensure_node_y_loc(data)
        return data

    def __getattr__(self, name):
        return getattr(self.base, name)


class NodeBatchAdapter:
    def __init__(self, loader, node_target_type: str):
        self.loader = loader
        self.node_target_type = node_target_type
        self.dataset = loader.dataset
        self.sampler = getattr(loader, "sampler", None)

    def __iter__(self):
        for data in self.loader:
            if (
                not hasattr(data, "node_types")
                or self.node_target_type not in data.node_types
            ):
                raise RuntimeError(
                    f"Node type '{self.node_target_type}' not found in OPF sample."
                )

            if not hasattr(data, "batch"):
                node_store = data[self.node_target_type]
                if hasattr(node_store, "batch"):
                    data.batch = node_store.batch
                elif (
                    hasattr(data, "batch_dict")
                    and self.node_target_type in data.batch_dict
                ):
                    data.batch = data.batch_dict[self.node_target_type]
                elif hasattr(data, "batch_dict") and len(data.batch_dict) > 0:
                    data.batch = next(iter(data.batch_dict.values()))

            if (
                not hasattr(data[self.node_target_type], "y")
                or data[self.node_target_type].y is None
            ):
                raise RuntimeError(
                    f"No targets found for node type '{self.node_target_type}' in OPF sample."
                )
            data.y = data[self.node_target_type].y
            ensure_node_y_loc(data)
            yield data

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)
