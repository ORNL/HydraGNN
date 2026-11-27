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
"""CompositeNeuralOperator: a typed encoder -> processors -> decoders pipeline.

HydraGNN-native hybrid. It reuses tested building blocks of MeshGraphNet,
RIGNO and GSNO, but no combination is a faithful published architecture.

The latent state has two entities: ``node`` (fine mesh nodes) and ``region``
(RIGNO regional nodes, see :func:`hydragnn.data.meshes.build_rigno_graph`).
Every component declares the entity it reads and writes, the data attributes
it needs and the spatial dimensions it supports. :func:`resolve_pipeline`
checks the chain before any module is built:

* the encoder starts the chain; every later component must read the entity
  the previous one wrote;
* decoders read node latents, so a regional state must go through the
  ``RegionToNode`` adapter (the fine-mesh projection) before any node, cell,
  face or query decoder;
* latent widths must match between components; ``Projection`` changes the
  width explicitly;
* ``skip: true`` adds a residual around a component; it is rejected on
  adapters, which change the entity.

Configuration (``Architecture`` keys, see :class:`CompositeSettings`)::

    "mpnn_type": "CompositeNeuralOperator",
    "composite_encoder": {"type": "RegionalGraphEncoder"},
    "composite_processors": [
        {"type": "RegionalMessagePassing", "layers": 4},
        {"type": "RegionToNode"},
        {"type": "LocalMessagePassing", "layers": 2, "skip": true}
    ],
    "composite_decoders": {"face": "ConservativeFluxDecoder"}

``layers`` defaults to ``num_conv_layers`` and ``hidden_dim`` to the
Architecture ``hidden_dim``. Each output head gets the default decoder of its
level unless ``composite_decoders`` names another one.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.fno import resolve_activation
from hydragnn.models.utils import activation_function_selection
from hydragnn.models.gino import _graph_ids
from hydragnn.models.gsno import GraphSpectralConv
from hydragnn.models.meshgraphnet import (
    _mlp,
    _ProcessorStep,
    cell_latent,
    decode_face,
    query_latent,
)
from hydragnn.models.operator_base import OperatorBase
from hydragnn.models.rigno import _edge_features, _Interaction

_MESH_CELLS = ("cell_conn", "cell_size")


class _Component(nn.Module):
    """``forward(node, region, data) -> (node, region)``."""

    kind = "processor"
    in_entity = "node"
    out_entity = "node"
    requires = frozenset()
    spatial_dims = frozenset({2, 3})
    options = frozenset()

    @classmethod
    def out_width(cls, spec, width):
        return width


class NodeEncoder(_Component):
    """MLP from the node features ``x``."""

    kind = "encoder"
    in_entity = None
    requires = frozenset({"x"})

    def __init__(self, ctx, spec, width):
        super().__init__()
        self.mlp = _mlp(
            ctx["input_dim"], width, width, ctx["mlp_layers"], ctx["act"], True
        )

    def forward(self, node, region, data):
        return self.mlp(data.x), None


class NodeToRegion(_Component):
    """RIGNO encoder interaction: physical nodes -> regional nodes over
    ``rigno_p2r_index``; both node sets are updated."""

    kind = "adapter"
    out_entity = "region"
    requires = frozenset({"rigno_x", "rigno_rpos", "rigno_rradius", "rigno_p2r_index"})
    spatial_dims = frozenset({2})

    def __init__(self, ctx, spec, width):
        super().__init__()
        d, layers, act = ctx["spatial_dim"], ctx["mlp_layers"], ctx["act"]
        self.rnode_embed = _mlp(d + 1, width, width, layers, act, True)
        self.edge_embed = _mlp(d + 1, width, width, layers, act, True)
        self.interaction = _Interaction(width, layers, act, False, True)

    def forward(self, node, region, data):
        radius = data.rigno_rradius.unsqueeze(1).to(node.dtype)
        rpos = data.rigno_rpos.to(node.dtype)
        region = self.rnode_embed(torch.cat([rpos, radius], dim=-1))
        index = data.rigno_p2r_index
        edge = self.edge_embed(_edge_features(data.rigno_x.to(node.dtype), rpos, index))
        node, region, _ = self.interaction(node, region, edge, index, (None,) * 3)
        return node, region


class RegionalGraphEncoder(_Component):
    """RIGNO-style encoder: embed ``[x, rigno_x]`` on the nodes, then
    :class:`NodeToRegion`. Leaves the regional state current."""

    kind = "encoder"
    in_entity = None
    out_entity = "region"
    requires = frozenset({"x"}) | NodeToRegion.requires
    spatial_dims = NodeToRegion.spatial_dims

    def __init__(self, ctx, spec, width):
        super().__init__()
        d = ctx["spatial_dim"]
        self.node_embed = _mlp(
            ctx["input_dim"] + d, width, width, ctx["mlp_layers"], ctx["act"], True
        )
        self.to_region = NodeToRegion(ctx, spec, width)

    def forward(self, node, region, data):
        node = self.node_embed(torch.cat([data.x, data.rigno_x.to(data.x.dtype)], -1))
        return self.to_region(node, None, data)


class LocalMessagePassing(_Component):
    """MeshGraphNet processor steps on ``edge_index`` with an own edge
    encoder over ``[x_s - x_r, |x_s - x_r|]``."""

    requires = frozenset({"pos", "edge_index"})
    options = frozenset({"layers"})

    def __init__(self, ctx, spec, width):
        super().__init__()
        d, layers, act = ctx["spatial_dim"], ctx["mlp_layers"], ctx["act"]
        self.spatial_dim = d
        self.edge_encoder = _mlp(d + 1, width, width, layers, act, True)
        self.steps = nn.ModuleList(
            _ProcessorStep(width, layers, act)
            for _ in range(spec.get("layers", ctx["num_layers"]))
        )

    def forward(self, node, region, data):
        pos = data.pos[:, : self.spatial_dim]
        src, dst = data.edge_index
        rel = pos[src] - pos[dst]
        edge = self.edge_encoder(
            torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1).to(node.dtype)
        )
        for step in self.steps:
            node, edge = step(node, edge, data.edge_index)
        return node, region


class GraphSpectral(_Component):
    """GSNO steady layers ``act(Phi R Phi^T v + W v)`` on ``gsno_basis``."""

    requires = frozenset({"gsno_basis"})
    options = frozenset({"layers", "modes"})

    def __init__(self, ctx, spec, width):
        super().__init__()
        self.modes = int(spec.get("modes", 8))
        if self.modes < 1:
            raise ValueError("GraphSpectral modes must be >= 1")
        self.act = ctx["act"]
        layers = spec.get("layers", ctx["num_layers"])
        self.convs = nn.ModuleList(
            GraphSpectralConv(width, self.modes) for _ in range(layers)
        )
        self.skips = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))

    def forward(self, node, region, data):
        if data.gsno_basis.shape[1] < self.modes:
            raise ValueError(
                f"GraphSpectral needs data.gsno_basis with >= {self.modes} columns"
            )
        graph = _graph_ids(data)
        num_graphs = int(graph.max()) + 1
        basis = data.gsno_basis.to(node.dtype)
        v = node.unsqueeze(1)
        for conv, skip in zip(self.convs, self.skips):
            v = self.act(conv(v, basis, graph, num_graphs) + skip(v))
        return v.squeeze(1), region


class RegionalMessagePassing(_Component):
    """RIGNO processor interactions on the regional multimesh
    ``rigno_r2r_index``."""

    in_entity = "region"
    out_entity = "region"
    requires = frozenset({"rigno_rpos", "rigno_r2r_index"})
    spatial_dims = frozenset({2})
    options = frozenset({"layers"})

    def __init__(self, ctx, spec, width):
        super().__init__()
        d, layers, act = ctx["spatial_dim"], ctx["mlp_layers"], ctx["act"]
        self.edge_embed = _mlp(d + 1, width, width, layers, act, True)
        self.steps = nn.ModuleList(
            _Interaction(width, layers, act, False, False)
            for _ in range(spec.get("layers", ctx["num_layers"]))
        )

    def forward(self, node, region, data):
        rpos = data.rigno_rpos.to(region.dtype)
        index = data.rigno_r2r_index
        edge = self.edge_embed(_edge_features(rpos, rpos, index))
        for step in self.steps:
            _, region, edge = step(region, region, edge, index, (None,) * 3)
        return node, region


class RegionToNode(_Component):
    """RIGNO decoder interaction: regional -> physical nodes over
    ``rigno_r2p_index``. The receivers are the latest node latents, so the
    fine-mesh state from before the regional stage is reused."""

    kind = "adapter"
    in_entity = "region"
    requires = frozenset({"rigno_x", "rigno_rpos", "rigno_r2p_index"})
    spatial_dims = frozenset({2})

    def __init__(self, ctx, spec, width):
        super().__init__()
        d, layers, act = ctx["spatial_dim"], ctx["mlp_layers"], ctx["act"]
        self.edge_embed = _mlp(d + 1, width, width, layers, act, True)
        self.interaction = _Interaction(width, layers, act, False, False)

    def forward(self, node, region, data):
        index = data.rigno_r2p_index
        edge = self.edge_embed(
            _edge_features(
                data.rigno_rpos.to(region.dtype),
                data.rigno_x.to(region.dtype),
                index,
            )
        )
        _, node, _ = self.interaction(region, node, edge, index, (None,) * 3)
        return node, region


class Projection(_Component):
    """Linear map of the current entity to ``hidden_dim`` channels."""

    kind = "adapter"
    in_entity = "any"
    out_entity = "any"
    options = frozenset({"hidden_dim"})

    @classmethod
    def out_width(cls, spec, width):
        if "hidden_dim" not in spec:
            raise ValueError("Projection needs hidden_dim")
        return int(spec["hidden_dim"])

    def __init__(self, ctx, spec, width):
        super().__init__()
        self.entity = spec["_entity"]
        self.linear = nn.Linear(width, self.out_width(spec, width))

    def forward(self, node, region, data):
        if self.entity == "region":
            return node, self.linear(region)
        return self.linear(node), region


COMPONENTS = {
    cls.__name__: cls
    for cls in (
        NodeEncoder,
        RegionalGraphEncoder,
        LocalMessagePassing,
        GraphSpectral,
        RegionalMessagePassing,
        NodeToRegion,
        RegionToNode,
        Projection,
    )
}

# Decoder type -> (output level, extra data attributes).
DECODERS = {
    "NodeDecoder": ("node", frozenset()),
    "CellDecoder": ("cell", frozenset(_MESH_CELLS)),
    "ConservativeFluxDecoder": (
        "face",
        frozenset({"face_index", "face_normal", "face_area", "pos", *_MESH_CELLS}),
    ),
    "FluxDecoder": (
        "face",
        frozenset({"face_index", "face_normal", "face_area", "pos", *_MESH_CELLS}),
    ),
    "QueryDecoder": ("query", frozenset({"query_pos", "query_cell", *_MESH_CELLS})),
}
DEFAULT_DECODERS = {
    "node": "NodeDecoder",
    "cell": "CellDecoder",
    "face": "ConservativeFluxDecoder",
    "query": "QueryDecoder",
}


def resolve_pipeline(encoder, processors, decoders, output_type, width, spatial_dim):
    """Validate a composite configuration.

    Returns ``(stages, decoder_types, latent_width, required_attributes)``
    where ``stages`` is a list of ``(class, spec, in_width)``.
    """
    if not isinstance(encoder, dict) or "type" not in encoder:
        raise ValueError("composite_encoder must be a dict with a 'type'")
    if not isinstance(processors, (list, tuple)):
        raise ValueError("composite_processors must be a list")
    specs = [dict(encoder)] + [dict(spec) for spec in processors]
    stages, required, entity = [], set(), None
    for position, spec in enumerate(specs):
        name = spec.get("type")
        cls = COMPONENTS.get(name)
        where = "composite_encoder" if position == 0 else f"processor {position}"
        if cls is None:
            raise ValueError(
                f"{where}: unknown component {name!r}; available: {sorted(COMPONENTS)}"
            )
        if (position == 0) != (cls.kind == "encoder"):
            raise ValueError(
                f"{where}: {name} is {'an' if cls.kind == 'encoder' else 'a'} "
                f"{cls.kind}; the pipeline must start with exactly one encoder"
            )
        if cls.in_entity not in (None, "any", entity):
            raise ValueError(
                f"{where}: {name} reads {cls.in_entity} latents but the previous "
                f"component produces {entity} latents"
            )
        if spatial_dim not in cls.spatial_dims:
            raise ValueError(
                f"{where}: {name} supports spatial_dim {sorted(cls.spatial_dims)}, "
                f"got {spatial_dim}"
            )
        unknown = set(spec) - {"type", "skip", "hidden_dim"} - cls.options
        if unknown:
            raise ValueError(f"{where}: {name} does not accept {sorted(unknown)}")
        skip = bool(spec.get("skip", False))
        if skip and cls.kind != "processor":
            raise ValueError(f"{where}: skip is only allowed on processors")
        if "hidden_dim" in spec and cls is not Projection:
            if int(spec["hidden_dim"]) != width:
                raise ValueError(
                    f"{where}: {name} hidden_dim {spec['hidden_dim']} does not match "
                    f"the incoming latent width {width}; insert a Projection"
                )
        spec["_entity"] = entity
        stages.append((cls, spec, width))
        width = cls.out_width(spec, width)
        if cls.out_entity != "any":
            entity = cls.out_entity
        required |= cls.requires

    decoders = dict(decoders or {})
    bad_levels = set(decoders) - set(DEFAULT_DECODERS)
    if bad_levels:
        raise ValueError(f"composite_decoders: unknown levels {sorted(bad_levels)}")
    decoder_types = []
    for level in output_type:
        name = decoders.get(level, DEFAULT_DECODERS.get(level))
        if name not in DECODERS or DECODERS[name][0] != level:
            raise ValueError(
                f"composite_decoders: {name!r} cannot decode {level} outputs"
            )
        if entity != "node":
            raise ValueError(
                f"{name} needs node latents, but the pipeline ends on {entity} "
                "latents; add a RegionToNode adapter to project the regional "
                "state back to the fine mesh"
            )
        if level == "query" and spatial_dim != 2:
            raise ValueError("QueryDecoder supports spatial_dim 2 only")
        decoder_types.append(name)
        required |= DECODERS[name][1]
    return stages, decoder_types, width, frozenset(required)


class CompositeNeuralOperator(OperatorBase):
    supported_output_levels = frozenset({"node", "cell", "face", "query"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_layers,
        encoder=None,
        processors=None,
        decoders=None,
        mlp_layers=2,
        spatial_dim=2,
        activation_function="relu",
        loss_function_type="mse",
        loss_weights=None,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if mlp_layers < 1:
            raise ValueError("composite_mlp_layers must be >= 1")
        if spatial_dim not in (2, 3):
            raise ValueError("composite_spatial_dim must be 2 or 3")
        if activation_function_selection(activation_function) is None:
            raise ValueError(
                f"CompositeNeuralOperator does not support activation "
                f"{activation_function!r}; MLP blocks use HydraGNN's "
                "activation_function_selection names (relu, silu, tanh, ...)"
            )
        encoder = encoder if encoder is not None else {"type": "NodeEncoder"}
        processors = (
            processors if processors is not None else [{"type": "LocalMessagePassing"}]
        )
        stages, self.decoder_types, latent, self.required_data = resolve_pipeline(
            encoder, processors, decoders, self.head_type, hidden_dim, spatial_dim
        )
        self.spatial_dim = spatial_dim
        ctx = dict(
            input_dim=input_dim,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            spatial_dim=spatial_dim,
            act=activation_function,
        )
        # GraphSpectral follows GSNO and takes a functional activation.
        spectral_ctx = {**ctx, "act": resolve_activation(activation_function)}
        self.components = nn.ModuleList(
            cls(spectral_ctx if cls is GraphSpectral else ctx, spec, width)
            for cls, spec, width in stages
        )
        self.skips = [bool(spec.get("skip", False)) for _, spec, _ in stages]
        out_mult = 1 + self.var_output
        face_in = 2 * latent + spatial_dim + 2
        self.decoders = nn.ModuleList(
            _mlp(
                face_in if DECODERS[name][0] == "face" else latent,
                latent,
                dim * out_mult,
                mlp_layers,
                activation_function,
                False,
            )
            for name, dim in zip(self.decoder_types, self.head_dims)
        )
        if freeze_conv:
            self.components.requires_grad_(False)

    def _check_data(self, data):
        missing = sorted(
            a for a in self.required_data if getattr(data, a, None) is None
        )
        if missing:
            raise ValueError(
                f"CompositeNeuralOperator pipeline needs data attributes {missing}"
            )

    def encode(self, data):
        self._check_data(data)
        node, region = self.components[0](None, None, data)
        return {"node": node, "region": region}

    def process(self, state, data):
        node, region = state["node"], state["region"]
        for component, skip in zip(self.components[1:], self.skips[1:]):
            if self.conv_checkpointing and self.training:
                new_node, new_region = checkpoint(
                    component, node, region, data, use_reentrant=False
                )
            else:
                new_node, new_region = component(node, region, data)
            if skip:
                if component.in_entity == "region":
                    new_region = new_region + region
                else:
                    new_node = new_node + node
            node, region = new_node, new_region
        return {"node": node, "region": region}

    def decode(self, state, data):
        node = state["node"]
        outputs = []
        for name, dim, decoder in zip(
            self.decoder_types, self.head_dims, self.decoders
        ):
            if name == "NodeDecoder":
                outputs.append(decoder(node))
            elif name == "CellDecoder":
                outputs.append(decoder(cell_latent(node, data)))
            elif name == "QueryDecoder":
                outputs.append(decoder(query_latent(node, data, "Composite")))
            else:
                outputs.append(
                    decode_face(
                        decoder,
                        dim,
                        node,
                        data,
                        self.spatial_dim,
                        name == "ConservativeFluxDecoder",
                    )
                )
        return outputs
