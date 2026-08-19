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
# Wraps:
#   FairChem UMA (Universal Models for Atoms) -- ``eSCNMDBackbone``
#   https://github.com/facebookresearch/fairchem
#   Distributed under the MIT License.
###############################################################################
"""
HydraGNN integration of FairChem's UMA model.

The upstream ``eSCNMDBackbone`` (and all of its transitive fairchem-core
dependencies -- SO(3)/SO(2) primitives, Wigner-D tables, rotation and
radial modules) is vendored with documented namespace/import transformations under
``hydragnn/utils/model/uma/_vendored/fairchem/core/**`` (Meta MIT licence
preserved). ``UMAStack`` imports the vendored class directly, so there
is no runtime dependency on the external ``fairchem-core`` distribution.

Refresh it from an explicit FAIR-Chem checkout with
``python tools/vendor_uma.py --apply --fairchem-source /path/to/fairchem``;
the tool records the upstream commit in ``PROVENANCE.json``.

HydraGNN does not currently vendor FAIR-Chem's distributed ``GPContext``,
graph partitioning, or differentiable all-to-all runtime. Consequently this
wrapper supports large local graphs but does **not** have current FAIR-Chem
UMA graph-parallel parity. The adapter boundary intentionally leaves room for
a future distributed data dictionary / context without changing task heads.

Design summary
--------------
UMA is a monolithic equivariant transformer: it owns its own block
loop, normalization, edge-degree embedding, and (optional)
charge / spin / dataset embeddings. To plug it into HydraGNN we:

* run the **entire** UMA backbone inside ``_embedding(...)``,
* slice the per-degree spherical-harmonic output (shape
  ``(N, (lmax+1)**2, sphere_channels)``) into an L=0 invariant
  feature for HydraGNN's standard scalar decoders, and
* register a single identity placeholder convolution so the
  ``Base.forward`` per-layer loop becomes a no-op.

Hyperparameter mapping
----------------------
HydraGNN's standard ``Architecture`` keys are reused wherever the
semantics overlap; the UMA-specific keys keep the ``uma_`` prefix.

============================  ===========================  ====================
HydraGNN key                  UMA constructor argument     Notes
============================  ===========================  ====================
``radius``                    ``cutoff``                   Edge cutoff (Å).
``max_neighbours``            ``max_neighbors``            Per-node neighbor cap.
``hidden_dim``                ``sphere_channels``          Width per irrep.
``max_ell``                   ``lmax``                     Max node degree.
``num_conv_layers``           ``num_layers``               Transformer depth.
``num_radial``                ``num_distance_basis``       Bessel basis size.
``activation_function``       (mapped to ``act_type``)     "gate" if "silu" / "swish" /
                                                          unknown, else passthrough.
``periodic_boundary_conditions`` ``always_use_pbc``        PBC handling.
``equivariance``              must be ``True``             UMA is genuinely equivariant.
============================  ===========================  ====================

UMA-specific keys (no HydraGNN equivalent):

================================  =========================================
Key                               Purpose
================================  =========================================
``uma_mmax``                      SO(2)-rotated convolution azimuthal cap.
``uma_grid_resolution``           Optional Gauss-Legendre grid size for
                                  SO(3) -> grid -> SO(3) round-trip.
``uma_edge_channels``             Edge MLP width for the radial path.
``uma_hidden_channels``           FFN / SO(2)-conv hidden width *inside*
                                  one transformer block. Distinct from
                                  ``hidden_dim`` (= ``sphere_channels``,
                                  the per-irrep node channel width carried
                                  *between* blocks). Defaults to
                                  ``hidden_dim`` if left as ``None``.
``uma_norm_type``                 Equivariant normalization ("rms_norm_sh").
``uma_ff_type``                   FFN style ("grid" or "spectral").
``uma_use_chg_spin``              Enable optional ChgSpinEmbedding.
``uma_max_num_elements``          Z embedding table size.
``uma_variant``                   UMA capacity tier: ``"S"`` (single
                                  dense backbone), ``"M"`` or ``"L"``
                                  (Mixture-of-Linear-Experts routing).
``uma_num_experts``               Override the routed-expert count for
                                  the ``"M"`` / ``"L"`` variants
                                  (defaults: M=8, L=32; ignored for S).
``uma_moe_dropout``               Dropout on the MoLE routing weights.
``uma_use_composition_embedding`` Route experts using the atomic
                                  composition in addition to charge/spin.
``uma_equivariant_vector_head``   Enable an SO(3)-equivariant per-node
                                  vector head (ported from fairchem's
                                  ``Linear_Force_Head``) that reads UMA's
                                  L=1 irrep to produce genuine 3-vectors
                                  for a designated 'node' output head.
``uma_vector_head_index``         Index of the 'node' output head the
                                  equivariant vector maps to (its dim must
                                  be a multiple of 3). ``None`` auto-detects
                                  the unique matching node head.
================================  =========================================

Equivariance
------------
**UMA IS genuinely e3nn-equivariant.** Hidden node features live in
SO(3) irrep space ``(N, (lmax+1)**2, sphere_channels)`` throughout the
network, and rotation-equivariant tensor products / SO(2) convolutions
preserve that structure. For graph-level scalar predictions this
yields rotation/translation invariance; for vector / tensor outputs
the L=1 / L=2 channels can be sliced from ``node_embedding`` to obtain
genuinely equivariant predictions. The wrapper sets
``equivariance=True`` in :class:`Base` so HydraGNN's downstream
machinery (vector outputs, forces from autograd) is consistent.

By default only the L=0 invariant scalar is read out (energies stay
invariant; conservative forces from autograd stay equivariant). Set
``uma_equivariant_vector_head=True`` to additionally expose a
direct SO(3)-equivariant per-node vector prediction from the L=1 irrep
(see :class:`_UMAEquivariantVectorHead`); this populates the node
output head selected by ``uma_vector_head_index``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Identity, ModuleList

from hydragnn.models.Base import Base

# NOTE: the vendored FairChem UMA backbone is imported lazily (see
# ``_load_uma_backbones``) rather than at module import time. This keeps
# ``import hydragnn`` fast -- the sizeable vendored fairchem-core tree is only
# loaded when a UMA model is actually constructed. Its lightweight import-time
# deps (omegaconf, monty) are declared in requirements-specific-models.txt; the
# heavier fairchem-core deps (torchtnt/ray/wandb/hydra) live only in checkpoint/
# inference code paths that HydraGNN never reaches.


def _load_uma_backbones():
    """Import and return the vendored UMA backbone classes.

    Raises a clear, actionable :class:`ImportError` if the UMA backbone
    dependencies are missing, instead of the opaque failure that would
    otherwise surface deep inside the vendored fairchem-core tree.
    """
    try:
        from hydragnn.utils.model.uma._vendored.fairchem.core.models.uma.escn_md import (
            eSCNMDBackbone,
        )
        from hydragnn.utils.model.uma._vendored.fairchem.core.models.uma.escn_moe import (
            eSCNMDMoeBackbone,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The UMA backbone could not be imported. Install its optional "
            "dependencies:\n"
            "    pip install omegaconf monty\n"
            "(declared in requirements-specific-models.txt; underlying import "
            f"error: {exc})"
        ) from exc
    return eSCNMDBackbone, eSCNMDMoeBackbone


# Default expert counts for the Mixture-of-Linear-Experts (MoLE) UMA
# variants when the user does not override ``uma_num_experts``. UMA's "S"
# variant is a single dense backbone (no experts); "M" and "L" grow the
# routed-expert capacity.
_UMA_VARIANT_DEFAULT_EXPERTS = {"M": 8, "L": 32}


class _UMADataDict(dict):
    """A ``dict`` that tolerates upstream UMA's ``.get(key, default=...)``.

    FairChem's ``eSCNMDBackbone.forward`` calls ``data_dict.get("dataset",
    default=None)`` (with ``default`` as a keyword), which is a syntax
    error against a plain ``dict`` (``dict.get`` takes only positional
    arguments). This subclass accepts the keyword form transparently.
    """

    def get(self, key, default=None, **kwargs):  # type: ignore[override]
        if "default" in kwargs:
            default = kwargs["default"]
        return super().get(key, default)


class _IdentityConv(torch.nn.Module):
    """No-op stand-in used in place of HydraGNN's standard graph_conv.

    UMA runs its full message-passing stack inside ``_embedding``, so
    the standard ``Base.forward`` conv loop has nothing to do. We
    register a single instance and force ``num_conv_layers = 1`` for
    this stack (see :class:`UMAStack.__init__`).
    """

    def forward(self, inv_node_feat, equiv_node_feat, **_kwargs):
        return inv_node_feat, equiv_node_feat


class _UMAEquivariantVectorHead(torch.nn.Module):
    """Rotation-equivariant per-node vector head for UMA.

    Ported from fairchem's ``Linear_Force_Head``: an ``SO3_Linear`` acts on
    the L=0/L=1 block of UMA's spherical-harmonic node embedding and the L=1
    (vector) component of the result is read out, yielding ``num_vectors``
    genuine 3-vectors per node. Because ``SO3_Linear`` applies a shared weight
    across the three L=1 components (mixing only the channel axis), the map
    commutes with the rotation representation and the output vectors transform
    equivariantly with the input.
    """

    def __init__(self, sphere_channels: int, num_vectors: int = 1) -> None:
        super().__init__()
        # Lazy import: keeps ``import hydragnn`` cheap and matches the rest of
        # the UMA integration (the vendored fairchem tree is only touched when
        # a UMA model is actually built). ``so3_layers`` itself only imports
        # ``math``/``torch``.
        from hydragnn.utils.model.uma._vendored.fairchem.core.models.uma.nn.so3_layers import (  # noqa: E501
            SO3_Linear,
        )

        self.num_vectors = int(num_vectors)
        # lmax=1 SO3_Linear consumes the L=0 (index 0) + L=1 (indices 1..3)
        # components and returns the same irrep layout with ``num_vectors``
        # output channels.
        self.linear = SO3_Linear(int(sphere_channels), self.num_vectors, lmax=1)

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        # node_embedding: (N, (lmax+1)**2, C). Take the first 4 harmonics
        # (L=0 + L=1) that SO3_Linear(lmax=1) expects.
        l0_l1 = node_embedding.narrow(1, 0, 4)  # (N, 4, C)
        out = self.linear(l0_l1)  # (N, 4, num_vectors)
        vec = out.narrow(1, 1, 3)  # (N, 3, num_vectors) -- L=1 component
        # (N, 3, V) -> (N, V, 3) -> (N, 3V) so each vector's xyz is contiguous.
        return vec.permute(0, 2, 1).reshape(
            node_embedding.shape[0], 3 * self.num_vectors
        )


class UMAStack(Base):
    """HydraGNN wrapper around FairChem's ``eSCNMDBackbone``.

    See module docstring for hyperparameter mapping and design notes.
    """

    def __init__(
        self,
        input_args,
        conv_args,
        radius: float,
        max_neighbours: int,
        max_ell: int,
        num_radial: int,
        uma_mmax: int = 2,
        uma_grid_resolution: Optional[int] = None,
        uma_edge_channels: int = 128,
        uma_hidden_channels: Optional[int] = None,
        uma_norm_type: str = "rms_norm_sh",
        uma_ff_type: str = "grid",
        uma_use_chg_spin: bool = False,
        uma_max_num_elements: int = 100,
        uma_variant: str = "S",
        uma_num_experts: Optional[int] = None,
        uma_moe_dropout: float = 0.0,
        uma_use_composition_embedding: bool = False,
        periodic_boundary_conditions: bool = False,
        *args,
        uma_equivariant_vector_head: bool = False,
        uma_vector_head_index: Optional[int] = None,
        **kwargs,
    ):
        # --- Stash UMA-specific args before calling Base.__init__ ---
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.max_ell = max_ell
        self.num_radial = num_radial
        # Re-use HydraGNN's standard num_conv_layers as the UMA depth.
        self.uma_num_layers = kwargs.get("num_conv_layers", 2)
        # mmax must satisfy mmax <= lmax; otherwise UMA's SO(2) convs
        # construct zero-channel layers and fail at runtime.
        self.uma_mmax = min(int(uma_mmax), int(max_ell))
        self.uma_grid_resolution = uma_grid_resolution
        self.uma_edge_channels = uma_edge_channels
        # Default the per-block FFN/SO(2) hidden width to hidden_dim so a
        # user only setting hidden_dim ends up with sphere_channels ==
        # hidden_channels (UMA's published default behaviour).
        self.uma_hidden_channels = (
            uma_hidden_channels
            if uma_hidden_channels is not None
            else int(kwargs.get("hidden_dim", args[1] if len(args) >= 2 else 128))
        )
        self.uma_norm_type = uma_norm_type
        self.uma_ff_type = uma_ff_type
        self.uma_use_chg_spin = uma_use_chg_spin
        self.uma_max_num_elements = uma_max_num_elements
        self.uma_periodic = bool(periodic_boundary_conditions)

        # UMA S / M / L variant selection. "S" is a single dense
        # backbone (eSCNMDBackbone); "M" / "L" enable Mixture-of-Linear-
        # Experts (MoLE) routing via eSCNMDMoeBackbone with progressively
        # larger expert counts.
        variant = str(uma_variant).upper()
        if variant not in ("S", "M", "L"):
            raise ValueError(
                f"uma_variant must be one of 'S', 'M', 'L'; got {uma_variant!r}."
            )
        self.uma_variant = variant
        if uma_num_experts is not None:
            self.uma_num_experts = int(uma_num_experts)
        else:
            self.uma_num_experts = _UMA_VARIANT_DEFAULT_EXPERTS.get(variant, 0)
        self.uma_moe_dropout = float(uma_moe_dropout)
        self.uma_use_composition_embedding = bool(uma_use_composition_embedding)

        # Optional rotation-equivariant per-node vector head (ported from
        # fairchem's Linear_Force_Head). Stashed here so _init_conv (called by
        # super().__init__) can build it once the head config is known.
        self.uma_equivariant_vector_head_enabled = bool(uma_equivariant_vector_head)
        self.uma_vector_head_index = uma_vector_head_index

        # Capture the HydraGNN activation-function string and map to UMA's
        # supported act_type. UMA accepts "gate" or "s2"; other names
        # fall back to "gate" (the published default).
        if "activation_function_type" in kwargs:
            hydragnn_act = kwargs["activation_function_type"]
        elif len(args) >= 10:
            hydragnn_act = args[9]
        else:
            hydragnn_act = "silu"
        _ACT_MAP = {"gate": "gate", "s2": "s2"}
        self.uma_act_type = _ACT_MAP.get(hydragnn_act, "gate")

        # UMA expects an externally provided edge_index (otf_graph=False).
        # Mark this as an edge-aware model so HydraGNN's preprocess sets
        # one up via the standard radius graph transform.
        self.is_edge_model = True
        # UMA completes its encoder in _embedding; do not alter the spherical
        # representation in Base's identity-convolution wrapper.
        self.skip_post_conv_processing = True
        # Force num_conv_layers=1 for the Base forward loop. The actual
        # UMA depth lives in self.uma_num_layers.
        kwargs["num_conv_layers"] = 1

        super().__init__(input_args, conv_args, *args, **kwargs)

    def _init_conv(self):
        # Build the UMA backbone with HydraGNN-derived configuration.
        # We disable the dataset embedding (no UMA-style multi-task
        # routing in HydraGNN datasets) and gradient-based force/stress
        # heads (HydraGNN computes those itself when requested).
        backbone_cfg: Dict[str, Any] = {
            "max_num_elements": self.uma_max_num_elements,
            "sphere_channels": int(self.hidden_dim),
            "lmax": int(self.max_ell),
            "mmax": int(self.uma_mmax),
            "grid_resolution": self.uma_grid_resolution,
            "otf_graph": False,
            "max_neighbors": int(self.max_neighbours),
            "cutoff": float(self.radius),
            "edge_channels": int(self.uma_edge_channels),
            "distance_function": "gaussian",
            "num_distance_basis": int(self.num_radial),
            "direct_forces": False,
            "regress_forces": False,
            "direct_stress": False,
            "regress_stress": False,
            "regress_hessian": False,
            "num_layers": int(self.uma_num_layers),
            "hidden_channels": int(self.uma_hidden_channels),
            "norm_type": self.uma_norm_type,
            "act_type": self.uma_act_type,
            "ff_type": self.uma_ff_type,
            "activation_checkpointing": False,
            "use_dataset_embedding": False,
            "dataset_list": None,
            "dataset_mapping": {"_hydragnn_default": "_hydragnn_default"},
            "use_quaternion_wigner": False,
            "always_use_pbc": self.uma_periodic,
        }

        if self.uma_variant == "S":
            # Single dense backbone -- no routed experts.
            eSCNMDBackbone, _ = _load_uma_backbones()
            self.uma_backbone = eSCNMDBackbone(**backbone_cfg)
        else:
            # "M" / "L": Mixture-of-Linear-Experts (MoLE) routing. The
            # routing coefficients are derived from the per-system
            # charge/spin (and optionally composition) embeddings, so no
            # HydraGNN-side dataset labelling is required.
            _, eSCNMDMoeBackbone = _load_uma_backbones()
            self.uma_backbone = eSCNMDMoeBackbone(
                num_experts=int(self.uma_num_experts),
                moe_dropout=self.uma_moe_dropout,
                use_composition_embedding=self.uma_use_composition_embedding,
                moe_type="so2",
                **backbone_cfg,
            )
        # Cache lmax once -- UMA's node_embedding has shape
        # (N, (lmax+1)**2, sphere_channels). We pull L=0 (index 0) as
        # the invariant scalar feature for HydraGNN's standard decoders.
        self._sph_l0_index = 0

        # Optional rotation-equivariant per-node vector head. When enabled it
        # populates a designated 'node' output head with genuine 3-vectors read
        # from UMA's L=1 irrep (see _UMAEquivariantVectorHead / forward()).
        self.equivariant_vector_head = None
        self._vector_head_index = None
        self._equivariant_vector_cache = None
        if self.uma_equivariant_vector_head_enabled:
            if int(self.max_ell) < 1:
                raise ValueError(
                    "uma_equivariant_vector_head requires max_ell >= 1 so the "
                    f"L=1 irrep exists (got max_ell={self.max_ell})."
                )
            idx = self._resolve_vector_head_index()
            head_dim = int(self.head_dims[idx])
            if head_dim % 3 != 0:
                raise ValueError(
                    "The UMA equivariant vector head targets node output head "
                    f"{idx}, whose dimension ({head_dim}) is not a multiple of "
                    "3; each equivariant vector contributes exactly 3 "
                    "components."
                )
            self._vector_head_index = idx
            self.equivariant_vector_head = _UMAEquivariantVectorHead(
                int(self.hidden_dim), num_vectors=head_dim // 3
            )

        # HydraGNN's Base.forward iterates over (graph_convs, feature_layers)
        # exactly num_conv_layers times. We forced num_conv_layers=1 in
        # __init__ and supply identity placeholders so the loop is a no-op.
        self.graph_convs = ModuleList([_IdentityConv()])
        self.feature_layers = ModuleList([Identity()])

    def _resolve_vector_head_index(self) -> int:
        """Pick which HydraGNN output head the equivariant vector maps to."""
        if self.uma_vector_head_index is not None:
            idx = int(self.uma_vector_head_index)
            if not (0 <= idx < self.num_heads):
                raise ValueError(
                    f"uma_vector_head_index={idx} is out of range for "
                    f"{self.num_heads} output heads."
                )
            if self.head_type[idx] != "node":
                raise ValueError(
                    "The UMA equivariant vector head must target a 'node' "
                    f"output head; head {idx} is of type "
                    f"'{self.head_type[idx]}'."
                )
            return idx
        candidates = [
            i
            for i, htype in enumerate(self.head_type)
            if htype == "node" and int(self.head_dims[i]) % 3 == 0
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Could not auto-detect a single 'node' output head whose "
                "dimension is a multiple of 3 for the UMA equivariant vector "
                "head; set uma_vector_head_index explicitly. Node-head "
                f"candidates (index) = {candidates}."
            )
        return candidates[0]

    # --- HydraGNN data adapter -----------------------------------------------

    def _build_data_dict(self, data) -> Dict[str, Any]:
        """Translate a HydraGNN PyG ``Data`` batch to UMA's data_dict.

        UMA's ``forward(data_dict)`` expects FairChem ``AtomicData``-style
        dictionary access; we hand-build the minimum subset required when
        ``otf_graph=False`` and dataset / charge / spin embeddings are
        either disabled or defaulted.
        """
        device = data.pos.device
        dtype = data.pos.dtype

        if hasattr(data, "atomic_numbers") and data.atomic_numbers is not None:
            atomic_numbers = data.atomic_numbers.long().view(-1)
        else:
            atomic_numbers = data.x[:, 0].long().view(-1)

        batch = (
            data.batch
            if data.batch is not None
            else torch.zeros(data.pos.shape[0], dtype=torch.long, device=device)
        )
        num_systems = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        natoms = torch.bincount(batch, minlength=num_systems).long()

        # Cell / PBC: optional. Default to identity cell with PBC off.
        if hasattr(data, "cell") and data.cell is not None:
            cell = data.cell.to(device=device, dtype=dtype)
            if cell.dim() == 2 and cell.shape == (3, 3):
                cell = cell.unsqueeze(0).expand(num_systems, 3, 3).contiguous()
            elif cell.dim() == 2 and cell.shape[0] == 3 * num_systems:
                # PyG batching of per-graph (3, 3) cells -> (3*B, 3).
                cell = cell.view(num_systems, 3, 3).contiguous()
            elif cell.dim() == 3:
                cell = cell.contiguous()
            else:
                raise ValueError(
                    f"Unexpected cell shape {tuple(cell.shape)} for "
                    f"num_systems={num_systems}."
                )
        else:
            cell = (
                torch.eye(3, device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(num_systems, 3, 3)
                .contiguous()
            )
        if hasattr(data, "pbc") and data.pbc is not None:
            pbc = data.pbc.to(device=device, dtype=torch.bool)
            if pbc.dim() == 1 and pbc.numel() == 3:
                pbc = pbc.unsqueeze(0).expand(num_systems, 3).contiguous()
            elif pbc.dim() == 1 and pbc.numel() == 3 * num_systems:
                # PyG batching of per-graph (3,) pbc -> (3*B,).
                pbc = pbc.view(num_systems, 3).contiguous()
            elif pbc.dim() == 2:
                pbc = pbc.contiguous()
            else:
                raise ValueError(
                    f"Unexpected pbc shape {tuple(pbc.shape)} for "
                    f"num_systems={num_systems}."
                )
        else:
            pbc = torch.full(
                (num_systems, 3),
                self.uma_periodic,
                device=device,
                dtype=torch.bool,
            )

        # Edge bookkeeping. HydraGNN stores Cartesian shifts using the
        # opposite edge-vector convention from UMA.
        edge_index = data.edge_index
        edge_batch = batch[edge_index[0]]
        nedges = torch.bincount(edge_batch, minlength=num_systems).long()
        if edge_batch.numel() > 1 and torch.any(edge_batch[1:] < edge_batch[:-1]):
            raise ValueError("UMA requires edges to be grouped by system.")

        if hasattr(data, "edge_shifts") and data.edge_shifts is not None:
            edge_shifts = data.edge_shifts.to(device=device, dtype=dtype).reshape(-1, 3)
            if edge_shifts.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    "edge_shifts must contain one Cartesian shift per edge; "
                    f"got {edge_shifts.shape[0]} shifts for {edge_index.shape[1]} edges."
                )
            cell_per_edge = cell[edge_batch]
            solve_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            cell_offsets = torch.linalg.solve(
                cell_per_edge.to(solve_dtype).transpose(1, 2),
                -edge_shifts.to(solve_dtype).unsqueeze(-1),
            ).squeeze(-1)
            rounded_offsets = cell_offsets.round()
            tolerance = 1e-10 if solve_dtype == torch.float64 else 1e-5
            if not torch.allclose(
                cell_offsets, rounded_offsets, atol=tolerance, rtol=tolerance
            ):
                max_error = (cell_offsets - rounded_offsets).abs().max().item()
                raise ValueError(
                    "edge_shifts are not integer combinations of their system cell; "
                    f"maximum fractional-offset error is {max_error:.3e}."
                )
            cell_offsets = rounded_offsets.to(dtype)
        else:
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=device, dtype=dtype
            )

        graph_charge = None
        graph_spin = None
        if hasattr(data, "graph_attr") and data.graph_attr is not None:
            graph_attr = data.graph_attr.to(device=device)
            if graph_attr.numel() != 2 * num_systems:
                raise ValueError(
                    "UMA expects graph_attr=[charge, spin] for each system; "
                    f"got shape {tuple(graph_attr.shape)} for "
                    f"num_systems={num_systems}."
                )
            graph_attr = graph_attr.reshape(num_systems, 2)
            graph_charge = graph_attr[:, 0].to(dtype=torch.long)
            graph_spin = graph_attr[:, 1].to(dtype=torch.long)

        if hasattr(data, "charge") and data.charge is not None:
            charge = data.charge.to(device=device, dtype=torch.long).view(-1)
        elif graph_charge is not None:
            charge = graph_charge
        else:
            charge = torch.zeros(num_systems, device=device, dtype=torch.long)
        if hasattr(data, "spin") and data.spin is not None:
            spin = data.spin.to(device=device, dtype=torch.long).view(-1)
        elif graph_spin is not None:
            spin = graph_spin
        else:
            spin = torch.zeros(num_systems, device=device, dtype=torch.long)

        return _UMADataDict(
            {
                "pos": data.pos,
                "atomic_numbers": atomic_numbers,
                "atomic_numbers_full": atomic_numbers,
                "batch": batch,
                "batch_full": batch,
                "natoms": natoms,
                "nedges": nedges,
                "cell": cell,
                "cell_offsets": cell_offsets,
                "pbc": pbc,
                "edge_index": edge_index,
                "charge": charge,
                "spin": spin,
            }
        )

    # --- HydraGNN Base hooks -------------------------------------------------

    def _embedding(self, data):
        """Run the full UMA backbone and slice the L=0 invariant feature."""
        # Base._embedding pre-builds edge_shifts / pbc bookkeeping. We
        # invoke it so any HydraGNN-side preprocessing (graph_attr
        # conditioning etc.) still runs, then discard the returned conv
        # args -- UMA does not consume them.
        super()._embedding(data)

        data_dict = self._build_data_dict(data)
        out = self.uma_backbone(data_dict)
        node_embedding = out["node_embedding"]
        # node_embedding: (N, (lmax+1)**2, sphere_channels)
        inv_node_feat = node_embedding[:, self._sph_l0_index, :]
        # Optional equivariant vector head: read genuine 3-vectors from the
        # L=1 irrep now (while node_embedding is in scope) and stash them for
        # forward() to slot into the designated node output head.
        if self.equivariant_vector_head is not None:
            self._equivariant_vector_cache = self.equivariant_vector_head(
                node_embedding
            )
        else:
            self._equivariant_vector_cache = None
        # Equivariant L=1 channel exists but is not currently consumed
        # by HydraGNN's standard MLP heads. Expose it via an empty tensor
        # for now to keep Base's signature stable.
        equiv_node_feat = inv_node_feat.new_zeros((inv_node_feat.shape[0], 0))
        conv_args: Dict[str, Any] = {}
        return inv_node_feat, equiv_node_feat, conv_args

    def forward(self, data):
        """Run ``Base.forward`` and, if enabled, overwrite the designated node
        head output with the rotation-equivariant vector prediction.

        ``Base.forward`` still evaluates the scalar MLP for that head slot; we
        replace its output with the equivariant vectors computed in
        ``_embedding``. This keeps the model output aligned with HydraGNN's
        head/loss configuration (the target must be declared as a 'node' head
        whose dimension is a multiple of 3).
        """
        out = super().forward(data)
        if self.equivariant_vector_head is None:
            return out

        vec = self._equivariant_vector_cache
        idx = self._vector_head_index
        if self.var_output:
            outputs, outputs_var = out
            outputs[idx] = vec
            # No aleatoric variance is predicted for the equivariant head.
            outputs_var[idx] = vec.new_zeros(vec.shape)
            return outputs, outputs_var
        out[idx] = vec
        return out

    def __str__(self):
        return "UMAStack"
