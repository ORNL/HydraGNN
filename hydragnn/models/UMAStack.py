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

Unlike :class:`AllScAIPStack`, which vendors its source under
``hydragnn/utils/model/allscaip``, UMA depends on a much larger set of
specialized SO(3) / SO(2) primitives, Wigner-D tables, and rotation
machinery that we deliberately do **not** vendor. Instead we declare
``fairchem-core`` as an *optional* HydraGNN dependency
(``requirements-optional.txt``) and import the upstream backbone class
lazily inside :meth:`UMAStack._init_conv`.

If ``fairchem-core`` is not installed at runtime, instantiating
``UMAStack`` raises a clear :class:`ImportError` with installation
instructions; HydraGNN itself continues to import and run all other
backbones normally.

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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Identity, ModuleList

from hydragnn.models.Base import Base


_FAIRCHEM_INSTALL_HINT = (
    "UMAStack requires the optional 'fairchem-core' package, which "
    "provides the eSCNMDBackbone implementation. Install it with:\n"
    "    pip install fairchem-core\n"
    "(see requirements-optional.txt)."
)


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
        periodic_boundary_conditions: bool = False,
        *args,
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
        # Force num_conv_layers=1 for the Base forward loop. The actual
        # UMA depth lives in self.uma_num_layers.
        kwargs["num_conv_layers"] = 1

        super().__init__(input_args, conv_args, *args, **kwargs)

    def _init_conv(self):
        try:
            from fairchem.core.models.uma.escn_md import eSCNMDBackbone
        except ImportError as exc:  # pragma: no cover - exercised at runtime
            raise ImportError(_FAIRCHEM_INSTALL_HINT) from exc

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

        self.uma_backbone = eSCNMDBackbone(**backbone_cfg)
        # Cache lmax once -- UMA's node_embedding has shape
        # (N, (lmax+1)**2, sphere_channels). We pull L=0 (index 0) as
        # the invariant scalar feature for HydraGNN's standard decoders.
        self._sph_l0_index = 0

        # HydraGNN's Base.forward iterates over (graph_convs, feature_layers)
        # exactly num_conv_layers times. We forced num_conv_layers=1 in
        # __init__ and supply identity placeholders so the loop is a no-op.
        self.graph_convs = ModuleList([_IdentityConv()])
        self.feature_layers = ModuleList([Identity()])

    # --- HydraGNN data adapter -----------------------------------------------

    def _build_data_dict(self, data) -> Dict[str, Any]:
        """Translate a HydraGNN PyG ``Data`` batch to UMA's data_dict.

        UMA's ``forward(data_dict)`` expects FairChem ``AtomicData``-style
        dictionary access; we hand-build the minimum subset required when
        ``otf_graph=False`` and dataset / charge / spin embeddings are
        either disabled or defaulted.
        """
        device = data.pos.device
        dtype = torch.get_default_dtype()

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

        # Edge bookkeeping. With otf_graph=False, UMA expects edge_index
        # plus per-edge cell_offsets and per-system nedges.
        edge_index = data.edge_index
        nedges = torch.bincount(batch[edge_index[0]], minlength=num_systems).long()
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=device, dtype=dtype)

        # Charge / spin: per-system scalars. Default to neutral / singlet.
        if hasattr(data, "charge") and data.charge is not None:
            charge = data.charge.to(device=device, dtype=torch.long).view(-1)
        else:
            charge = torch.zeros(num_systems, device=device, dtype=torch.long)
        if hasattr(data, "spin") and data.spin is not None:
            spin = data.spin.to(device=device, dtype=torch.long).view(-1)
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
        # Equivariant L=1 channel exists but is not currently consumed
        # by HydraGNN's standard MLP heads. Expose it via an empty tensor
        # for now to keep Base's signature stable.
        equiv_node_feat = inv_node_feat.new_zeros((inv_node_feat.shape[0], 0))
        conv_args: Dict[str, Any] = {}
        return inv_node_feat, equiv_node_feat, conv_args

    def __str__(self):
        return "UMAStack"
