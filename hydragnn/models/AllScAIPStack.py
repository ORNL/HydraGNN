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
# Adapted from:
#   FairChem AllScAIP (All-to-all Scalable Attention Interatomic Potential)
#   https://github.com/FAIR-Chem/fairchem
#   Distributed under the MIT License.
###############################################################################
"""
HydraGNN integration of FairChem's AllScAIP model.

The vendored FairChem code lives under
``hydragnn.utils.model.allscaip`` (and its EScAIP utility deps under
``hydragnn.utils.model.escaip.utils``). This stack wraps the vendored
:class:`AllScAIPBackbone` so it composes with HydraGNN's ``Base`` model
machinery (multi-head decoders, graph conditioning, training loop).

Design summary
--------------
AllScAIP is a monolithic transformer-style backbone: it performs its
own kNN radius-graph construction, runs an input block + N transformer
blocks, and emits a per-node hidden representation. To plug it into
HydraGNN we:

* run the **entire** AllScAIP backbone inside ``_embedding(...)``,
* expose the result as the invariant node feature consumed by
  HydraGNN's standard decoder pipeline, and
* register a single identity placeholder convolution so the
  ``Base.forward`` loop becomes a no-op.

The number of AllScAIP transformer blocks is controlled by the
standard HydraGNN ``num_conv_layers`` architecture flag (same key used
by every other backbone). Internally we capture it as the AllScAIP
depth and then force Base's per-layer forward loop to a single
no-op iteration.

Equivariance
------------
**AllScAIP is NOT an e3nn-style equivariant model.** It is a plain
scalar transformer over a kNN graph:

* Hidden features are scalar tensors ``(N, hidden_dim)`` and padded
  kNN tensors ``(N, k_max, hidden_dim)``. There is no
  ``e3nn.o3.Irreps`` decomposition (no ``0e + 1o + 2e`` channels) and
  no Clebsch-Gordan tensor products in the message-passing path.
* Spherical harmonics are used as **input features only** (computed
  via ``e3nn.o3._spherical_harmonics`` of edge directions and fed
  into the attention as additional scalar channels via
  ``frequency_vectors`` / ``node_sincx_matrix``). They lose their
  irrep semantics on the first ``Linear`` / ``LayerNorm`` inside
  :class:`InputBlock`.
* For graph-level scalar targets (e.g. energy) the prediction is
  **rotation/translation invariant** because all SH magnitudes and
  pairwise distances fed in are themselves invariants -- same kind of
  invariance SchNet provides via its distance features. But internal
  representations are NOT equivariant.

For genuinely equivariant predictions of vectorial / tensorial
quantities (forces from auto-grad excepted, but also dipoles,
stresses, ...), use :class:`EGNN`, :class:`PAINN`, :class:`MACE`, or
:class:`PNAEqStack` instead. The wrapper therefore always passes
``equivariance=False`` through to ``Base``, and AllScAIP is
deliberately excluded from the equivariant CI test families
(``pytest_train_equivariant_model`` / ``*_lengths`` /
``*_vectoroutput`` / ``*_lengths_global_attention``) in
``tests/test_graphs.py`` and ``tests/test_graphs_graphattr.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Identity, ModuleList

from hydragnn.models.Base import Base
from hydragnn.utils.model.allscaip.AllScAIP import AllScAIPBackbone


@dataclass
class _BatchStats:
    """Container returned by the FairChem-style ``get_batch_stats`` call."""

    slices: Dict[str, torch.Tensor]
    cumsum: Dict[str, torch.Tensor]
    cat_dims: Dict[str, int]
    natoms_list: List[int]


class _FairChemAdapter:
    """Lightweight FairChem ``AtomicData`` look-alike.

    The vendored AllScAIP graph-construction and preprocessing code only
    accesses a small set of attributes / methods on the input object.
    We expose exactly those (without inheriting from any FairChem class)
    so HydraGNN can keep using PyG ``Data`` batches.
    """

    def __init__(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
    ) -> None:
        self.pos = pos
        self.atomic_numbers = atomic_numbers
        self.batch = batch
        self.num_graphs = num_graphs
        self.cell = cell
        self.pbc = pbc
        self.charge = charge
        self.spin = spin
        self.num_nodes = int(pos.shape[0])
        # AllScAIP code stamps these in the entry-point forward:
        self.atomic_numbers_full = atomic_numbers
        self.batch_full = batch

    # FairChem ``AtomicData`` exposes these as attributes via dict-style access.
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get_batch_stats(self):
        # natoms per graph in the batch
        natoms_list = (
            torch.bincount(self.batch, minlength=self.num_graphs).tolist()
        )
        # cumulative offsets for ``pos`` -> matches what FairChem's
        # ``slices["pos"]`` provides downstream.
        offsets = torch.zeros(
            self.num_graphs + 1, dtype=torch.long, device=self.pos.device
        )
        offsets[1:] = torch.cumsum(
            torch.tensor(natoms_list, dtype=torch.long, device=self.pos.device),
            dim=0,
        )
        slices = {"pos": offsets}
        # ``cumsum`` and ``cat_dims`` are not consumed by the
        # vendored radius-graph code; pass empty dicts so downstream
        # ``None`` checks in biknn_radius_graph see truthy values.
        return slices, {}, {}, natoms_list


class _IdentityConv(torch.nn.Module):
    """No-op stand-in used in place of HydraGNN's standard graph_conv.

    AllScAIP runs its full message-passing stack inside ``_embedding``,
    so the standard ``Base.forward`` conv loop has nothing to do. We
    register a single instance and force ``num_conv_layers = 1`` for
    this stack (see :class:`AllScAIPStack.__init__`).
    """

    def forward(self, inv_node_feat, equiv_node_feat, **_kwargs):
        return inv_node_feat, equiv_node_feat


class AllScAIPStack(Base):
    """HydraGNN wrapper around FairChem's :class:`AllScAIPBackbone`.

    Notable arguments
    -----------------
    radius : float
        Cutoff distance (Å) for the internal differentiable kNN graph.
    max_neighbours : int
        ``knn_k`` for the kNN graph.
    allscaip_num_heads : int
        Attention head count. ``hidden_dim`` must be divisible by this.
    allscaip_freq_list : list[int] | None
        Per-degree frequency repeats for spherical-harmonic attention
        masks. Must sum to ``hidden_dim // allscaip_num_heads``. When
        ``None`` we synthesize a single-degree (l=0) list of that size,
        which matches the constraint trivially at the cost of dropping
        higher-l directional info in the attention bias.
    allscaip_atten_name : {"math", "memory_efficient", "flash"}
        SDPA backend selection. ``"math"`` is the safe default for
        gradient-based forces / mixed CPU/GPU runs.
    """

    def __init__(
        self,
        input_args,
        conv_args,
        radius: float,
        max_neighbours: int,
        allscaip_num_heads: int = 8,
        allscaip_freq_list: Optional[List[int]] = None,
        allscaip_atten_name: str = "math",
        allscaip_use_node_path: bool = True,
        allscaip_use_sincx_mask: bool = True,
        allscaip_use_freq_mask: bool = True,
        allscaip_max_num_elements: int = 119,
        *args,
        **kwargs,
    ):
        # --- Stash AllScAIP-specific args before calling Base.__init__ ---
        self.radius = radius
        self.max_neighbours = max_neighbours
        # Re-use HydraGNN's standard num_conv_layers as the AllScAIP
        # transformer depth. Default to 4 if the caller did not set it.
        self.allscaip_num_layers = kwargs.get("num_conv_layers", 4)
        self.allscaip_num_heads = allscaip_num_heads
        self.allscaip_freq_list = allscaip_freq_list
        self.allscaip_atten_name = allscaip_atten_name
        self.allscaip_use_node_path = allscaip_use_node_path
        self.allscaip_use_sincx_mask = allscaip_use_sincx_mask
        self.allscaip_use_freq_mask = allscaip_use_freq_mask
        self.allscaip_max_num_elements = allscaip_max_num_elements

        # Capture the HydraGNN activation-function string so the
        # AllScAIP backbone can be configured with the same activation
        # used by the rest of the model. Base only keeps the callable,
        # not the string, so we sniff it out of args/kwargs here.
        # Position in args matches Base.__init__'s signature:
        # (input_dim, hidden_dim, output_dim, pe_dim, global_attn_engine,
        #  global_attn_type, global_attn_heads, output_type, config_heads,
        #  activation_function_type, ...)
        if "activation_function_type" in kwargs:
            hydragnn_act = kwargs["activation_function_type"]
        elif len(args) >= 10:
            hydragnn_act = args[9]
        else:
            hydragnn_act = "gelu"
        # Map HydraGNN activation names to AllScAIP-supported ones
        # (AllScAIP allows: squared_relu, gelu, leaky_relu, relu, smelu,
        # star_relu). Unknown names fall back to "gelu" (AllScAIP default).
        _ACT_MAP = {
            "relu": "relu",
            "lrelu_01": "leaky_relu",
            "lrelu_025": "leaky_relu",
            "lrelu_05": "leaky_relu",
            "gelu": "gelu",
        }
        self.allscaip_activation = _ACT_MAP.get(hydragnn_act, "gelu")

        # AllScAIP performs its own graph construction, so HydraGNN should
        # not pass edge_attr through it. Mark the model as edge-free for
        # Base.__init__'s edge handling.
        self.is_edge_model = False
        # Force num_conv_layers=1 for the Base forward loop. The actual
        # AllScAIP depth lives in self.allscaip_num_layers.
        kwargs["num_conv_layers"] = 1

        super().__init__(input_args, conv_args, *args, **kwargs)

    def _init_conv(self):
        # Build the AllScAIP backbone with HydraGNN-derived configuration.
        head_dim = self.hidden_dim // self.allscaip_num_heads
        if self.hidden_dim % self.allscaip_num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by allscaip_num_heads "
                f"(got hidden_dim={self.hidden_dim}, "
                f"allscaip_num_heads={self.allscaip_num_heads})."
            )

        if self.allscaip_freq_list is None:
            freq_list = [head_dim]
        else:
            freq_list = list(self.allscaip_freq_list)
            if sum(freq_list) != head_dim:
                raise ValueError(
                    "allscaip_freq_list must sum to hidden_dim // "
                    f"allscaip_num_heads (= {head_dim}); got {freq_list}."
                )

        backbone_cfg: Dict[str, Any] = {
            # GlobalConfigs
            "regress_forces": False,
            "direct_forces": False,
            "regress_stress": False,
            "hidden_size": self.hidden_dim,
            "num_layers": self.allscaip_num_layers,
            "activation": self.allscaip_activation,
            "use_residual_scaling": True,
            "use_node_path": self.allscaip_use_node_path,
            "dataset_list": [],
            # MolecularGraphConfigs
            "max_num_elements": self.allscaip_max_num_elements,
            "max_radius": float(self.radius),
            "knn_k": int(self.max_neighbours),
            "knn_soft": True,
            "knn_sigmoid_scale": 0.2,
            "knn_lse_scale": 0.1,
            "distance_function": "gaussian",
            "use_envelope": True,
            # GraphNeuralNetworksConfigs
            "atten_name": self.allscaip_atten_name,
            "atten_num_heads": self.allscaip_num_heads,
            "freequency_list": freq_list,
            "use_freq_mask": self.allscaip_use_freq_mask,
            "use_sincx_mask": self.allscaip_use_sincx_mask,
            # RegularizationConfigs - rely on dataclass defaults
        }

        self.allscaip_backbone = AllScAIPBackbone(**backbone_cfg)

        # HydraGNN's Base.forward iterates over (graph_convs, feature_layers)
        # exactly num_conv_layers times. We forced num_conv_layers=1 in
        # __init__ and supply identity placeholders so the loop is a no-op.
        self.graph_convs = ModuleList([_IdentityConv()])
        self.feature_layers = ModuleList([Identity()])

    # --- HydraGNN data adapter -----------------------------------------------

    def _build_adapter(self, data) -> _FairChemAdapter:
        """Map a HydraGNN PyG ``Data`` batch to a FairChem-style object."""
        device = data.pos.device

        # Atomic numbers: HydraGNN typically stores them in data.x[:, 0]
        # (first node feature column). Allow an explicit override via
        # data.atomic_numbers if the dataset already provides it.
        if hasattr(data, "atomic_numbers") and data.atomic_numbers is not None:
            atomic_numbers = data.atomic_numbers.long()
        else:
            atomic_numbers = data.x[:, 0].long()

        batch = (
            data.batch
            if data.batch is not None
            else torch.zeros(data.pos.shape[0], dtype=torch.long, device=device)
        )
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Cell / PBC: optional. Default to a zero cell with PBC off.
        if hasattr(data, "cell") and data.cell is not None:
            cell = data.cell.to(device=device, dtype=torch.get_default_dtype())
            if cell.dim() == 2:
                cell = cell.unsqueeze(0).expand(num_graphs, 3, 3).contiguous()
        else:
            cell = torch.zeros(
                num_graphs, 3, 3, device=device, dtype=torch.get_default_dtype()
            )
        if hasattr(data, "pbc") and data.pbc is not None:
            pbc = data.pbc.to(device=device, dtype=torch.bool)
            if pbc.dim() == 1:
                pbc = pbc.unsqueeze(0).expand(num_graphs, 3).contiguous()
        else:
            pbc = torch.zeros(num_graphs, 3, device=device, dtype=torch.bool)

        # Charge / spin: per-graph scalars. Pull from data if present,
        # otherwise default to neutral / singlet.
        if hasattr(data, "charge") and data.charge is not None:
            charge = data.charge.to(device=device, dtype=torch.long).view(-1)
        else:
            charge = torch.zeros(num_graphs, device=device, dtype=torch.long)
        if hasattr(data, "spin") and data.spin is not None:
            spin = data.spin.to(device=device, dtype=torch.long).view(-1)
        else:
            spin = torch.zeros(num_graphs, device=device, dtype=torch.long)

        return _FairChemAdapter(
            pos=data.pos,
            atomic_numbers=atomic_numbers,
            batch=batch,
            num_graphs=num_graphs,
            cell=cell,
            pbc=pbc,
            charge=charge,
            spin=spin,
        )

    # --- HydraGNN Base hooks -------------------------------------------------

    def _embedding(self, data):
        """Run the full AllScAIP backbone and return invariant node features."""
        # NOTE: We deliberately do NOT call ``super()._embedding(data)``.
        # Base's implementation expects ``data.edge_index`` to already
        # exist (it pre-pads ``edge_shifts`` from it), but AllScAIP builds
        # its own kNN radius graph internally and HydraGNN datasets used
        # with this stack should not be required to carry an edge_index.

        adapter = self._build_adapter(data)
        results = self.allscaip_backbone(adapter)
        # ``node_reps`` is shape [N, hidden_dim] (no padding — AllScAIP
        # always runs unpadded under HydraGNN).
        inv_node_feat = results["node_reps"]
        # AllScAIP is not equivariant; provide an empty equiv tensor so
        # Base.forward signatures still match.
        equiv_node_feat = inv_node_feat.new_zeros((inv_node_feat.shape[0], 0))
        # No edge-level args are needed by the identity placeholder conv.
        conv_args: Dict[str, Any] = {}
        return inv_node_feat, equiv_node_feat, conv_args

    def __str__(self):
        return "AllScAIPStack"
