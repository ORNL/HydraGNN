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
# Adapted from:
#   FairChem AllScAIP (All-to-all Scalable Attention Interatomic Potential)
#   https://github.com/FAIR-Chem/fairchem
#   Distributed under the MIT License.
###############################################################################
"""
HydraGNN integration of FairChem's AllScAIP model.

The vendored FairChem code lives under ``hydragnn.utils.model.allscaip``
(and its EScAIP utility deps under ``hydragnn.utils.model.escaip.utils``).
This stack wraps the vendored :class:`AllScAIPBackbone` so it composes with
HydraGNN's ``Base`` model machinery (multi-head decoders, graph
conditioning, and the interatomic-potential training loop).

Design summary
--------------
AllScAIP is a monolithic transformer-style backbone: it performs its own
differentiable kNN radius-graph construction, runs an input block + N
transformer blocks, and emits a per-node scalar hidden representation. To
plug it into HydraGNN we:

* run the **entire** AllScAIP backbone inside ``_embedding(...)``,
* expose the result as the invariant node feature consumed by HydraGNN's
  standard decoder pipeline, and
* register a single identity placeholder convolution so the
  ``Base.forward`` per-layer loop becomes a no-op.

The number of AllScAIP transformer blocks is controlled by the standard
HydraGNN ``num_conv_layers`` architecture flag (same key used by every
other backbone). Internally we capture it as the AllScAIP depth and then
force Base's per-layer forward loop to a single no-op iteration.

Hyperparameter mapping
----------------------
HydraGNN's standard ``Architecture`` keys are reused wherever the
semantics overlap; AllScAIP-specific keys keep the ``allscaip_`` prefix.

============================  ==============================  ===============
HydraGNN key                  AllScAIP config argument        Notes
============================  ==============================  ===============
``radius``                    ``max_radius``                  kNN cutoff (A).
``max_neighbours``            ``knn_k``                       kNN degree.
``hidden_dim``                ``hidden_size``                 Scalar width.
``num_conv_layers``           ``num_layers``                  Transformer depth.
``activation_function``       (mapped to ``activation``)      See ``_ACT_MAP``.
============================  ==============================  ===============

AllScAIP-specific keys (no HydraGNN equivalent):

======================================  ====================================
Key                                     Purpose
======================================  ====================================
``allscaip_num_heads``                  Attention head count. ``hidden_dim``
                                        must be divisible by this.
``allscaip_freq_list``                  Per-degree spherical-harmonic
                                        frequency repeats. Must sum to
                                        ``hidden_dim // allscaip_num_heads``
                                        (defaults to a single l=0 bucket).
``allscaip_atten_name``                 SDPA backend: ``"math"`` (safe for
                                        autograd forces / CPU), or
                                        ``"memory_efficient"`` / ``"flash"``
                                        for direct-force GPU inference.
``allscaip_use_node_path``              Enable the global node-attention path.
``allscaip_use_sincx_mask``             sinc(x) node-attention distance mask.
``allscaip_use_freq_mask``              Frequency-vector neighbor mask.
``allscaip_max_num_elements``           Z embedding table size.
``allscaip_knn_soft``                   Differentiable soft-kNN gate. **Keep
                                        ``True`` for gradient-based force
                                        training** (positions must flow
                                        through the graph construction).
``allscaip_distance_function``          Radial basis: ``"gaussian"`` /
                                        ``"sigmoid"`` / ``"linearsigmoid"`` /
                                        ``"silu"``.
``allscaip_normalization``              Block norm: ``"rmsnorm"`` /
                                        ``"layernorm"`` / ``"skip"``.
``allscaip_mlp_dropout``                Dropout on the MLP path.
``allscaip_atten_dropout``              Dropout on attention weights.
``allscaip_use_residual_scaling``       Learnable per-layer residual scaling.
``allscaip_regress_stress``             Track cell displacement so stress can
                                        be recovered by autograd.
``allscaip_dataset_list``               List of dataset names for
                                        multi-dataset routing. When non-empty
                                        the backbone adds a per-graph dataset
                                        embedding and the wrapper reads a
                                        per-graph label from ``data.dataset``.
======================================  ====================================

Force training
--------------
AllScAIP is a scalar (invariant) energy model, so forces are obtained the
same way HydraGNN handles every invariant potential: by differentiating the
predicted energy with respect to atomic positions
(:meth:`energy_force_loss` sets ``data.pos.requires_grad = True`` and takes
the autograd gradient). For those gradients to be non-trivial the backbone's
graph construction must itself be differentiable in the positions, which is
why ``allscaip_knn_soft`` defaults to ``True`` (a hard top-k selection would
detach the position gradient at the neighbor boundaries). The wrapper
therefore leaves ``regress_forces`` / ``direct_forces`` off and relies on
HydraGNN's autograd force path; ``allscaip_atten_name="math"`` is the safe
SDPA backend for that double-backward.

HydraGNN integrates the FAIR-Chem backbone representation and applies its
configured output normalization, then uses HydraGNN's generic multi-task
decoders. Those decoders intentionally do not reproduce every FAIR-Chem
energy/force/stress head or its task-specific reductions.

Equivariance
------------
**AllScAIP is NOT an e3nn-style equivariant model.** Hidden features are
scalar tensors; spherical harmonics of edge directions enter only as
invariant input channels and lose their irrep semantics on the first
``Linear`` / ``LayerNorm``. Graph-level scalar predictions (energy) are
rotation/translation invariant, and autograd forces derived from an
invariant energy are correctly equivariant, but the internal
representations are not. The wrapper always passes ``equivariance=False``
through to :class:`Base`, and AllScAIP is deliberately excluded from the
equivariant CI test families.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Identity, ModuleList

from hydragnn.utils.model.escaip.utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
)

from hydragnn.models.Base import Base
from hydragnn.utils.model.allscaip.AllScAIP import AllScAIPBackbone

# Map HydraGNN activation-function names to AllScAIP-supported ones.
# AllScAIP accepts: squared_relu, gelu, leaky_relu, relu, smelu, star_relu.
# Unknown names fall back to "gelu" (the AllScAIP default).
_ACT_MAP = {
    "relu": "relu",
    "lrelu_01": "leaky_relu",
    "lrelu_025": "leaky_relu",
    "lrelu_05": "leaky_relu",
    "gelu": "gelu",
}


def _resolve_frequency_list(
    head_dim: int, explicit: Optional[List[int]], use_freq_mask: bool
) -> List[int]:
    """Resolve the AllScAIP l-frequency multiplicities.

    FAIR-Chem defines a canonical spectrum only for its 64-wide attention
    heads. Other masked head sizes must be configured explicitly so they do
    not silently degrade to an all-l=0 representation.
    """
    if explicit is not None:
        freq_list = list(explicit)
    elif not use_freq_mask:
        freq_list = [head_dim]
    elif head_dim == 64:
        freq_list = [20, 10, 4, 10, 20]
    else:
        raise ValueError(
            "Frequency masking requires allscaip_freq_list for head_dim "
            f"{head_dim}; FAIR-Chem only defines the canonical automatic "
            "default [20, 10, 4, 10, 20] for head_dim=64."
        )
    if sum(freq_list) != head_dim:
        raise ValueError(
            f"allscaip_freq_list must sum to head_dim (= {head_dim}); "
            f"got {freq_list}."
        )
    return freq_list


class _FairChemAdapter:
    """Lightweight FairChem ``AtomicData`` look-alike.

    The vendored AllScAIP graph-construction and preprocessing code only
    accesses a small set of attributes / methods on the input object. We
    expose exactly those (without inheriting from any FairChem class) so
    HydraGNN can keep using PyG ``Data`` batches.
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
        dataset: Optional[torch.Tensor] = None,
    ) -> None:
        self.pos = pos
        self.atomic_numbers = atomic_numbers
        self.batch = batch
        self.num_graphs = num_graphs
        self.cell = cell
        self.pbc = pbc
        self.charge = charge
        self.spin = spin
        self.dataset = dataset
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
        natoms_list = torch.bincount(self.batch, minlength=self.num_graphs).tolist()
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
        # ``cumsum`` and ``cat_dims`` are not consumed by the vendored
        # radius-graph code; pass empty dicts so downstream ``None`` checks
        # in biknn_radius_graph see truthy values.
        return slices, {}, {}, natoms_list


class _IdentityConv(torch.nn.Module):
    """No-op stand-in used in place of HydraGNN's standard graph_conv.

    AllScAIP runs its full message-passing stack inside ``_embedding``, so
    the standard ``Base.forward`` conv loop has nothing to do. We register a
    single instance and force ``num_conv_layers = 1`` for this stack (see
    :class:`AllScAIPStack.__init__`).
    """

    def forward(self, inv_node_feat, equiv_node_feat, **_kwargs):
        return inv_node_feat, equiv_node_feat


class AllScAIPStack(Base):
    """HydraGNN wrapper around FairChem's :class:`AllScAIPBackbone`.

    See the module docstring for the full hyperparameter mapping, the
    force-training contract, and the (non-)equivariance discussion.
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
        allscaip_knn_soft: bool = True,
        allscaip_distance_function: str = "gaussian",
        allscaip_normalization: str = "rmsnorm",
        allscaip_mlp_dropout: float = 0.0,
        allscaip_atten_dropout: float = 0.0,
        allscaip_use_residual_scaling: bool = True,
        allscaip_regress_stress: bool = False,
        allscaip_dataset_list: Optional[List[str]] = None,
        allscaip_use_chunked_graph: bool = False,
        allscaip_graph_chunk_size: int = 512,
        allscaip_knn_use_low_mem: bool = True,
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
        self.allscaip_knn_soft = bool(allscaip_knn_soft)
        self.allscaip_distance_function = allscaip_distance_function
        self.allscaip_normalization = allscaip_normalization
        self.allscaip_mlp_dropout = float(allscaip_mlp_dropout)
        self.allscaip_atten_dropout = float(allscaip_atten_dropout)
        self.allscaip_use_residual_scaling = bool(allscaip_use_residual_scaling)
        self.allscaip_regress_stress = bool(allscaip_regress_stress)
        self.allscaip_use_chunked_graph = bool(allscaip_use_chunked_graph)
        self.allscaip_graph_chunk_size = int(allscaip_graph_chunk_size)
        if self.allscaip_graph_chunk_size <= 0:
            raise ValueError("allscaip_graph_chunk_size must be positive")
        self.allscaip_knn_use_low_mem = bool(allscaip_knn_use_low_mem)
        # Dataset routing: an ordered list of dataset names. Non-empty
        # enables the backbone's per-graph dataset embedding, and the
        # wrapper maps ``data.dataset`` labels to indices into this list.
        self.allscaip_dataset_list = (
            list(allscaip_dataset_list) if allscaip_dataset_list else []
        )
        self._dataset_name_to_index = {
            name: idx for idx, name in enumerate(self.allscaip_dataset_list)
        }

        # Capture the HydraGNN activation-function string so the AllScAIP
        # backbone uses the same activation as the rest of the model. Base
        # only keeps the callable, not the string, so we sniff it out of
        # args/kwargs here. Position in args matches Base.__init__'s
        # signature (activation_function_type is the 10th positional).
        if "activation_function_type" in kwargs:
            hydragnn_act = kwargs["activation_function_type"]
        elif len(args) >= 10:
            hydragnn_act = args[9]
        else:
            hydragnn_act = "gelu"
        self.allscaip_activation = _ACT_MAP.get(hydragnn_act, "gelu")

        # AllScAIP performs its own graph construction, so HydraGNN should
        # not pass edge_attr through it. Mark the model as edge-free for
        # Base.__init__'s edge handling.
        self.is_edge_model = False
        # AllScAIP completes message passing in _embedding.  Avoid an extra
        # generic HydraGNN activation after the identity placeholder conv.
        self.skip_post_conv_processing = True
        # Force num_conv_layers=1 for the Base forward loop. The actual
        # AllScAIP depth lives in self.allscaip_num_layers.
        kwargs["num_conv_layers"] = 1

        super().__init__(input_args, conv_args, *args, **kwargs)

    def _init_conv(self):
        # Build the AllScAIP backbone with HydraGNN-derived configuration.
        if self.hidden_dim % self.allscaip_num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by allscaip_num_heads "
                f"(got hidden_dim={self.hidden_dim}, "
                f"allscaip_num_heads={self.allscaip_num_heads})."
            )
        head_dim = self.hidden_dim // self.allscaip_num_heads

        freq_list = _resolve_frequency_list(
            head_dim, self.allscaip_freq_list, self.allscaip_use_freq_mask
        )

        backbone_cfg: Dict[str, Any] = {
            # GlobalConfigs
            "regress_forces": False,
            "direct_forces": False,
            "regress_stress": self.allscaip_regress_stress,
            "hidden_size": self.hidden_dim,
            "num_layers": self.allscaip_num_layers,
            "activation": self.allscaip_activation,
            "use_residual_scaling": self.allscaip_use_residual_scaling,
            "use_node_path": self.allscaip_use_node_path,
            "dataset_list": self.allscaip_dataset_list,
            # MolecularGraphConfigs
            "max_num_elements": self.allscaip_max_num_elements,
            "max_radius": float(self.radius),
            "knn_k": int(self.max_neighbours),
            "knn_soft": self.allscaip_knn_soft,
            "knn_sigmoid_scale": 0.2,
            "knn_lse_scale": 0.1,
            "distance_function": self.allscaip_distance_function,
            "use_envelope": True,
            "knn_use_low_mem": self.allscaip_knn_use_low_mem,
            "use_chunked_graph": self.allscaip_use_chunked_graph,
            "graph_chunk_size": self.allscaip_graph_chunk_size,
            # GraphNeuralNetworksConfigs
            "atten_name": self.allscaip_atten_name,
            "atten_num_heads": self.allscaip_num_heads,
            "freequency_list": freq_list,
            "use_freq_mask": self.allscaip_use_freq_mask,
            "use_sincx_mask": self.allscaip_use_sincx_mask,
            # RegularizationConfigs
            "normalization": self.allscaip_normalization,
            "mlp_dropout": self.allscaip_mlp_dropout,
            "atten_dropout": self.allscaip_atten_dropout,
        }

        self.allscaip_backbone = AllScAIPBackbone(**backbone_cfg)
        # FAIR-Chem AllScAIP heads normalize backbone node representations
        # before their prediction FFN. HydraGNN keeps its generic task heads,
        # but mirrors that normalization at this adapter boundary.
        self.allscaip_output_norm = get_normalization_layer(
            NormalizationType(self.allscaip_normalization)
        )(self.hidden_dim)

        # HydraGNN's Base.forward iterates over (graph_convs, feature_layers)
        # exactly num_conv_layers times. We forced num_conv_layers=1 in
        # __init__ and supply identity placeholders so the loop is a no-op.
        self.graph_convs = ModuleList([_IdentityConv()])
        self.feature_layers = ModuleList([Identity()])

    # --- HydraGNN data adapter -----------------------------------------------

    def _resolve_dataset_index(
        self, data, num_graphs: int, device
    ) -> Optional[torch.Tensor]:
        """Return a per-graph dataset index tensor, or ``None`` if routing off.

        When ``allscaip_dataset_list`` is empty the backbone has no dataset
        embedding and we return ``None``. Otherwise we accept either an
        integer index tensor (used verbatim after a range check) or a list of
        dataset-name strings (mapped through ``allscaip_dataset_list``); a
        missing ``data.dataset`` defaults every graph to index 0.
        """
        if not self.allscaip_dataset_list:
            return None

        raw = getattr(data, "dataset", None)
        num_datasets = len(self.allscaip_dataset_list)
        if raw is None:
            return torch.zeros(num_graphs, dtype=torch.long, device=device)

        if isinstance(raw, torch.Tensor):
            index = raw.to(device=device, dtype=torch.long).view(-1)
        elif isinstance(raw, (list, tuple)):
            try:
                index = torch.tensor(
                    [self._dataset_name_to_index[str(name)] for name in raw],
                    dtype=torch.long,
                    device=device,
                )
            except KeyError as exc:
                raise ValueError(
                    f"Unknown dataset label {exc.args[0]!r}; expected one of "
                    f"{self.allscaip_dataset_list}."
                ) from exc
        else:
            index = torch.full(
                (num_graphs,),
                self._dataset_name_to_index.get(str(raw), 0),
                dtype=torch.long,
                device=device,
            )

        if index.numel() == 1 and num_graphs > 1:
            index = index.expand(num_graphs).contiguous()
        if int(index.min()) < 0 or int(index.max()) >= num_datasets:
            raise ValueError(
                f"data.dataset index out of range for dataset_list of length "
                f"{num_datasets}: got min/max "
                f"{int(index.min())}/{int(index.max())}."
            )
        return index

    def _build_adapter(self, data) -> _FairChemAdapter:
        """Map a HydraGNN PyG ``Data`` batch to a FairChem-style object."""
        device = data.pos.device
        dtype = data.pos.dtype

        # Atomic numbers: HydraGNN typically stores them in data.x[:, 0]
        # (first node feature column). Allow an explicit override via
        # data.atomic_numbers if the dataset already provides it.
        if hasattr(data, "atomic_numbers") and data.atomic_numbers is not None:
            atomic_numbers = data.atomic_numbers.long().view(-1)
        else:
            atomic_numbers = data.x[:, 0].long().view(-1)

        batch = (
            data.batch
            if data.batch is not None
            else torch.zeros(data.pos.shape[0], dtype=torch.long, device=device)
        )
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Cell / PBC: optional. Default to a zero cell with PBC off.
        if hasattr(data, "cell") and data.cell is not None:
            cell = data.cell.to(device=device, dtype=dtype)
            if cell.dim() == 2 and cell.shape == (3, 3):
                cell = cell.unsqueeze(0).expand(num_graphs, 3, 3).contiguous()
            elif cell.dim() == 2 and cell.shape[0] == 3 * num_graphs:
                # PyG batching of per-graph (3, 3) cells -> (3*B, 3).
                cell = cell.view(num_graphs, 3, 3).contiguous()
            elif cell.dim() == 3:
                cell = cell.contiguous()
            else:
                raise ValueError(
                    f"Unexpected cell shape {tuple(cell.shape)} for "
                    f"num_graphs={num_graphs}."
                )
        else:
            cell = torch.zeros(num_graphs, 3, 3, device=device, dtype=dtype)
        if hasattr(data, "pbc") and data.pbc is not None:
            pbc = data.pbc.to(device=device, dtype=torch.bool)
            if pbc.dim() == 1 and pbc.numel() == 3:
                pbc = pbc.unsqueeze(0).expand(num_graphs, 3).contiguous()
            elif pbc.dim() == 1 and pbc.numel() == 3 * num_graphs:
                # PyG batching of per-graph (3,) pbc -> (3*B,).
                pbc = pbc.view(num_graphs, 3).contiguous()
            elif pbc.dim() == 2:
                pbc = pbc.contiguous()
            else:
                raise ValueError(
                    f"Unexpected pbc shape {tuple(pbc.shape)} for "
                    f"num_graphs={num_graphs}."
                )
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

        dataset = self._resolve_dataset_index(data, num_graphs, device)

        return _FairChemAdapter(
            pos=data.pos,
            atomic_numbers=atomic_numbers,
            batch=batch,
            num_graphs=num_graphs,
            cell=cell,
            pbc=pbc,
            charge=charge,
            spin=spin,
            dataset=dataset,
        )

    # --- HydraGNN Base hooks -------------------------------------------------

    def _embedding(self, data):
        """Run the full AllScAIP backbone and return invariant node features."""
        # NOTE: We deliberately do NOT call ``super()._embedding(data)``.
        # Base's implementation expects ``data.edge_index`` to already exist
        # (it pre-pads ``edge_shifts`` from it), but AllScAIP builds its own
        # kNN radius graph internally and HydraGNN datasets used with this
        # stack should not be required to carry an edge_index.

        adapter = self._build_adapter(data)
        results = self.allscaip_backbone(adapter)
        # ``node_reps`` is shape [N, hidden_dim] (no padding -- AllScAIP
        # always runs unpadded under HydraGNN).
        inv_node_feat = self.allscaip_output_norm(results["node_reps"])
        # AllScAIP is not equivariant; provide an empty equiv tensor so
        # Base.forward signatures still match.
        equiv_node_feat = inv_node_feat.new_zeros((inv_node_feat.shape[0], 0))
        # No edge-level args are needed by the identity placeholder conv.
        conv_args: Dict[str, Any] = {}
        return inv_node_feat, equiv_node_feat, conv_args

    def __str__(self):
        return "AllScAIPStack"
