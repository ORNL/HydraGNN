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
"""FluxGNN-style conservative finite-volume operator.

Inspired by Horie & Mitsume, "Graph Neural PDE Solvers with Conservation and
Similarity-Equivariance", ICML 2024 (reference code
yellowshippo/fluxgnn-icml2024 @ d4d44497). This is not a faithful port; see
``docs/development/pde_wp03_design.md`` for the differences.

Unknowns live on cells (``MeshData``). ``data.cell_x`` holds the cell inputs,
split by ``layout`` into the conserved state ``u [C, k]``, vector fields
(``[C, d]`` each; the first one advects) and scalar features.

* Encoder: ``z = E u`` with ``E`` linear and bias-free (``hidden_dim >= k``);
  the decoder is its pseudo-inverse, so a conservative latent update is a
  conservative physical update.
* Each of ``num_conv_layers`` explicit substeps of length
  ``dt / num_conv_layers`` applies
  ``z <- z - dt_s / V (sum_f s_f A_f F_f + sum_b A_b F_b) + dt_s q``.
* Interior flux density ``F_f`` (owner -> neighbour): upwind advection by the
  face-averaged normal velocity, two-point diffusion with a learned
  non-negative diffusivity per channel, and an MLP correction
  ``g(o, n, +n_f) - g(n, o, -n_f)``. Every part changes sign with the face
  orientation, so interior fluxes cancel exactly.
* Boundary flux ``F_b`` (``boundary="learned"``): upwind outflow plus an MLP
  of the owner state, the boundary normal velocity and an embedding of
  ``bface_tag``; ``"closed"`` sets it to zero.
* Optional source ``q`` (``source=True``): MLP of the cell state and scalars.

Features enter only as rotation/reflection/translation invariants (vector
components along the face normal, squared tangential magnitude, centroid
distance). The correction MLPs end in zero-initialised layers, so an
untrained model is a plain upwind/diffusion finite-volume scheme.
"""

import torch
from torch import nn

from hydragnn.models.meshgraphnet import _mlp
from hydragnn.models.operator_base import OperatorBase
from hydragnn.physics.conservation import fv_increment

BOUNDARY_MODES = ("learned", "closed")


def _zero_last(mlp):
    last = mlp[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return mlp


class FluxGNN(OperatorBase):
    supported_output_levels = frozenset({"cell", "face"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_steps,
        activation_function="relu",
        loss_function_type="mse",
        loss_weights=None,
        layout=None,
        dt=1.0,
        spatial_dim=2,
        mlp_layers=2,
        advection=True,
        diffusion=True,
        boundary="learned",
        num_bface_tags=1,
        source=False,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        if self.var_output:
            raise ValueError("FluxGNN does not support GaussianNLLLoss outputs")
        if num_steps < 1 or mlp_layers < 1:
            raise ValueError("FluxGNN needs num_conv_layers >= 1 and mlp_layers >= 1")
        if boundary not in BOUNDARY_MODES:
            raise ValueError(f"fluxgnn_boundary must be one of {BOUNDARY_MODES}")
        if layout is None:
            layout = {"state": list(range(input_dim)), "vectors": [], "scalars": []}
        self.state_cols = list(layout["state"])
        self.vector_cols = [list(cols) for cols in layout.get("vectors", [])]
        self.scalar_cols = list(layout.get("scalars", []))
        k, nv, ns = len(self.state_cols), len(self.vector_cols), len(self.scalar_cols)
        if k == 0:
            raise ValueError("FluxGNN needs at least one conserved state channel")
        if any(len(cols) != spatial_dim for cols in self.vector_cols):
            raise ValueError(f"FluxGNN vector inputs must have dim {spatial_dim}")
        used = self.state_cols + sum(self.vector_cols, []) + self.scalar_cols
        if sorted(used) != list(range(input_dim)):
            raise ValueError("FluxGNN layout must partition the cell input columns")
        if hidden_dim < k:
            raise ValueError("FluxGNN needs hidden_dim >= number of state channels")
        if any(dim != k for dim in self.head_dims):
            raise ValueError(
                f"FluxGNN cell and face heads must have the state dimension {k}"
            )
        if self.head_type.count("cell") > 1 or self.head_type.count("face") > 1:
            raise ValueError("FluxGNN supports at most one cell and one face head")

        self.num_steps = num_steps
        self.dt = float(dt)
        self.spatial_dim = spatial_dim
        self.advection = advection and nv > 0
        self.diffusion = diffusion
        self.boundary = boundary
        act = activation_function

        self.encoder = nn.Linear(k, hidden_dim, bias=False)
        nn.init.orthogonal_(self.encoder.weight)
        flux_in = 2 * hidden_dim + 2 * ns + 4 * nv + 1
        self.flux_mlp = _zero_last(
            _mlp(flux_in, hidden_dim, hidden_dim, mlp_layers, act, False)
        )
        if diffusion:
            # softplus(-4) ~ 0.018; learned, non-negative per latent channel.
            self.diffusivity = nn.Parameter(torch.full((hidden_dim,), -4.0))
        if boundary == "learned":
            self.tag_embedding = nn.Embedding(num_bface_tags, 8)
            self.bface_mlp = _zero_last(
                _mlp(
                    hidden_dim + ns + 2 * nv + 8,
                    hidden_dim,
                    hidden_dim,
                    mlp_layers,
                    act,
                    False,
                )
            )
        self.source_mlp = (
            _zero_last(
                _mlp(
                    hidden_dim + ns + nv, hidden_dim, hidden_dim, mlp_layers, act, False
                )
            )
            if source
            else None
        )
        if freeze_conv:
            for name, param in self.named_parameters():
                if not name.startswith("encoder."):
                    param.requires_grad_(False)

    # -- features ---------------------------------------------------------
    def _split(self, data):
        x = data.cell_x
        u = x[:, self.state_cols]
        scalars = x[:, self.scalar_cols]
        if self.vector_cols:
            vectors = torch.stack([x[:, cols] for cols in self.vector_cols], dim=1)
        else:
            vectors = x.new_zeros(x.shape[0], 0, self.spatial_dim)
        return u, vectors, scalars

    @staticmethod
    def _project(vectors, normal):
        """Normal component and squared tangential magnitude, ``[*, nv]`` each."""
        vn = (vectors * normal.unsqueeze(1)).sum(-1)
        vt = ((vectors * vectors).sum(-1) - vn * vn).clamp_min(0.0)
        return vn, vt

    def _decoder_matrix(self):
        return torch.linalg.pinv(self.encoder.weight)  # [k, hidden]

    # -- encode / process / decode ---------------------------------------
    def encode(self, data):
        u, vectors, scalars = self._split(data)
        dtype = u.dtype
        o, n = data.face_index
        normal = data.face_normal[:, : self.spatial_dim].to(dtype)
        centroid = data.cell_centroid[:, : self.spatial_dim].to(dtype)
        vn_o, vt_o = self._project(vectors[o], normal)
        vn_n, vt_n = self._project(vectors[n], normal)
        dist = (centroid[n] - centroid[o]).norm(dim=-1, keepdim=True)
        geometry = {
            "s_o": scalars[o],
            "s_n": scalars[n],
            "vn_o": vn_o,
            "vn_n": vn_n,
            "vt_o": vt_o,
            "vt_n": vt_n,
            "dist": dist,
        }
        if self.boundary == "learned":
            b = data.bface_owner
            tag = data.bface_tag.long()
            if tag.numel() and (
                int(tag.min()) < 0
                or int(tag.max()) >= self.tag_embedding.num_embeddings
            ):
                raise ValueError(
                    "bface_tag values must lie in [0, fluxgnn_num_bface_tags)"
                )
            bnormal = data.bface_normal[:, : self.spatial_dim].to(dtype)
            vn_b, vt_b = self._project(vectors[b], bnormal)
            geometry.update(
                s_b=scalars[b],
                vn_b=vn_b,
                vt_b=vt_b,
                tag=self.tag_embedding(tag).to(dtype),
            )
        if self.source_mlp is not None:
            geometry["cell"] = torch.cat([scalars, (vectors * vectors).sum(-1)], dim=-1)
        return {"cell": self.encoder(u), "u0": u, "geometry": geometry}

    def _terms(self, z, g, data):
        o, n = data.face_index
        z_o, z_n = z[o], z[n]
        forward = torch.cat(
            [
                z_o,
                z_n,
                g["s_o"],
                g["s_n"],
                g["vn_o"],
                g["vn_n"],
                g["vt_o"],
                g["vt_n"],
                g["dist"],
            ],
            dim=-1,
        )
        backward = torch.cat(
            [
                z_n,
                z_o,
                g["s_n"],
                g["s_o"],
                -g["vn_n"],
                -g["vn_o"],
                g["vt_n"],
                g["vt_o"],
                g["dist"],
            ],
            dim=-1,
        )
        flux = self.flux_mlp(forward) - self.flux_mlp(backward)
        if self.advection:
            un = 0.5 * (g["vn_o"][:, :1] + g["vn_n"][:, :1])
            flux = flux + un * torch.where(un > 0, z_o, z_n)
        if self.diffusion:
            kappa = nn.functional.softplus(self.diffusivity)
            flux = flux - kappa * (z_n - z_o) / g["dist"]

        bflux = None
        if self.boundary == "learned":
            z_b = z[data.bface_owner]
            bflux = self.bface_mlp(
                torch.cat([z_b, g["s_b"], g["vn_b"], g["vt_b"], g["tag"]], dim=-1)
            )
            if self.advection:
                bflux = bflux + g["vn_b"][:, :1].clamp_min(0.0) * z_b
        source = None
        if self.source_mlp is not None:
            source = self.source_mlp(torch.cat([z, g["cell"]], dim=-1))
        return flux, bflux, source

    def process(self, state, data):
        z = state["cell"]
        g = state["geometry"]
        dt_sub = self.dt / self.num_steps
        totals = {"face_flux": 0.0, "bface_flux": None, "source": None}
        for _ in range(self.num_steps):
            flux, bflux, source = self._terms(z, g, data)
            z = z + fv_increment(data, flux, bflux, source, dt_sub)
            totals["face_flux"] = totals["face_flux"] + flux
            if bflux is not None:
                prev = totals["bface_flux"]
                totals["bface_flux"] = bflux if prev is None else prev + bflux
            if source is not None:
                prev = totals["source"]
                totals["source"] = source if prev is None else prev + source
        averages = {
            key: None if value is None else value / self.num_steps
            for key, value in totals.items()
        }
        return {"cell": z, "z0": state["cell"], "u0": state["u0"], **averages}

    def decode(self, state, data):
        decoder = self._decoder_matrix().to(state["cell"].dtype)
        # u0 + D (z - z0) avoids the round-off of D E u0 ~ u0.
        u_new = state["u0"] + (state["cell"] - state["z0"]) @ decoder.T
        face = state["face_flux"] @ decoder.T
        return [u_new if level == "cell" else face for level in self.head_type]

    def fluxes(self, data):
        """Physical-space update and its time-averaged terms over one forward.

        Returns ``u_new [C, k]``, ``face_flux [F, k]``, ``bface_flux [Fb, k]``
        (or ``None``) and ``source [C, k]`` (or ``None``), so that
        ``u_new = u0 - dt / V (sum s A face_flux + sum A bface_flux) + dt source``.
        """
        state = self.process(self.encode(data), data)
        decoder = self._decoder_matrix().to(state["cell"].dtype)
        project = lambda value: None if value is None else value @ decoder.T
        return {
            "u_new": state["u0"] + (state["cell"] - state["z0"]) @ decoder.T,
            "face_flux": project(state["face_flux"]),
            "bface_flux": project(state["bface_flux"]),
            "source": project(state["source"]),
        }
