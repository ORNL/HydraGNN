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
# Portions ported from neuraloperator/neuraloperator@a6648eeeb60fe0837f2d7d021517a945a159a0c8
# (neuralop/layers/{spectral_convolution,fno_block,channel_mlp,skip_connections,
# padding,embeddings}.py, neuralop/models/fno.py),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# LICENSES/NEURALOP-MIT.txt.
"""FNO: Fourier Neural Operator (Li et al., ICLR 2021) on structured grids.

Port of the dense, real-data, postactivation path of ``neuralop.models.FNO``:

* ``GridEmbeddingND`` appends ``linspace(0, 1, res + 1)[:-1]`` coordinates;
* lifting ``ChannelMLP(in + d -> ratio*hidden -> hidden)``;
* optional symmetric zero ``DomainPadding`` by a fraction of each resolution;
* ``n_layers`` Fourier blocks
  ``x = act(conv(x) + W x); x = act(mlp(x) + g * x)`` (no act after the last);
* projection ``ChannelMLP(hidden -> ratio*hidden -> out)``, one per head.

``SpectralConv`` uses ``rfftn(norm="forward")``, keeps ``n_modes`` modes
centred on the zero frequency in all but the last dim (``n_modes[-1] // 2 + 1``
in the last), enforces Hermitian symmetry before ``irfft`` and adds a bias.
Weights are stored as real ``(..., 2)`` tensors so ``Module.double()`` and
``.to(dtype)`` keep them complex; the FFT dtype follows the input dtype.

HydraGNN interface: node features ``data.x`` live on a tensor-product grid
whose shape is the graph-level ``data.grid_shape [1, d]``, nodes ordered
C-style (``meshgrid(..., indexing="ij")``). Graphs in a batch may have
different grid shapes; equal shapes are stacked into one FFT batch.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydragnn.models.operator_base import OperatorBase
from hydragnn.models.utils.model import activation_function_selection


def resolve_activation(name):
    """``"gelu"`` (the neuralop default) or any HydraGNN activation name."""
    if name == "gelu":
        return F.gelu
    act = activation_function_selection(name)
    if act is None:
        raise ValueError(f"Unknown activation function: {name}")
    return act


class ChannelMLP(nn.Module):
    """Pointwise MLP over dim 1 of ``(B, C, *spatial)`` via ``Conv1d(k=1)``."""

    def __init__(
        self, in_channels, out_channels, hidden_channels, n_layers=2, act=F.gelu
    ):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.fcs = nn.ModuleList(nn.Conv1d(a, b, 1) for a, b in zip(dims, dims[1:]))
        self.act = act

    def forward(self, x):
        shape = x.shape
        if x.ndim != 3:
            x = x.reshape(shape[0], shape[1], -1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.act(x)
        return x.reshape(shape[0], -1, *shape[2:])


class LinearChannelMLP(nn.Module):
    """MLP over the last dim with ``nn.Linear`` layers, ``act`` between them."""

    def __init__(self, layers, act=F.gelu):
        super().__init__()
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(layers, layers[1:]))
        self.act = act

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.act(x)
        return x


class Flattened1dConv(nn.Module):
    """Bias-free 1x1 convolution on ``(B, C, *spatial)`` (the ``linear`` skip)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        shape = x.shape
        x = self.conv(x.reshape(shape[0], shape[1], -1))
        return x.reshape(shape[0], -1, *shape[2:])


class SoftGating(nn.Module):
    def __init__(self, channels, n_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, *(1,) * n_dim))

    def forward(self, x):
        return x * self.weight


class SpectralConv(nn.Module):
    """Dense real-data spectral convolution over the last ``len(n_modes)`` dims."""

    def __init__(self, in_channels, out_channels, n_modes, bias=True):
        super().__init__()
        n_modes = [int(m) for m in n_modes]
        if not n_modes or any(m < 1 for m in n_modes):
            raise ValueError("n_modes must be a non-empty list of positive ints")
        self.order = len(n_modes)
        self.n_modes = n_modes[:-1] + [n_modes[-1] // 2 + 1]
        init_std = (2.0 / (in_channels + out_channels)) ** 0.5
        # Complex normal with total std init_std (as tensor.normal_ on cfloat).
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, *self.n_modes, 2)
            * (init_std / math.sqrt(2.0))
        )
        self.bias = (
            nn.Parameter(init_std * torch.randn(out_channels, *(1,) * self.order))
            if bias
            else None
        )

    def complex_weight(self):
        return torch.view_as_complex(self.weight)

    def forward(self, x):
        batch, _, *sizes = x.shape
        fft_dims = list(range(-self.order, 0))
        fft_size = list(sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1
        xf = torch.fft.rfftn(x, dim=fft_dims, norm="forward")
        if self.order > 1:
            xf = torch.fft.fftshift(xf, dim=fft_dims[:-1])
        weight = self.complex_weight()
        out_fft = xf.new_zeros(batch, weight.shape[1], *fft_size)

        starts = [m - min(s, m) for s, m in zip(fft_size, self.n_modes)]
        slices_w = [slice(None), slice(None)]
        slices_w += [slice(s // 2, -s // 2) if s else slice(None) for s in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = weight[tuple(slices_w)]

        slices_x = [slice(None), slice(None)]
        for size, kept in zip(fft_size[:-1], weight.shape[2:-1]):
            center = size // 2
            slices_x.append(slice(center - kept // 2, center + kept // 2 + kept % 2))
        kept = weight.shape[-1]
        slices_x.append(slice(None, kept) if kept < fft_size[-1] else slice(None))
        slices_x = tuple(slices_x)
        out_fft[slices_x] = torch.einsum("bi...,io...->bo...", xf[slices_x], weight)

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
            out_fft = torch.fft.ifftn(
                out_fft, s=sizes[:-1], dim=fft_dims[:-1], norm="forward"
            )
        # irfft requires real DC (and Nyquist, for even sizes) coefficients.
        keep = torch.ones(fft_size[-1], dtype=x.dtype, device=x.device)
        keep[0] = 0.0
        if sizes[-1] % 2 == 0:
            keep[-1] = 0.0
        out_fft = torch.complex(out_fft.real, out_fft.imag * keep)
        out = torch.fft.irfft(out_fft, n=sizes[-1], dim=-1, norm="forward")
        if self.bias is not None:
            out = out + self.bias
        return out


class DomainPadding(nn.Module):
    """Symmetric zero padding by ``round(fraction * resolution)`` per dim."""

    def __init__(self, fractions):
        super().__init__()
        self.fractions = [float(f) for f in fractions]

    def _pads(self, sizes):
        return [round(f * s) for f, s in zip(self.fractions, sizes)]

    def pad(self, x):
        pads = self._pads(x.shape[2:])
        flat = [p for pad in reversed(pads) for p in (pad, pad)]
        return F.pad(x, flat, mode="constant"), pads

    @staticmethod
    def unpad(x, pads):
        return x[(Ellipsis,) + tuple(slice(p, -p) if p else slice(None) for p in pads)]


def grid_embedding(x):
    """Append ``linspace(0, 1, res + 1)[:-1]`` grids (``ij``) to channels."""
    axes = [
        torch.linspace(0.0, 1.0, s + 1, dtype=x.dtype, device=x.device)[:-1]
        for s in x.shape[2:]
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    grids = [g.expand(x.shape[0], 1, *g.shape) for g in grids]
    return torch.cat([x, *grids], dim=1)


class FNOBlocks(nn.Module):
    """``n_layers`` postactivation Fourier layers (linear skip, soft-gated MLP)."""

    def __init__(
        self, channels, n_modes, n_layers, channel_mlp_expansion=0.5, act=F.gelu
    ):
        super().__init__()
        n_dim = len(n_modes)
        self.n_layers = n_layers
        self.act = act
        self.convs = nn.ModuleList(
            SpectralConv(channels, channels, n_modes) for _ in range(n_layers)
        )
        self.fno_skips = nn.ModuleList(
            Flattened1dConv(channels, channels) for _ in range(n_layers)
        )
        hidden = round(channels * channel_mlp_expansion)
        self.channel_mlp = nn.ModuleList(
            ChannelMLP(channels, channels, hidden, 2, act) for _ in range(n_layers)
        )
        self.channel_mlp_skips = nn.ModuleList(
            SoftGating(channels, n_dim) for _ in range(n_layers)
        )

    def forward(self, x, index):
        skip_fno = self.fno_skips[index](x)
        skip_mlp = self.channel_mlp_skips[index](x)
        x = self.convs[index](x) + skip_fno
        if index < self.n_layers - 1:
            x = self.act(x)
        x = self.channel_mlp[index](x) + skip_mlp
        if index < self.n_layers - 1:
            x = self.act(x)
        return x


def graph_ptr(data):
    """Node offsets per graph (``[0, N]`` for an unbatched graph)."""
    ptr = getattr(data, "ptr", None)
    if ptr is not None:
        return ptr
    return torch.tensor([0, data.num_nodes], device=data.x.device)


class FNO(OperatorBase):
    supported_output_levels = frozenset({"node"})

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        output_type,
        num_layers,
        n_modes=(16, 16),
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        positional_embedding="grid",
        domain_padding=None,
        channel_mlp_expansion=0.5,
        activation_function="gelu",
        loss_function_type="mse",
        loss_weights=None,
        freeze_conv=False,
    ):
        super().__init__(output_dim, output_type, loss_function_type, loss_weights)
        self.n_modes = [int(m) for m in n_modes]
        self.n_dim = len(self.n_modes)
        if not 1 <= self.n_dim <= 3:
            raise ValueError("fno_n_modes must have 1, 2 or 3 entries")
        if num_layers < 1:
            raise ValueError("FNO needs num_conv_layers >= 1")
        if positional_embedding not in ("grid", None, "none"):
            raise ValueError("fno_positional_embedding must be 'grid' or 'none'")
        self.grid_embedding = positional_embedding == "grid"
        if domain_padding is None or domain_padding == 0:
            self.domain_padding = None
        else:
            fractions = (
                [domain_padding] * self.n_dim
                if isinstance(domain_padding, (int, float))
                else list(domain_padding)
            )
            if len(fractions) != self.n_dim:
                raise ValueError("fno_domain_padding must have one entry per dim")
            self.domain_padding = DomainPadding(fractions)
        act = resolve_activation(activation_function)
        lifting_in = input_dim + self.n_dim * self.grid_embedding
        self.lifting = ChannelMLP(
            lifting_in, hidden_dim, int(lifting_channel_ratio * hidden_dim), 2, act
        )
        self.fno_blocks = FNOBlocks(
            hidden_dim, self.n_modes, num_layers, channel_mlp_expansion, act
        )
        projection = int(projection_channel_ratio * hidden_dim)
        out_mult = 1 + self.var_output
        self.heads = nn.ModuleList(
            ChannelMLP(hidden_dim, dim * out_mult, projection, 2, act)
            for dim in self.head_dims
        )
        if freeze_conv:
            for name, module in self.named_children():
                if name != "heads":
                    module.requires_grad_(False)

    def _grid_groups(self, data):
        """Lists of (graph ids, grid shape) whose graphs share a grid shape."""
        shapes = getattr(data, "grid_shape", None)
        if shapes is None:
            raise ValueError("FNO needs the graph-level data.grid_shape [1, d]")
        shapes = shapes.reshape(-1, self.n_dim).tolist()
        ptr = graph_ptr(data).tolist()
        groups = {}
        for g, shape in enumerate(shapes):
            if math.prod(shape) != ptr[g + 1] - ptr[g]:
                raise ValueError(
                    f"grid_shape {shape} does not match {ptr[g + 1] - ptr[g]} nodes"
                )
            groups.setdefault(tuple(shape), []).append(g)
        return ptr, groups

    def _core(self, x):
        if self.grid_embedding:
            x = grid_embedding(x)
        x = self.lifting(x)
        pads = None
        if self.domain_padding is not None:
            x, pads = self.domain_padding.pad(x)
        for i in range(self.fno_blocks.n_layers):
            if self.conv_checkpointing and self.training:
                x = checkpoint(self.fno_blocks, x, i, use_reentrant=False)
            else:
                x = self.fno_blocks(x, i)
        if pads is not None:
            x = DomainPadding.unpad(x, pads)
        return x

    def encode(self, data):
        return {"x": data.x}

    def process(self, state, data):
        x = state["x"]
        ptr, groups = self._grid_groups(data)
        latent = x.new_empty(x.shape[0], self.fno_blocks.convs[0].weight.shape[1])
        for shape, graphs in groups.items():
            index = torch.cat(
                [torch.arange(ptr[g], ptr[g + 1], device=x.device) for g in graphs]
            )
            grid = x[index].reshape(len(graphs), *shape, -1).movedim(-1, 1)
            out = self._core(grid).movedim(1, -1).reshape(index.numel(), -1)
            latent = latent.index_copy(0, index, out)
        return {"node": latent}

    def decode(self, state, data):
        # Pointwise heads: (N, C) -> (1, C, N) -> (N, out).
        latent = state["node"].t().unsqueeze(0)
        return [head(latent)[0].t() for head in self.heads]
