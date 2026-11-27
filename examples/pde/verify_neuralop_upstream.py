"""Parity of HydraGNN's FNO and GINO ports with upstream ``neuralop``.

Upstream: https://github.com/neuraloperator/neuraloperator (MIT), verified at
``a6648eeeb60fe0837f2d7d021517a945a159a0c8``. Both run on torch, so one
process imports both: put the upstream checkout (and its pure-Python
dependencies ``tensorly``, ``tensorly-torch``, ``opt_einsum``, ``configmypy``)
on ``--upstream``/``PYTHONPATH``.

For every configuration the script builds the upstream model with matching
settings, copies its weights into the HydraGNN model (complex spectral weights
into the real ``(..., 2)`` storage), runs both on the same inputs and reports
the max relative error ``max|a - b| / max|b|``. Configurations:

* FNO 1D/2D/3D, even and odd resolutions, modes below and above the
  resolution limit (truncation), with and without domain padding, in FP64
  and FP32;
* GINO 2D and 3D on random point clouds with output queries that do not
  match the input points (``linear`` transforms, transformer embedding, no
  open3d / torch_scatter), FP64 and FP32.

Upstream settings matched: ``use_channel_mlp=True``, ``channel_mlp_skip=
"soft-gating"``, ``fno_skip="linear"``, ``preactivation=False``,
``factorization=None`` (dense), ``complex_data=False``, ``positional_embedding=
"grid"`` (FNO), domain padding ``"symmetric"``. Exits non-zero if any error
exceeds ``--tol`` (FP64) or ``--tol32`` (FP32).

Upstream ``SpectralConv`` hard-codes a ``complex64`` output buffer; for the
FP64 runs the script redirects that single dtype to ``complex128`` (see
``_TorchFP64Complex``), otherwise FP64 parity would be capped near 1e-9.
The ``set_default_dtype`` call is needed because upstream's pure-torch
``segment_csr`` allocates its output in the default dtype.
"""

import argparse
import json
import platform
import sys

import torch

FNO_CONFIGS = {
    "fno1d_even": dict(n_modes=(8,), res=(32,), padding=None),
    "fno1d_odd_trunc": dict(n_modes=(40,), res=(21,), padding=None),
    "fno2d": dict(n_modes=(6, 6), res=(16, 20), padding=None),
    "fno2d_odd_trunc": dict(n_modes=(24, 9), res=(15, 11), padding=None),
    "fno2d_pad": dict(n_modes=(6, 6), res=(16, 16), padding=0.25),
    "fno3d": dict(n_modes=(4, 4, 4), res=(8, 9, 10), padding=0.125),
}
GINO_CONFIGS = {
    "gino2d": dict(coord_dim=2, n_modes=(6, 6), latent=(12, 12), points=150),
    "gino3d": dict(coord_dim=3, n_modes=(4, 4, 4), latent=(8, 8, 8), points=200),
}
HIDDEN, IN, OUT, LAYERS = 12, 3, 2, 3


class _TorchFP64Complex:
    """Stand-in for ``torch`` inside upstream ``spectral_convolution``.

    Upstream allocates the spectral output buffer as ``torch.cfloat``
    regardless of the input precision, which caps FP64 parity near 1e-9.
    During FP64 runs ``cfloat`` resolves to ``complex128`` so the reference
    is a genuine double-precision computation; everything else is ``torch``.
    """

    cfloat = torch.complex128

    def __getattr__(self, name):
        return getattr(torch, name)


def _set_upstream_complex(dtype):
    from neuralop.layers import spectral_convolution

    spectral_convolution.torch = (
        _TorchFP64Complex() if dtype == torch.float64 else torch
    )


def _cast_upstream(module, dtype):
    """``Module.to(float64)`` would drop the imaginary part of complex weights."""
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    for p in module.parameters():
        p.data = p.data.to(complex_dtype if p.is_complex() else dtype)
    return module.eval()


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


def _copy(dst, src):
    with torch.no_grad():
        dst.copy_(src.reshape(dst.shape))


def _copy_channel_mlp(ours, theirs):
    for a, b in zip(ours.fcs, theirs.fcs):
        _copy(a.weight, b.weight)
        _copy(a.bias, b.bias)


def _copy_fno_blocks(ours, theirs):
    for i in range(ours.n_layers):
        weight = theirs.convs[i].weight
        weight = getattr(weight, "tensor", weight)
        if callable(getattr(weight, "to_tensor", None)):
            weight = weight.to_tensor()
        _copy(ours.convs[i].weight, torch.view_as_real(weight.detach()))
        _copy(ours.convs[i].bias, theirs.convs[i].bias)
        _copy(ours.fno_skips[i].conv.weight, theirs.fno_skips[i].conv.weight)
        _copy_channel_mlp(ours.channel_mlp[i], theirs.channel_mlp[i])
        _copy(ours.channel_mlp_skips[i].weight, theirs.channel_mlp_skips[i].weight)


def fno_case(cfg, dtype):
    from neuralop.models import FNO as UpstreamFNO

    from hydragnn.models.fno import FNO

    torch.manual_seed(0)
    up = UpstreamFNO(
        n_modes=cfg["n_modes"],
        in_channels=IN,
        out_channels=OUT,
        hidden_channels=HIDDEN,
        n_layers=LAYERS,
        domain_padding=cfg["padding"],
    )
    ours = FNO(
        IN,
        HIDDEN,
        [OUT],
        ["node"],
        LAYERS,
        n_modes=cfg["n_modes"],
        domain_padding=cfg["padding"],
    )
    _copy_channel_mlp(ours.lifting, up.lifting)
    _copy_fno_blocks(ours.fno_blocks, up.fno_blocks)
    _copy_channel_mlp(ours.heads[0], up.projection)
    up, ours = _cast_upstream(up, dtype), ours.to(dtype).eval()
    batch = 2
    x = torch.randn(batch, IN, *cfg["res"], dtype=dtype)
    with torch.no_grad():
        ref = up(x)
        got = ours.heads[0](ours._core(x))
    return _rel(got, ref)


def gino_case(cfg, dtype):
    from neuralop.models import GINO as UpstreamGINO

    from hydragnn.models.gino import GINO

    d = cfg["coord_dim"]
    radius = 0.3 if d == 2 else 0.4
    common_hidden_in, common_hidden_out = [16, 16], [24]
    torch.manual_seed(0)
    up = UpstreamGINO(
        in_channels=IN,
        out_channels=OUT,
        gno_coord_dim=d,
        in_gno_radius=radius,
        out_gno_radius=radius,
        fno_n_modes=cfg["n_modes"],
        fno_hidden_channels=HIDDEN,
        fno_n_layers=LAYERS,
        fno_in_channels=IN,
        gno_embed_channels=6,
        in_gno_channel_mlp_hidden_layers=list(common_hidden_in),
        out_gno_channel_mlp_hidden_layers=list(common_hidden_out),
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    )
    ours = GINO(
        IN,
        HIDDEN,
        [OUT],
        ["node", "query"][:1],
        LAYERS,
        coord_dim=d,
        n_modes=cfg["n_modes"],
        latent_resolution=cfg["latent"],
        in_radius=radius,
        out_radius=radius,
        embed_channels=6,
        in_channel_mlp_hidden_layers=common_hidden_in,
        out_channel_mlp_hidden_layers=common_hidden_out,
    )
    for a, b in (
        (ours.gno_in.channel_mlp, up.gno_in.integral_transform.channel_mlp),
        (ours.gno_out.channel_mlp, up.gno_out.integral_transform.channel_mlp),
        (ours.lifting, up.lifting),
        (ours.heads[0], up.projection),
    ):
        _copy_channel_mlp(a, b)
    _copy_fno_blocks(ours.fno_blocks, up.fno_blocks)
    up, ours = _cast_upstream(up, dtype), ours.to(dtype).eval()

    g = torch.Generator().manual_seed(1)
    geom = torch.rand(cfg["points"], d, generator=g, dtype=dtype)
    queries = torch.rand(cfg["points"] // 2 + 7, d, generator=g, dtype=dtype)
    x = torch.randn(1, cfg["points"], IN, generator=g, dtype=dtype)
    axes = [torch.linspace(0, 1, n, dtype=dtype) for n in cfg["latent"]]
    latent = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).unsqueeze(0)
    with torch.no_grad():
        ref = up(input_geom=geom, latent_queries=latent, output_queries=queries, x=x)
        state = ours.process(ours.encode(_Data(geom, x[0])), None)
        latent_q = ours._query(state, queries, torch.zeros(len(queries), dtype=int))
        got = ours.heads[0](latent_q.t().unsqueeze(0)).permute(0, 2, 1)
    return _rel(got, ref)


class _Data:
    def __init__(self, pos, x):
        self.pos, self.x, self.batch = pos, x, None
        self.num_nodes = pos.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--tol32", type=float, default=1e-5)
    args = parser.parse_args()
    if args.upstream:
        sys.path.insert(0, args.upstream)
    import neuralop

    report = {}
    for dtype, tag in ((torch.float64, "fp64"), (torch.float32, "fp32")):
        # Upstream's naive segment_csr allocates in the default dtype.
        torch.set_default_dtype(dtype)
        _set_upstream_complex(dtype)
        for name, cfg in FNO_CONFIGS.items():
            report[f"{name}_{tag}"] = fno_case(cfg, dtype)
        for name, cfg in GINO_CONFIGS.items():
            report[f"{name}_{tag}"] = gino_case(cfg, dtype)
    failed = sorted(
        k
        for k, v in report.items()
        if not v <= (args.tol if "fp64" in k else args.tol32)
    )
    print(
        json.dumps(
            {
                "report": report,
                "tol_fp64": args.tol,
                "tol_fp32": args.tol32,
                "failed": failed,
                "neuralop_version": getattr(neuralop, "__version__", None),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "device": "cpu",
                "seed": 0,
            },
            indent=2,
        )
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
