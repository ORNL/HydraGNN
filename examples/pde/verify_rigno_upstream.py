"""Parity of HydraGNN's RIGNO port with the upstream JAX implementation.

Upstream: https://github.com/camlab-ethz/rigno (MIT). The two stages run in
different environments, because upstream needs JAX/Flax/jraph and HydraGNN
needs torch:

1. ``--stage export`` (HydraGNN env) builds region interaction graphs on an
   L-shaped mesh, initializes ``hydragnn.models.rigno.RIGNO`` with random
   weights in FP64, and saves graphs, weights, inputs and intermediate
   outputs (encoded/processed latents and final predictions) to ``--npz``.
2. ``--stage compare`` (JAX env with ``--upstream`` on the path) checks the
   graph construction against upstream's builder methods on the same
   regional nodes. It also loads the torch weights into upstream ``RIGNO``,
   runs it on the same graphs in FP64 and compares every intermediate. Prints
   a JSON report and exits non-zero if any error exceeds ``--tol``.

Upstream settings: ``mlp_hidden_layers = mlp_layers - 1`` and evaluation mode
(no edge masking; the masks come from different RNGs) and FP64 aggregation
(upstream's encoder casts its mean to float32), and LayerNorm epsilon 1e-5
(torch default; flax uses 1e-6). The output MLP is
compared end to end by making the head width equal to the latent width,
because upstream's output MLP has width ``num_outputs`` throughout.
Configurations: plain (no time inputs) and time-conditioned
(``concat_time`` + ``time_conditioning`` versus upstream
``concatenate_t/tau`` + ``conditioned_normalization``).
"""

import argparse
import json
import platform
import sys

import numpy as np

CONFIGS = {
    "plain": dict(mlp_layers=2, num_steps=3, time=False),
    "time": dict(mlp_layers=2, num_steps=3, time=True),
    "deep_mlp": dict(mlp_layers=3, num_steps=2, time=True),
}
HIDDEN = 16
T, DT = 0.25, 0.5  # exactly representable: upstream casts time to float32


# -- stage 1: HydraGNN --------------------------------------------------------
def _l_shape(n):
    import torch

    from hydragnn.data.meshes import from_triangles

    h = 1.0 / n
    ids, pos = {}, []
    for j in range(n + 1):
        for i in range(n + 1):
            if i * h > 0.5 + 1e-12 and j * h > 0.5 + 1e-12:
                continue
            ids[i, j] = len(pos)
            pos.append((i * h, j * h))
    tris = []
    for j in range(n):
        for i in range(n):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(c in ids for c in corners):
                a, b, c, d = (ids[c] for c in corners)
                tris += [(a, b, c), (a, c, d)]
    return from_triangles(torch.tensor(pos, dtype=torch.float64), torch.tensor(tris))


def export(path, resolutions, seed):
    import torch

    from hydragnn.data.meshes import build_rigno_graph
    from hydragnn.models.rigno import RIGNO

    out = {}
    for n in resolutions:
        g = torch.Generator().manual_seed(seed + n)
        mesh = _l_shape(n)
        mesh.x = torch.randn(mesh.num_nodes, HIDDEN, generator=g, dtype=torch.float64)
        mesh.time = torch.tensor([T], dtype=torch.float64)
        mesh.dt = torch.tensor([DT], dtype=torch.float64)
        state = g.get_state()
        data = build_rigno_graph(mesh, generator=g)
        g.set_state(state)
        raw = build_rigno_graph(mesh, generator=g, ensure_coverage=False)
        assert torch.equal(raw.rigno_rpos, data.rigno_rpos)
        key = f"n{n}"
        for field in ("rigno_x", "rigno_rpos", "rigno_rradius"):
            out[f"{key}/{field}"] = data[field].numpy()
        for field in ("rigno_p2r_index", "rigno_r2r_index", "rigno_r2p_index"):
            out[f"{key}/{field}"] = data[field].numpy()
        out[f"{key}/raw_r2p_index"] = raw.rigno_r2p_index.numpy()
        out[f"{key}/u"] = data.x.numpy()

        for name, cfg in CONFIGS.items():
            torch.manual_seed(seed)
            model = RIGNO(
                HIDDEN,
                HIDDEN,
                [HIDDEN],
                ["node"],
                cfg["num_steps"],
                mlp_layers=cfg["mlp_layers"],
                concat_time=cfg["time"],
                time_conditioning=cfg["time"],
            ).double()
            model.eval()
            with torch.no_grad():
                enc = model.encode(data)
                proc = model.process(enc, data)
                pred = model.decode(proc, data)[0]
            prefix = f"{key}/{name}"
            out[f"{prefix}/pnodes_encoded"] = enc["pnode"].numpy()
            out[f"{prefix}/rnodes_encoded"] = enc["rnode"].numpy()
            out[f"{prefix}/rnodes_processed"] = proc["rnode"].numpy()
            out[f"{prefix}/output"] = pred.numpy()
            for pname, value in model.state_dict().items():
                out[f"{prefix}/param/{pname}"] = value.numpy()
    out["meta"] = np.array(
        json.dumps(
            {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "resolutions": resolutions,
                "seed": seed,
                "configs": CONFIGS,
            }
        )
    )
    np.savez(path, **out)


# -- stage 2: upstream --------------------------------------------------------
def _block(params, prefix, mlp_layers, conditioned):
    """Flax FeedForwardBlock params from a torch ``_Block`` (or ``_mlp``)."""
    mlp = f"{prefix}.mlp" if conditioned is not None else prefix
    block = {}
    for i in range(mlp_layers):
        block[f"layers_{i}"] = {
            "kernel": params[f"{mlp}.{2 * i}.weight"].T,
            "bias": params[f"{mlp}.{2 * i}.bias"],
        }
    ln = f"{mlp}.{2 * mlp_layers - 1}"
    if f"{ln}.weight" in params:
        block["layernorm"] = {
            "scale": params[f"{ln}.weight"],
            "bias": params[f"{ln}.bias"],
        }
    if conditioned:
        block["correction"] = {
            f"mlp_{part}": {
                f"layers_{i}": {
                    "kernel": params[f"{prefix}.cond.{part}.{i}.weight"].T,
                    "bias": params[f"{prefix}.cond.{part}.{i}.bias"],
                }
                for i in (0, 2)
            }
            for part in ("scale", "bias")
        }
    return block


def _upstream_params(params, cfg, template):
    m, c = cfg["mlp_layers"], cfg["time"]
    p = {
        "encoder": {
            "gnn": {
                "encoder_edges_p2r": _block(params, "p2r_embed", m, None),
                "encoder_nodes_pnodes": _block(params, "pnode_embed", m, None),
                "encoder_nodes_rnodes": _block(params, "rnode_embed", m, None),
                "processor_0_edges_p2r": _block(params, "encoder.edge", m, c),
                "processor_0_nodes_rnodes": _block(params, "encoder.recv", m, c),
                "processor_0_nodes_pnodes": _block(params, "encoder.send", m, c),
            }
        },
        "processor": {
            "gnn": {"encoder_edges_r2r": _block(params, "r2r_embed", m, None)}
        },
        "decoder": {
            "gnn": {
                "encoder_edges_r2p": _block(params, "r2p_embed", m, None),
                "processor_0_edges_r2p": _block(params, "decoder.edge", m, c),
                "processor_0_nodes_pnodes": _block(params, "decoder.recv", m, c),
                # Updated by upstream but never read: keep its random init.
                "processor_0_nodes_rnodes": template["decoder"]["gnn"][
                    "processor_0_nodes_rnodes"
                ],
                "decoder_nodes_pnodes": _block(params, "heads.0", m, None),
            }
        },
    }
    for k in range(cfg["num_steps"]):
        gnn = p["processor"]["gnn"]
        gnn[f"processor_{k}_edges_r2r"] = _block(params, f"processor.{k}.edge", m, c)
        gnn[f"processor_{k}_nodes_rnodes"] = _block(params, f"processor.{k}.recv", m, c)
    return p


def _edge_set(a):
    return {tuple(e) for e in np.asarray(a).tolist()}


def compare(path, upstream, tol):
    sys.path.insert(0, upstream)
    import jax

    jax.config.update("jax_enable_x64", True)
    import flax
    import jax.numpy as jnp
    import jraph

    import rigno.models.rigno as upstream_rigno
    from rigno.models.operator import Inputs
    from rigno.models.rigno import (
        RIGNO,
        RegionInteractionGraphBuilder,
        RegionInteractionGraphMetadata,
    )

    # Upstream's encoder aggregates in float32; force the input dtype instead
    # so the comparison is FP64 throughout.
    graph_net = upstream_rigno.DeepTypedGraphNet

    def fp64_graph_net(**kwargs):
        kwargs["f32_aggregation"] = False
        return graph_net(**kwargs)

    upstream_rigno.DeepTypedGraphNet = fp64_graph_net

    # torch.nn.LayerNorm uses eps=1e-5, flax.linen.LayerNorm 1e-6.
    import functools
    import types

    import rigno.models.utils as upstream_utils

    layer_norm = functools.partial(flax.linen.LayerNorm, epsilon=1e-5)
    upstream_utils.nn = types.SimpleNamespace(
        **{**vars(flax.linen), "LayerNorm": layer_norm}
    )

    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    report = {
        "export": meta,
        "jax": jax.__version__,
        "flax": flax.__version__,
        "jraph": jraph.__version__ if hasattr(jraph, "__version__") else "unknown",
        "tol": tol,
        "checks": {},
    }
    failed = []

    def record(name, value):
        report["checks"][name] = value
        if (isinstance(value, bool) and not value) or (
            isinstance(value, float) and not value <= tol
        ):
            failed.append(name)

    builder = RegionInteractionGraphBuilder(
        periodic=False,
        rmesh_levels=4,
        subsample_factor=4.0,
        overlap_factor_p2r=1.0,
        overlap_factor_r2p=2.0,
        node_coordinate_freqs=0,
    )
    for n in meta["resolutions"]:
        key = f"n{n}"
        x, rpos = data[f"{key}/rigno_x"], data[f"{key}/rigno_rpos"]
        radius = data[f"{key}/rigno_rradius"]
        N, R = x.shape[0], rpos.shape[0]

        # Graph construction on the same regional nodes.
        up_r = builder._compute_minimum_support_radius(rpos)
        record(f"{key}/graph/radius_max_abs", float(np.abs(up_r - radius).max()))
        clip = lambda f: np.clip(f * up_r, 0, up_r.max())  # noqa: E731
        up_p2r = np.asarray(builder._get_supported_pnodes_by_rnodes(rpos, x, clip(1.0)))
        record(
            f"{key}/graph/p2r_equal",
            _edge_set(up_p2r) == _edge_set(data[f"{key}/rigno_p2r_index"].T),
        )
        up_r2p = np.flip(
            np.asarray(builder._get_supported_pnodes_by_rnodes(rpos, x, clip(2.0))), -1
        )
        record(
            f"{key}/graph/r2p_equal",
            _edge_set(up_r2p) == _edge_set(data[f"{key}/raw_r2p_index"].T),
        )
        up_r2r, _ = builder._get_r2r_edges(rpos)
        record(
            f"{key}/graph/r2r_equal",
            _edge_set(up_r2r) == _edge_set(data[f"{key}/rigno_r2r_index"].T),
        )
        extra = data[f"{key}/rigno_r2p_index"].shape[1] - up_r2p.shape[0]
        report["checks"][f"{key}/graph/coverage_edges_added"] = int(extra)

        # Model parity on HydraGNN's graphs, with upstream's dummy node/edge.
        def dummy(idx, s, r):
            return np.concatenate([idx.T, [[s, r]]]).astype(np.int64)[None]

        graph_meta = RegionInteractionGraphMetadata(
            x_pnodes_inp=np.concatenate([x, np.zeros((1, 2))])[None],
            x_pnodes_out=np.concatenate([x, np.zeros((1, 2))])[None],
            x_rnodes=np.concatenate([rpos, np.zeros((1, 2))])[None],
            r_rnodes=np.concatenate([radius, [0.0]])[None],
            p2r_edge_indices=dummy(data[f"{key}/rigno_p2r_index"], N, R),
            r2r_edge_indices=dummy(data[f"{key}/rigno_r2r_index"], R, R),
            r2r_edge_domains=np.zeros(
                (1, data[f"{key}/rigno_r2r_index"].shape[1] + 1, 2), np.uint8
            ),
            r2p_edge_indices=dummy(data[f"{key}/rigno_r2p_index"], R, N),
        )
        graphs = builder.build_graphs(graph_meta)
        u = data[f"{key}/u"]
        for name, cfg in meta["configs"].items():
            prefix = f"{key}/{name}"
            model = RIGNO(
                num_outputs=HIDDEN,
                processor_steps=cfg["num_steps"],
                node_latent_size=HIDDEN,
                edge_latent_size=HIDDEN,
                mlp_hidden_layers=cfg["mlp_layers"] - 1,
                concatenate_t=cfg["time"],
                concatenate_tau=cfg["time"],
                conditioned_normalization=cfg["time"],
                cond_norm_hidden_size=16,
                p_edge_masking=0.0,
            )
            inputs = Inputs(
                u=jnp.asarray(u)[None, None],
                c=None,
                x_inp=jnp.asarray(x)[None, None],
                x_out=jnp.asarray(x)[None, None],
                t=T if cfg["time"] else None,
                tau=DT if cfg["time"] else None,
            )
            template = model.init(jax.random.PRNGKey(0), inputs, graphs=graphs)[
                "params"
            ]
            params = {
                k[len(prefix) + 7 :]: data[k]
                for k in data.files
                if k.startswith(f"{prefix}/param/")
            }
            new = _upstream_params(params, cfg, template)
            shapes = jax.tree_util.tree_map(np.shape, template)
            record(
                f"{prefix}/param_tree_equal",
                shapes == jax.tree_util.tree_map(np.shape, new),
            )
            pred, state = model.apply(
                {"params": new}, inputs, graphs=graphs, mutable=["intermediates"]
            )
            inter = state["intermediates"]
            for field in ("pnodes_encoded", "rnodes_encoded", "rnodes_processed"):
                ours = data[f"{prefix}/{field}"]
                theirs = np.asarray(inter[field][0])[0, 0]
                record(f"{prefix}/{field}_max_abs", float(np.abs(ours - theirs).max()))
            ours = data[f"{prefix}/output"]
            theirs = np.asarray(pred)[0, 0]
            err = float(np.abs(ours - theirs).max())
            record(f"{prefix}/output_max_abs", err)
            report["checks"][f"{prefix}/output_scale"] = float(np.abs(ours).max())
    report["passed"] = not failed
    report["failed"] = failed
    print(json.dumps(report, indent=2))
    return 0 if not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=("export", "compare"), required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--upstream", help="path of the camlab-ethz/rigno clone")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[12, 20])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.stage == "export":
        export(args.npz, args.resolutions, args.seed)
        return 0
    if not args.upstream:
        parser.error("--upstream is required for --stage compare")
    return compare(args.npz, args.upstream, args.tol)


if __name__ == "__main__":
    sys.exit(main())
