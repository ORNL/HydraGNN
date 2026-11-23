"""Faithful frozen primary-checkpoint K=30 dropout capture; no fitting.

Captures dropout predictions from the primary saved checkpoints.
Run in the recorded scientific environment. Writes only --out.
"""
from fip_release_paths import release_root
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import argparse, importlib.util, json, pickle, time
import numpy as np
import torch
from torch_geometric.loader import DataLoader

WORK = Path(str(release_root() / ""))
FROZEN = WORK / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CACHE = WORK / "revision_R2/studies_2026_11_17/mechanical_checkpoint_selection/outer"
REPLAY = WORK / "revision_R2/studies_2026_11_15/inference/replay_saved.py"
sp = importlib.util.spec_from_file_location("verified_uq_replay", REPLAY)
replay = importlib.util.module_from_spec(sp)
sp.loader.exec_module(replay)
runner, joint = (replay.runner, replay.joint)
DROP_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
)


def components(ds, level, c):
    if level == "node":
        state = (
            ds._pergrain_target_state
            if c == 0
            else ds._per_grain_aux_target_states[c - 1]
        )
        scaler = ds._scaler_pergrain_y if c == 0 else ds._scaler_per_grain_aux_y[c - 1]
    else:
        state = (
            ds._pergraph_target_state
            if c == 0
            else ds._aux_pergraph_target_states[c - 1]
        )
        scaler = ds._scaler_pergraph_y if c == 0 else ds._scaler_aux_pergraph_y[c - 1]
    return (state, scaler)


def forward(model, values, device, batch_size):
    nodes, graphs, nt, gt = ([], [], [], [])
    with torch.no_grad():
        for batch in DataLoader(
            values, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            batch = batch.to(device)
            pn, pg = model(batch)
            assert pn.shape[1] == 20 and pg.shape[1] == 18
            assert torch.isfinite(pn).all() and torch.isfinite(pg).all()
            nodes.append(pn.cpu().numpy())
            graphs.append(pg.cpu().numpy())
            nt.append(batch.pn_target.cpu().numpy())
            gt.append(batch.pg_target.cpu().numpy())
    return tuple((np.concatenate(x) for x in [nodes, graphs, nt, gt]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--only", default="", help="repeat,fold; empty runs all ten primary folds"
    )
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=2026111700)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        assert (
            torch.cuda.is_available()
        ), "GPU unavailable; do not silently switch devices"
    device = torch.device(args.device)
    torch.set_num_threads(2)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(35400 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    assert joint.sha(FROZEN / "partitions.json") == runner.EXPECTED_PARTITIONS_SHA
    parts = json.loads((FROZEN / "partitions.json").read_text())
    if args.only:
        r, f = map(int, args.only.split(","))
        parts = [p for p in parts if p["repeat"] == r and p["fold"] == f]
        assert len(parts) == 1
    truth_file = np.load(FROZEN / "regressors/raw_truth940.npz")
    raw, reports = (None, [])
    for part in parts:
        start = time.monotonic()
        key = f"repeat{part['repeat']}_fold{part['fold']}"
        base = FROZEN / "gnn" / key
        checkpoint = base / "baseline/best.pt"
        prep = json.loads((base / "preparation.json").read_text())
        assert prep["partition"] == part
        with (base / "fitted_preprocessing.pkl").open("rb") as stream:
            ds = pickle.load(stream)
        assert (
            ds.splits.train == part["fitting"]
            and ds.splits.val == part["stopping"]
            and (ds.splits.test == part["evaluation"])
        )
        assert ds.target_residual_baseline is None
        assert ds.per_grain_transform == ds.per_graph_transform == "yeo_johnson"
        for spec in ds.per_grain_aux_targets[:3] + ds.aux_pergraph_targets[:3]:
            assert spec["transform"] == "yeo_johnson"
            assert spec.get("scale", 1) == spec.get("loss_weight", 1) == 1
        if raw is None:
            ds._raw = None
            ds._load()
            raw = ds._raw
        else:
            ds._raw = raw
        ids = np.asarray(part["evaluation"], dtype=np.int64)
        graphs = [ds._transform(raw[int(i)]) for i in ids]
        buckets = np.array([int(g.size_bucket[0]) for g in graphs], dtype=np.int64)
        counts = np.array([g.num_nodes for g in graphs], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(counts)]
        truth = np.array(
            [
                [
                    ds._pergraph_target_raw_by_source(raw[int(i)], c + "_pergraph_max")
                    for c in joint.CASES
                ]
                for i in ids
            ],
            dtype=np.float64,
        )
        assert np.array_equal(truth, truth_file["raw_truth"][ids])
        stat = np.load(base / "fitting_statistics.npz")
        values = joint.build_graphs(
            graphs, truth, buckets, stat["mean_by_size"], stat["sd_by_size"], False
        )
        model = runner.construct(
            prep["original_source_config"], part["initialization_seed"], device
        )
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        before = joint.state_hash(model)
        model.eval()
        pn, pg, nt, gt = forward(model, values, device, args.batch_size)
        with np.load(CACHE / f"{key}_best.npz") as cache:
            for name, value in [
                ("dataset_idx", ids),
                ("size_bucket", buckets),
                ("grain_count", counts),
                ("offsets", offsets),
                ("y_true", truth),
            ]:
                assert np.array_equal(cache[name], value), name
            errors = {}
            for name, value in [
                ("node_transformed", pn),
                ("graph_transformed", pg),
                ("node_target_transformed", nt),
                ("graph_target_transformed", gt),
            ]:
                np.testing.assert_allclose(
                    value, cache[name], atol=2e-05, rtol=2e-05, equal_nan=True
                )
                errors[name] = float(np.nanmax(np.abs(value - cache[name])))
            node_truth = cache["true_peak"].copy()
        fitted, transform_checks, mathematical_targets = ({}, {}, {})
        for level, yy, zz in [("node", node_truth, nt), ("graph", truth, gt)]:
            mathematical_targets[level] = np.empty_like(yy, dtype=np.float64)
            for c in range(4):
                st, scaler = components(ds, level, c)
                z = scaler.transform(
                    st["power_transformer"].transform(yy[:, c : c + 1])
                ).ravel()
                source_dtype = np.float32 if level == "node" and c == 0 else np.float64
                supervised = (
                    scaler.transform(
                        st["power_transformer"].transform(
                            yy[:, c : c + 1].astype(source_dtype)
                        )
                    )
                    .ravel()
                    .astype(np.float32)
                )
                np.testing.assert_array_equal(supervised, zz[:, c])
                inverse = (
                    st["power_transformer"]
                    .inverse_transform(scaler.inverse_transform(z[:, None]))
                    .ravel()
                )
                np.testing.assert_allclose(inverse, yy[:, c], rtol=1e-09, atol=1e-12)
                mathematical_targets[level][:, c] = z
                transform_checks[level + "_" + joint.CASES[c]] = dict(
                    source_dtype=np.dtype(source_dtype).name,
                    exact_supervised_reproduction=True,
                    max_abs_float64_vs_supervised=float(np.max(np.abs(z - zz[:, c]))),
                    max_abs_float64_roundtrip=float(np.max(np.abs(inverse - yy[:, c]))),
                )
                fitted[level + "_" + joint.CASES[c]] = {"state": st, "scaler": scaler}
        dropout_names = []
        for name, module in model.named_modules():
            if isinstance(module, DROP_TYPES):
                module.train()
                dropout_names.append(name)
        assert dropout_names
        assert all(
            (not m.training for m in model.modules() if not isinstance(m, DROP_TYPES))
        )
        seed = args.seed_base + 100 * part["repeat"] + part["fold"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()
        tick = time.monotonic()
        node_draws = np.empty((30, len(pn), 8), dtype=np.float32)
        graph_draws = np.empty((30, len(pg), 8), dtype=np.float32)
        for k in range(30):
            nd, gd, _, _ = forward(model, values, device, args.batch_size)
            node_draws[k] = nd[:, [0, 1, 2, 3, 16, 17, 18, 19]]
            graph_draws[k] = gd[:, [0, 1, 2, 3, 14, 15, 16, 17]]
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds = time.monotonic() - tick
        assert joint.state_hash(model) == before
        assert np.any(node_draws.std(0) > 0) and np.any(graph_draws.std(0) > 0)
        np.savez_compressed(
            args.out / f"{key}_draws.npz",
            node=node_draws,
            graph=graph_draws,
            node_truth=node_truth,
            graph_truth=truth,
            node_target_transformed=nt[:, :4],
            graph_target_transformed=gt[:, :4],
            node_target_mathematical=mathematical_targets["node"],
            graph_target_mathematical=mathematical_targets["graph"],
            deterministic_node=pn,
            deterministic_graph=pg,
            dataset_idx=ids,
            node_counts=counts,
            offsets=offsets,
            size_bucket=buckets,
            cases=np.array(joint.CASES),
            repeat=part["repeat"],
            fold=part["fold"],
            K=30,
            seed=seed,
            node_source_columns=np.array([0, 1, 2, 3, 16, 17, 18, 19]),
            graph_source_columns=np.array([0, 1, 2, 3, 14, 15, 16, 17]),
            node_grain_id=np.concatenate(
                [np.arange(n, dtype=np.int32) for n in counts]
            ),
        )
        with (args.out / f"{key}_fitted_maps.pkl").open("wb") as stream:
            pickle.dump(fitted, stream)
        report = dict(
            repeat=part["repeat"],
            fold=part["fold"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=joint.sha(checkpoint),
            preprocessing_sha256=joint.sha(base / "fitted_preprocessing.pkl"),
            state_hash_before_after=before,
            n_sves=len(ids),
            n_grains=len(pn),
            K=30,
            seed=seed,
            batch_size=args.batch_size,
            device=str(device),
            dropout_modules=dropout_names,
            deterministic_replay_max_abs=errors,
            transform_checks=transform_checks,
            capture_script_sha256=joint.sha(__file__),
            partitions_sha256=joint.sha(FROZEN / "partitions.json"),
            draw_archive_sha256=joint.sha(args.out / f"{key}_draws.npz"),
            maps_archive_sha256=joint.sha(args.out / f"{key}_fitted_maps.pkl"),
            scale_floor=-3.0,
            mc_forward_capture_seconds=forward_seconds,
            total_seconds=time.monotonic() - start,
        )
        (args.out / f"{key}_capture.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
        del model, values, graphs, node_draws, graph_draws
    (args.out / "capture_completion.json").write_text(
        json.dumps(
            dict(
                status="complete",
                policy="primary maximum-selected best.pt",
                reports=reports,
            ),
            indent=2,
        )
        + "\n"
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
