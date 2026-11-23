"""Read-only CPU replay of best/last checkpoints on stopping SVEs only."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
FROZEN = WORK / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
sys.path.insert(0, str(WORK / "revision_R2/studies_2026_11_15/inference"))
import replay_saved as replay

runner, joint = replay.runner, replay.joint
BRIDGE_PATH = WORK / "revision_R2/studies_2026_11_15/bridge/evaluate_bridges.py"
spec = importlib.util.spec_from_file_location("fixed_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
CASES, SIZES = list(joint.CASES), np.array(joint.SIZES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metric(y, pred):
    valid = np.isfinite(y)
    assert np.isfinite(pred).all()
    yy, pp = y[valid].astype(np.float64), pred[valid].astype(np.float64)
    assert len(yy) > 1
    sse = float(np.sum((pp - yy) ** 2))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    varp = float(np.sum((pp - pp.mean()) ** 2))
    return dict(
        n=int(len(yy)),
        missing=int((~valid).sum()),
        sse=sse,
        mse=sse / len(yy),
        rmse=float(np.sqrt(sse / len(yy))),
        r2=1 - sse / sst if sst > 0 else np.nan,
        pearson=float(np.corrcoef(yy, pp)[0, 1]) if sst > 0 and varp > 0 else np.nan,
    )


def extract(ds, ids, stat):
    """Transform and target-extract only allowed stopping objects."""
    allowed = {id(ds._raw[int(i)]) for i in ids}
    transformed_ids = []
    graphs = []
    for i in ids:
        source = ds._raw[int(i)]
        assert id(source) in allowed
        graphs.append(ds._transform(source))
        transformed_ids.append(int(i))
    assert transformed_ids == ids.tolist()
    buckets = np.array([int(g.size_bucket[0]) for g in graphs], dtype=np.int64)
    counts = np.array([g.num_nodes for g in graphs], dtype=np.int64)
    truth = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(ds._raw[int(i)], c + "_pergraph_max")
                for c in CASES
            ]
            for i in ids
        ],
        dtype=np.float64,
    )
    values = joint.build_graphs(
        graphs, truth, buckets, stat["mean_by_size"], stat["sd_by_size"], False
    )
    for i, g in zip(ids, values):
        g.sve_id = torch.tensor([i], dtype=torch.long)
    node_truth = np.concatenate(
        [
            np.column_stack(
                [
                    getattr(ds._raw[int(i)], ds.per_grain_target_field)[
                        :, ds.per_grain_target_col
                    ].numpy()
                ]
                + [
                    getattr(ds._raw[int(i)], s.get("field", "fs_fip"))[
                        :, int(s["col"])
                    ].numpy()
                    for s in ds.per_grain_aux_targets
                ]
            )
            for i in ids
        ]
    ).astype(np.float64)
    true_peak = np.concatenate(
        [
            np.column_stack(
                [getattr(ds._raw[int(i)], c + "_fip_dist")[:, 0].numpy() for c in CASES]
            )
            for i in ids
        ]
    ).astype(np.float64)
    node_truth[:, :4] = true_peak
    mask = np.concatenate([ds._raw[int(i)].per_grain_aux_nan_mask.numpy() for i in ids])
    assert mask.shape == (len(node_truth), 15)
    node_truth[:, 1:][mask] = np.nan
    graph_truth = np.column_stack(
        [truth[:, 0]]
        + [
            np.array(
                [
                    ds._pergraph_target_raw_by_source(ds._raw[int(i)], s["source"])
                    for i in ids
                ]
            )
            for s in ds.aux_pergraph_targets
        ]
    )
    assert graph_truth.shape == (75, 14)
    assert np.array_equal(graph_truth[:, :4], truth)
    offsets = np.r_[0, np.cumsum(counts)]
    for j in range(len(ids)):
        assert np.array_equal(true_peak[offsets[j] : offsets[j + 1]].max(0), truth[j])
    return dict(
        ids=ids,
        values=values,
        truth=truth,
        buckets=buckets,
        counts=counts,
        offsets=offsets,
        node_truth=node_truth,
        graph_truth=graph_truth,
        true_peak=true_peak,
        mask=mask,
        transformed_ids=transformed_ids,
        node_names=np.array(
            ["lcf_fs_peak"] + [s["name"] for s in ds.per_grain_aux_targets]
        ),
        graph_names=np.array(
            ["lcf_fs_max"] + [s["name"] for s in ds.aux_pergraph_targets]
        ),
    )


def infer(model, ds, inp, stat):
    model.eval()
    pn_all, pg_all, nt_all, gt_all, direct, seen, batches = [], [], [], [], [], [], []
    with torch.no_grad():
        for k, batch in enumerate(
            DataLoader(inp["values"], batch_size=16, shuffle=False, num_workers=0)
        ):
            batch_ids = batch.sve_id.tolist()
            assert set(batch_ids).issubset(set(inp["ids"].tolist()))
            pn, pg = model(batch)
            assert pn.shape[1] == 20 and pg.shape[1] == 18
            assert torch.isfinite(pn).all() and torch.isfinite(pg).all()
            ln = model.loss_function(pn, batch.pn_target)
            lg = model.loss_function(pg, batch.pg_target)
            l0 = 0.5 * ln + 0.5 * lg
            assert all(torch.isfinite(x) for x in (ln, lg, l0))
            batches.append(
                dict(
                    batch=k,
                    n_sves=batch.num_graphs,
                    n_grains=pn.shape[0],
                    ids=json.dumps(batch_ids),
                    joint_loss=l0.item(),
                    node_loss=ln.item(),
                    graph_loss=lg.item(),
                )
            )
            seen.extend(batch_ids)
            pn_all.append(pn.numpy())
            pg_all.append(pg.numpy())
            nt_all.append(batch.pn_target.numpy())
            gt_all.append(batch.pg_target.numpy())
            direct.append(
                joint.inverse_graph(
                    pg,
                    batch.size_bucket.flatten().long(),
                    False,
                    stat["mean_by_size"],
                    stat["sd_by_size"],
                    replay.components_for(ds),
                ).numpy()
            )
    assert seen == inp["ids"].tolist()
    pn, pg, nt, gt, maximum = map(
        np.concatenate, (pn_all, pg_all, nt_all, gt_all, direct)
    )
    nb = np.repeat(inp["buckets"], inp["counts"])
    pn64, pg64 = pn.astype(np.float64), pg.astype(np.float64)
    raw_node = np.column_stack(
        [ds.inverse_pergrain_target(pn64[:, 0], bucket_ids=nb)]
        + [
            ds.inverse_per_grain_aux_target(c - 1, pn64[:, c], bucket_ids=nb)
            for c in range(1, 16)
        ]
    )
    raw_graph = np.column_stack(
        [ds.inverse_pergraph_target(pg64[:, 0])]
        + [
            ds.inverse_aux_pergraph_target(c - 1, pg64[:, c], bucket_ids=inp["buckets"])
            for c in range(1, 14)
        ]
    )
    np.testing.assert_allclose(raw_graph[:, :4], maximum, rtol=1e-5, atol=1e-8)
    assert all(
        np.isfinite(x).all() for x in (pn, pg, raw_node, raw_graph, maximum, nt, gt)
    )
    score, cells = joint.selection_score(
        inp["truth"], maximum, inp["buckets"], stat["checkpoint_variance_by_size"]
    )
    operational, grain_max = np.zeros_like(maximum), np.zeros_like(maximum)
    fallback = np.zeros(maximum.shape, dtype=bool)
    selected_count = np.zeros(maximum.shape, dtype=np.int64)
    for j in range(len(inp["ids"])):
        for c in range(4):
            x = raw_node[inp["offsets"][j] : inp["offsets"][j + 1], c]
            sel, _, _ = bridge.selected_population(x, "original_recut")
            val, failure = bridge.untruncated(sel, int(inp["counts"][j]), len(x))
            fallback[j, c] = failure != "none"
            selected_count[j, c] = len(sel)
            operational[j, c] = (
                stat["mean_by_size"][inp["buckets"][j], c] if fallback[j, c] else val
            )
            grain_max[j, c] = x.max()
    assert np.isfinite(operational).all() and np.isfinite(grain_max).all()
    return dict(
        pn=pn,
        pg=pg,
        nt=nt,
        gt=gt,
        raw_node=raw_node,
        raw_graph=raw_graph,
        maximum=maximum,
        operational=operational,
        grain_max=grain_max,
        fallback=fallback,
        selected_count=selected_count,
        batches=batches,
        score=score,
        cells=cells,
        joint_loss=sum(b["n_sves"] * b["joint_loss"] for b in batches) / 75,
        node_loss=sum(b["n_sves"] * b["node_loss"] for b in batches) / 75,
        graph_loss=sum(b["n_sves"] * b["graph_loss"] for b in batches) / 75,
    )


def parity(inp, result, archive, stat, grain_sd):
    with np.load(archive, allow_pickle=False) as z:
        for a, b in (
            ("dataset_idx", "ids"),
            ("size_bucket", "buckets"),
            ("grain_count", "counts"),
            ("offsets", "offsets"),
            ("y_true", "truth"),
            ("true_peak", "true_peak"),
        ):
            assert np.array_equal(z[a], inp[b]), a
        assert str(z["subset"]) == "stopping" and list(z["cases"]) == CASES
        assert np.array_equal(
            z["grain_dataset_idx"], np.repeat(inp["ids"], inp["counts"])
        )
        assert np.array_equal(
            z["grain_id"], np.concatenate([np.arange(n) for n in inp["counts"]])
        )
        ed = np.abs(result["maximum"] - z["direct_prediction"])
        eg = np.abs(result["raw_node"][:, :4] - z["pred_peak"])
    ds = float((ed / stat["sd_by_size"][inp["buckets"]]).max())
    gs = float((eg / grain_sd).max())
    assert ds < 1e-4 and gs < 1e-4, (ds, gs)
    return dict(
        status="pass",
        tolerance_in_fitting_sd=1e-4,
        maximum_abs_direct_by_case=ed.max(0).tolist(),
        maximum_abs_grain_by_case=eg.max(0).tolist(),
        direct_max_error_in_fitting_size_sd=ds,
        grain_max_error_in_fitting_case_sd=gs,
    )


def save_archive(path, inp, result, part, label, epoch):
    np.savez_compressed(
        path,
        checkpoint=label,
        epoch_zero_based=epoch,
        repeat=part["repeat"],
        fold=part["fold"],
        subset="stopping",
        cases=np.array(CASES),
        dataset_idx=inp["ids"],
        size_bucket=inp["buckets"],
        size_um=SIZES[inp["buckets"]],
        grain_count=inp["counts"],
        offsets=inp["offsets"],
        grain_dataset_idx=np.repeat(inp["ids"], inp["counts"]),
        grain_id=np.concatenate([np.arange(n, dtype=np.int32) for n in inp["counts"]]),
        node_transformed=result["pn"],
        graph_transformed=result["pg"],
        node_target_transformed=result["nt"],
        graph_target_transformed=result["gt"],
        node_mean_raw=result["raw_node"],
        node_target_raw=inp["node_truth"],
        node_mean_names=inp["node_names"],
        node_aux_source_nan_mask=inp["mask"],
        graph_mean_raw=result["raw_graph"],
        graph_target_raw=inp["graph_truth"],
        graph_mean_names=inp["graph_names"],
        pred_peak=result["raw_node"][:, :4],
        true_peak=inp["true_peak"],
        y_true=inp["truth"],
        direct_prediction=result["maximum"],
        bridge_prediction=result["operational"],
        maximum_predicted_grain=result["grain_max"],
        bridge_fallback=result["fallback"],
        bridge_selected_count=result["selected_count"],
        raw_maximum_equal20_score=result["score"],
        raw_maximum_equal20_cells=result["cells"],
    )


def metric_rows(part, label, inp, result):
    rows = []
    node_sizes = np.repeat(SIZES[inp["buckets"]], inp["counts"])
    graph_sizes = SIZES[inp["buckets"]]
    groups = [
        (
            "grain_peak",
            CASES,
            inp["node_truth"][:, :4],
            result["raw_node"][:, :4],
            node_sizes,
        ),
        (
            "mechanical",
            inp["graph_names"][8:14],
            inp["graph_truth"][:, 8:14],
            result["raw_graph"][:, 8:14],
            graph_sizes,
        ),
        ("legacy_maximum", CASES, inp["truth"], result["maximum"], graph_sizes),
        (
            "uncalibrated_operational_bridge",
            CASES,
            inp["truth"],
            result["operational"],
            graph_sizes,
        ),
        (
            "maximum_predicted_grain",
            CASES,
            inp["truth"],
            result["grain_max"],
            graph_sizes,
        ),
    ]
    for target, names, truth, pred, sizes in groups:
        for size in ["pooled"] + list(SIZES):
            subset = (
                np.ones(len(truth), dtype=bool) if size == "pooled" else sizes == size
            )
            for c, name in enumerate(names):
                rows.append(
                    dict(
                        repeat=part["repeat"],
                        fold=part["fold"],
                        checkpoint=label,
                        target=target,
                        name=name,
                        size_um=str(size),
                        **metric(truth[subset, c], pred[subset, c]),
                    )
                )
    return rows


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(35500 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    sources = [
        Path(__file__),
        HERE / "PRESPEC.txt",
        Path(replay.__file__),
        Path(runner.__file__),
        Path(joint.__file__),
        Path(joint.dataset.__file__),
        Path(joint.losses.__file__),
        Path(joint.train.__file__),
        joint.CONTROL / "create_seed_controlled.py",
        BRIDGE_PATH,
        HERE.parent / "checkpoint_selection/SELECTION_AUDIT.json",
    ]
    source_hashes = {str(p): sha(p) for p in sources}
    assert sha(FROZEN / "partitions.json") == runner.EXPECTED_PARTITIONS_SHA
    parts = json.loads((FROZEN / "partitions.json").read_text())
    all_metrics, all_batches, summaries, checks = [], [], [], []
    raw = None
    for part in parts:
        key = f"repeat{part['repeat']}_fold{part['fold']}"
        pair = FROZEN / "gnn" / key
        prep = json.loads((pair / "preparation.json").read_text())
        completion = json.loads((pair / "baseline/completion.json").read_text())
        assert prep["partition"] == part
        ids = np.array(part["stopping"], dtype=np.int64)
        assert len(ids) == len(set(ids)) == 75
        assert not (
            set(ids) & set(part["evaluation"]) or set(ids) & set(part["fitting"])
        )
        with (pair / "fitted_preprocessing.pkl").open("rb") as f:
            ds = pickle.load(f)
        assert (
            ds.splits.train == part["fitting"]
            and ds.splits.val == part["stopping"]
            and ds.splits.test == part["evaluation"]
        )
        if raw is None:
            raw_sha = sha(__import__("fip_release_paths").rebase_path(ds.pt_path))
            assert (
                raw_sha
                == "cf6a9c5a2b3b210a58f11429f9dc1bd2f8e923c681983d6d4a10c754cdee7615"
            )
            ds._raw = None
            ds._load()
            raw = ds._raw
        else:
            ds._raw = raw
        with np.load(pair / "fitting_statistics.npz") as z:
            stat = {k: z[k] for k in z.files}
        inp = extract(ds, ids, stat)
        saved = (
            WORK
            / "revision_R2/studies_2026_11_15/data"
            / f"grain_predictions_stopping_{key}.npz"
        )
        quality = (
            pd.read_csv(pair / "baseline/stopping_grain_quality.csv")
            .set_index("case")
            .loc[CASES]
        )
        grain_sd = quality.fitting_case_sd.to_numpy()
        inputs = [
            pair / "preparation.json",
            pair / "fitted_preprocessing.pkl",
            pair / "fitting_statistics.npz",
            pair / "baseline/best.pt",
            pair / "baseline/last.pt",
            pair / "baseline/history.json",
            pair / "baseline/completion.json",
            pair / "baseline/stopping_grain_quality.csv",
            saved,
        ]
        hashes_before = {str(p): sha(p) for p in inputs}
        assert (
            hashes_before[str(pair / "baseline/best.pt")]
            == completion["checkpoint_sha256"]
        )
        best_result, best_parity = None, None
        fold_check = dict(
            repeat=part["repeat"],
            fold=part["fold"],
            stopping_ids=ids.tolist(),
            transformed_ids=inp["transformed_ids"],
            input_sha256=hashes_before,
            original_device=completion["device"],
            replay_device="cpu",
            checkpoints=[],
        )
        for label in ["best", "last"]:
            if label == "last":
                assert best_parity["status"] == "pass"
            model = runner.construct(
                prep["original_source_config"],
                part["initialization_seed"],
                torch.device("cpu"),
            )
            state = torch.load(
                pair / "baseline" / (label + ".pt"),
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(state["model_state_dict"], strict=True)
            result = infer(model, ds, inp, stat)
            if label == "best":
                best_parity = parity(inp, result, saved, stat, grain_sd)
                best_result = result
            else:
                assert np.array_equal(best_result["nt"], result["nt"])
                assert np.array_equal(best_result["gt"], result["gt"])
                assert [b["ids"] for b in best_result["batches"]] == [
                    b["ids"] for b in result["batches"]
                ]
            output = HERE / f"{key}_{label}.npz"
            save_archive(output, inp, result, part, label, int(state["epoch"]))
            all_metrics.extend(metric_rows(part, label, inp, result))
            all_batches.extend(
                [
                    dict(
                        repeat=part["repeat"], fold=part["fold"], checkpoint=label, **b
                    )
                    for b in result["batches"]
                ]
            )
            summary = dict(
                repeat=part["repeat"],
                fold=part["fold"],
                checkpoint=label,
                epoch_zero_based=int(state["epoch"]),
                joint_loss=result["joint_loss"],
                node_loss=result["node_loss"],
                graph_loss=result["graph_loss"],
                raw_maximum_equal20_score=result["score"],
                bridge_fallback_count=int(result["fallback"].sum()),
                n_sve_case=300,
            )
            summaries.append(summary)
            fold_check["checkpoints"].append(
                dict(
                    **summary,
                    all_predictions_finite=True,
                    output_widths=[20, 18],
                    source_aux_missing_entries=int(inp["mask"].sum()),
                    output_path=str(output),
                    output_sha256=sha(output),
                    forwarded_ids=ids.tolist(),
                    no_outer_forward=True,
                )
            )
            print(
                json.dumps(dict(event="STOPPING_CHECKPOINT_COMPLETE", **summary)),
                flush=True,
            )
            del state, model, result
        fold_check["best_replay_parity"] = best_parity
        fold_check["inputs_unchanged"] = hashes_before == {
            str(p): sha(p) for p in inputs
        }
        assert fold_check["inputs_unchanged"]
        checks.append(fold_check)
        dump(
            HERE / "progress.json",
            dict(status="running", completed_folds=len(checks), checks=checks),
        )
        del ds, inp, best_result
    frame = pd.DataFrame(all_metrics)
    frame.to_csv(HERE / "metrics_per_fold.csv", index=False)
    pd.DataFrame(all_batches).to_csv(HERE / "batch_losses.csv", index=False)
    pd.DataFrame(summaries).to_csv(HERE / "checkpoint_summaries.csv", index=False)
    keys = ["repeat", "fold", "target", "name", "size_um"]
    paired = frame[frame.checkpoint == "best"].merge(
        frame[frame.checkpoint == "last"],
        on=keys,
        suffixes=("_best", "_last"),
        validate="one_to_one",
    )
    for col in ["mse", "rmse", "r2", "pearson"]:
        paired[col + "_last_minus_best"] = paired[col + "_last"] - paired[col + "_best"]
    paired["rmse_percent_change"] = 100 * (paired.rmse_last / paired.rmse_best - 1)
    paired.to_csv(HERE / "paired_metrics.csv", index=False)
    aggregate = []
    for key, g in paired.groupby(["target", "name", "size_um"], sort=False):
        aggregate.append(
            dict(zip(["target", "name", "size_um"], key))
            | dict(
                folds=len(g),
                best_mean_rmse=g.rmse_best.mean(),
                last_mean_rmse=g.rmse_last.mean(),
                best_mean_r2=g.r2_best.mean(),
                last_mean_r2=g.r2_last.mean(),
                best_mean_pearson=g.pearson_best.mean(),
                last_mean_pearson=g.pearson_last.mean(),
                mean_rmse_percent_change=g.rmse_percent_change.mean(),
                median_rmse_percent_change=g.rmse_percent_change.median(),
                folds_rmse_improved=int((g.rmse_last < g.rmse_best).sum()),
                folds_r2_improved=int((g.r2_last > g.r2_best).sum()),
                observation_weighted_best_rmse=float(
                    np.sqrt(g.sse_best.sum() / g.n_best.sum())
                ),
                observation_weighted_last_rmse=float(
                    np.sqrt(g.sse_last.sum() / g.n_last.sum())
                ),
            )
        )
    pd.DataFrame(aggregate).to_csv(HERE / "descriptive_aggregates.csv", index=False)
    assert all(sha(p) == v for p, v in source_hashes.items())
    verification = dict(
        status="pass",
        created_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        cpu_only=True,
        no_training=True,
        no_calibration=True,
        source_sha256=source_hashes,
        partitions_sha256=runner.EXPECTED_PARTITIONS_SHA,
        raw_source_sha256=raw_sha,
        raw_container_loaded=True,
        raw_outer_targets_extracted=False,
        outer_graphs_transformed=0,
        outer_graphs_forwarded=0,
        outer_scores_computed=0,
        exactly_75_stopping_per_fold=True,
        best_verified_before_last_in_each_fold=True,
        source_20_18_contract_preserved=True,
        joint_loss_original_targets_preserved=True,
        all_inputs_sources_unchanged=True,
        scope="Stopping-only development diagnosis; labels already shaped checkpoint selection and trajectories. No production checkpoint selection.",
        fold_checks=checks,
        output_sha256={str(p): sha(p) for p in HERE.glob("*.csv")},
    )
    dump(HERE / "verification.json", verification)
    dump(
        HERE / "progress.json",
        dict(
            status="complete",
            completed_folds=10,
            elapsed_seconds=verification["elapsed_seconds"],
        ),
    )
    torch.distributed.destroy_process_group()
    print(
        "PROBE_COMPLETE",
        json.dumps(dict(seconds=verification["elapsed_seconds"], folds=10)),
        flush=True,
    )


if __name__ == "__main__":
    main()
