"""Preparation, gated training and complete exports for matched joint branches."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
from pathlib import Path
import argparse
import gc
import hashlib
import json
import pickle
import socket
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
FROZEN = WORK / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
sys.path.insert(0, str(WORK / "revision_R2/studies_2026_11_15/inference"))
import replay_saved as replay
from branch_model import JointRawBranch, balanced_score

runner, joint = (replay.runner, replay.joint)
CASES, SIZES = (joint.CASES, np.array(joint.SIZES))
ARMS = ("detached", "attached")
MAX_EPOCHS, PATIENCE = (120, 50)
_RAW_CACHE = None
_RAW_SOURCE_SHA = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_manifest():
    paths = [
        HERE / "study.py",
        HERE / "branch_model.py",
        HERE / "smoke.py",
        HERE / "PRESPEC.txt",
        Path(replay.__file__),
        Path(runner.__file__),
        Path(joint.__file__),
        joint.CONTROL / "create_seed_controlled.py",
        Path(joint.dataset.__file__),
        Path(joint.losses.__file__),
        Path(joint.train.__file__),
        joint.ROOT / "hydragnn/models/Base.py",
        joint.ROOT / "hydragnn/models/SAGEStack.py",
    ]
    return {str(p): sha(p) for p in paths}


def manifest_id():
    return hashlib.sha256(
        json.dumps(source_manifest(), sort_keys=True).encode()
    ).hexdigest()[:16]


def preparation_dir(part):
    return (
        HERE
        / "preparation"
        / manifest_id()
        / f"repeat{part['repeat']}_fold{part['fold']}"
    )


def load_partitions():
    path = FROZEN / "partitions.json"
    assert sha(path) == runner.EXPECTED_PARTITIONS_SHA
    parts = json.loads(path.read_text())
    assert len(parts) == 10
    for p in parts:
        a, b, c = (set(p[k]) for k in ("fitting", "stopping", "evaluation"))
        assert not (a & b or a & c or b & c) and a | b | c == set(range(940))
    return parts


def fold_inputs(part):
    """Restore saved transforms; extract only fitting/stopping target arrays."""
    global _RAW_CACHE, _RAW_SOURCE_SHA
    pair = FROZEN / "gnn" / f"repeat{part['repeat']}_fold{part['fold']}"
    prep = json.loads((pair / "preparation.json").read_text())
    assert prep["partition"] == part
    cfg = prep["original_source_config"]
    with (pair / "fitted_preprocessing.pkl").open("rb") as f:
        ds = pickle.load(f)
    assert ds.splits.train == part["fitting"]
    assert ds.splits.val == part["stopping"] and ds.splits.test == part["evaluation"]
    if _RAW_CACHE is None:
        _RAW_SOURCE_SHA = sha(__import__("fip_release_paths").rebase_path(ds.pt_path))
        assert (
            _RAW_SOURCE_SHA
            == "cf6a9c5a2b3b210a58f11429f9dc1bd2f8e923c681983d6d4a10c754cdee7615"
        )
        ds._raw = None
        ds._load()
        _RAW_CACHE = ds._raw
    else:
        ds._raw = _RAW_CACHE
    assert len(ds._raw) == 940
    ids = np.array(part["fitting"] + part["stopping"], dtype=np.int64)
    graphs = [ds._transform(ds._raw[int(i)]) for i in ids]
    raw = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(ds._raw[int(i)], c + "_pergraph_max")
                for c in CASES
            ]
            for i in ids
        ],
        dtype=np.float64,
    )
    buckets = np.array([int(g.size_bucket[0]) for g in graphs], dtype=np.int64)
    n = len(part["fitting"])
    counts = np.array([g.num_nodes for g in graphs], dtype=np.int64)
    mean = np.stack([raw[:n][buckets[:n] == b].mean(0) for b in range(5)])
    population_sd = np.stack(
        [raw[:n][buckets[:n] == b].std(0, ddof=0) for b in range(5)]
    )
    sample_variance = np.stack(
        [raw[:n][buckets[:n] == b].var(0, ddof=1) for b in range(5)]
    )
    floor = np.maximum(1e-06 * raw[:n].var(0, ddof=1), 1e-30)
    variance = np.maximum(sample_variance, floor[None, :])
    assert np.array_equal(
        variance, sample_variance
    ), "Variance floor activated; review before fitting"
    log_count = np.log(counts[:n].astype(np.float64))
    count_mean, count_sd = (float(log_count.mean()), float(log_count.std(ddof=0)))
    cell_n = np.bincount(buckets[:n], minlength=5)
    weights = n / (5.0 * cell_n)
    with np.load(pair / "fitting_statistics.npz") as f:
        saved = {k: f[k].copy() for k in f.files}
    assert np.array_equal(ids, saved["dataset_idx"])
    assert np.array_equal(buckets, saved["size_bucket"])
    for actual, key in (
        (mean, "mean_by_size"),
        (population_sd, "sd_by_size"),
        (sample_variance, "checkpoint_variance_by_size"),
    ):
        assert np.array_equal(actual, saved[key]), key
    permutations = saved["permutations"]
    assert permutations.shape == (120, n)
    assert all((np.array_equal(np.sort(row), np.arange(n)) for row in permutations))
    values = joint.build_graphs(graphs, raw, buckets, mean, population_sd, False)
    for i, g in enumerate(values):
        g.sve_id = torch.tensor([ids[i]], dtype=torch.long)
    train_cfg = cfg["NeuralNetwork"]["Training"]
    assert train_cfg["Optimizer"]["type"] == "AdamW"
    assert float(train_cfg["Optimizer"]["learning_rate"]) == 0.0001586668862464421
    assert float(train_cfg["weight_decay"]) == 1e-05
    assert train_cfg["lr_scheduler"] == "reduce_on_plateau"
    assert (
        train_cfg["lr_scheduler_factor"],
        train_cfg["lr_scheduler_patience"],
        train_cfg["lr_scheduler_min_lr"],
    ) == (0.3, 15, 1e-07)
    return dict(
        part=part,
        pair=pair,
        prep=prep,
        cfg=cfg,
        ds=ds,
        ids=ids,
        graphs=graphs,
        raw=raw,
        buckets=buckets,
        counts=counts,
        values=values,
        n=n,
        mean=mean,
        population_sd=population_sd,
        variance=variance,
        sample_variance=sample_variance,
        variance_floor=floor,
        count_mean=count_mean,
        count_sd=count_sd,
        cell_weights=weights,
        cell_counts=cell_n,
        permutations=permutations,
        raw_source_path=str(ds.pt_path),
        raw_source_sha256=_RAW_SOURCE_SHA,
        components=replay.components_for(ds),
    )


def head_seed(part):
    return 2026111600 + int(part["repeat"]) * 10 + int(part["fold"])


def construct(inp, arm, device):
    part = inp["part"]
    core = runner.construct(inp["cfg"], part["initialization_seed"], device)
    expected = inp["prep"]["full_initial_state_sha256"]
    assert joint.state_hash(core) == expected
    before_cpu = torch.get_rng_state().clone()
    before_gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    wrapper = JointRawBranch(
        core,
        arm,
        inp["mean"],
        inp["variance"],
        inp["count_mean"],
        inp["count_sd"],
        inp["cell_weights"],
        head_seed(part),
    ).to(device)
    assert joint.state_hash(wrapper.core) == expected
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert all(
        (torch.equal(a, b) for a, b in zip(before_gpu, torch.cuda.get_rng_state_all()))
    )
    assert sum((p.numel() for p in wrapper.parameters())) == 394746
    return wrapper


def prepare(inp):
    p = inp["part"]
    dest = preparation_dir(p)
    a = construct(inp, "detached", torch.device("cpu"))
    b = construct(inp, "attached", torch.device("cpu"))
    assert joint.state_hash(a) == joint.state_hash(b)
    n = inp["n"]
    dummy = np.arange(n * 4, dtype=float).reshape(n, 4) / (n * 4)
    expected = np.stack(
        [dummy[inp["buckets"][:n] == k].mean(0) for k in range(5)]
    ).mean()
    actual = (dummy.mean(1) * inp["cell_weights"][inp["buckets"][:n]]).mean()
    assert np.isclose(actual, expected, rtol=1e-14, atol=1e-14)
    state_path = dest / "initial_state.pt"
    if dest.exists():
        report = json.loads((dest / "preparation.json").read_text())
        assert report["source_manifest"] == source_manifest()
        assert report["partition"] == p
        assert report["paired_full_state_sha256"] == joint.state_hash(a)
        assert report["preprocessing_sha256"] == sha(
            inp["pair"] / "fitted_preprocessing.pkl"
        )
        assert report["source_preparation_sha256"] == sha(
            inp["pair"] / "preparation.json"
        )
        assert report["saved_batch_orders_sha256"] == array_sha(inp["permutations"])
        assert report["raw_source_sha256"] == inp["raw_source_sha256"]
        assert report["initial_file_sha256"] == sha(state_path)
        assert report["statistics_file_sha256"] == sha(dest / "fitting_statistics.npz")
        loaded = torch.load(state_path, weights_only=True, map_location="cpu")
        assert loaded.keys() == a.state_dict().keys()
        assert all((torch.equal(loaded[k], v.cpu()) for k, v in a.state_dict().items()))
        return report
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=False)
    torch.save(a.state_dict(), state_path)
    np.savez_compressed(
        dest / "fitting_statistics.npz",
        dataset_idx=inp["ids"],
        size_bucket=inp["buckets"],
        mean_by_size=inp["mean"],
        legacy_population_sd=inp["population_sd"],
        sample_variance=inp["variance"],
        variance_floor=inp["variance_floor"],
        count_mean=inp["count_mean"],
        count_sd=inp["count_sd"],
        cell_weights=inp["cell_weights"],
        cell_counts=inp["cell_counts"],
        permutations=inp["permutations"],
    )
    report = dict(
        status="prepared_not_trained",
        partition=p,
        full_core_state_sha256=joint.state_hash(a.core),
        paired_full_state_sha256=joint.state_hash(a),
        initial_file_sha256=sha(state_path),
        statistics_file_sha256=sha(dest / "fitting_statistics.npz"),
        preparation_manifest_id=manifest_id(),
        parameters=dict(core=377974, branch=16772, total=394746),
        new_head_seed=head_seed(p),
        core_rng_preserved=True,
        all_current_fit_statistics_exact=True,
        sample_variance_floor_active=False,
        size_weight_identity=True,
        target_extract_ids=inp["ids"].tolist(),
        outer_targets_extracted=False,
        outer_graphs_transformed=False,
        preprocessing_sha256=sha(inp["pair"] / "fitted_preprocessing.pkl"),
        source_preparation_sha256=sha(inp["pair"] / "preparation.json"),
        raw_source_path=inp["raw_source_path"],
        raw_source_sha256=inp["raw_source_sha256"],
        saved_batch_orders_sha256=array_sha(inp["permutations"]),
        fit_count_mean=inp["count_mean"],
        fit_count_sd=inp["count_sd"],
        source_manifest=source_manifest(),
    )
    dump(dest / "preparation.json", report)
    return report


def clip_counts(pn, pg, inp):
    ds = inp["ds"]
    result = {"node_peak": [], "legacy_maximum": []}
    for c in range(4):
        scaler = ds._scaler_pergrain_y if c == 0 else ds._scaler_per_grain_aux_y[c - 1]
        state = (
            ds._pergrain_target_state
            if c == 0
            else ds._per_grain_aux_target_states[c - 1]
        )
        z = pn[:, c] * float(scaler.scale_[0]) + float(scaler.mean_[0])
        result["node_peak"].append(
            int(((z < state["z_min_safe"]) | (z > state["z_max_safe"])).sum())
        )
        cp = inp["components"][c]
        z = pg[:, c] * cp["outer_sd"] + cp["outer_mean"]
        result["legacy_maximum"].append(
            int(((z < cp["clip_low"]) | (z > cp["clip_high"])).sum())
        )
    return result


def evaluate(model, inp, graphs, ids, raw_truth, buckets):
    model.eval()
    rows, legacy, seen, transformed_graph = ([], [], [], [])
    clips = dict(
        node_peak=np.zeros(4, dtype=np.int64),
        legacy_maximum=np.zeros(4, dtype=np.int64),
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False, num_workers=0):
            batch = batch.to(device)
            pn, pg, raw = model(batch)
            rows.append(raw.cpu().numpy())
            transformed_graph.append(pg.cpu().numpy())
            legacy.append(
                joint.inverse_graph(
                    pg,
                    batch.size_bucket.flatten().long(),
                    False,
                    inp["mean"],
                    inp["population_sd"],
                    inp["components"],
                )
                .cpu()
                .numpy()
            )
            seen.extend(batch.sve_id.cpu().tolist())
            bc = clip_counts(pn, pg, inp)
            for k in clips:
                clips[k] += bc[k]
    assert np.array_equal(seen, ids)
    pred, old = (np.concatenate(rows), np.concatenate(legacy))
    score, cells = balanced_score(raw_truth, pred, buckets, inp["variance"])
    assert np.isfinite(old).all()
    return dict(
        score=score,
        cells=cells,
        prediction=pred,
        legacy=old,
        negative_counts=(pred < 0).sum(0).tolist(),
        inverse_clip_counts={k: v.tolist() for k, v in clips.items()},
        transformed_graph=np.concatenate(transformed_graph),
    )


def subset_data(inp, subset):
    if subset in ("fitting", "stopping"):
        sl = slice(0, inp["n"]) if subset == "fitting" else slice(inp["n"], None)
        return (inp["values"][sl], inp["ids"][sl], inp["raw"][sl], inp["buckets"][sl])
    if subset != "evaluation":
        raise ValueError(subset)
    ids = np.array(inp["part"]["evaluation"], dtype=np.int64)
    ds = inp["ds"]
    graphs = [ds._transform(ds._raw[int(i)]) for i in ids]
    raw = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(ds._raw[int(i)], c + "_pergraph_max")
                for c in CASES
            ]
            for i in ids
        ],
        dtype=np.float64,
    )
    with np.load(FROZEN / "regressors/raw_truth940.npz") as f:
        assert np.array_equal(raw, f["raw_truth"][ids])
    buckets = np.array([int(g.size_bucket[0]) for g in graphs])
    values = joint.build_graphs(
        graphs, raw, buckets, inp["mean"], inp["population_sd"], False
    )
    for i, g in enumerate(values):
        g.sve_id = torch.tensor([ids[i]], dtype=torch.long)
    return (values, ids, raw, buckets)


def export_subset(model, inp, subset, dest, explicit=None):
    graphs, ids, truth, buckets = (
        subset_data(inp, subset) if explicit is None else explicit
    )
    ds, part = (inp["ds"], inp["part"])
    endpoint = evaluate(model, inp, graphs, ids, truth, buckets)
    device = next(model.parameters()).device
    pn_all, pg_all, nt_all, gt_all = ([], [], [], [])
    seen = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False, num_workers=0):
            batch = batch.to(device)
            pn, pg, _ = model(batch)
            pn_all.append(pn.cpu().numpy())
            pg_all.append(pg.cpu().numpy())
            nt_all.append(batch.pn_target.cpu().numpy())
            gt_all.append(batch.pg_target.cpu().numpy())
            seen.extend(batch.sve_id.cpu().tolist())
    assert np.array_equal(seen, ids)
    pn, pg, nt, gt = map(np.concatenate, (pn_all, pg_all, nt_all, gt_all))
    counts = np.array([ds._raw[int(i)].num_nodes for i in ids], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    nb = np.repeat(buckets, counts)
    pn64, pg64 = (pn.astype(np.float64), pg.astype(np.float64))
    raw_node = np.column_stack(
        [ds.inverse_pergrain_target(pn64[:, 0], bucket_ids=nb)]
        + [
            ds.inverse_per_grain_aux_target(c - 1, pn64[:, c], bucket_ids=nb)
            for c in range(1, 16)
        ]
    )
    raw_node_truth = np.concatenate(
        [
            np.column_stack(
                [
                    getattr(ds._raw[int(i)], ds.per_grain_target_field)[
                        :, ds.per_grain_target_col
                    ].numpy()
                ]
                + [
                    getattr(ds._raw[int(i)], spec.get("field", "fs_fip"))[
                        :, int(spec["col"])
                    ].numpy()
                    for spec in ds.per_grain_aux_targets
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
    raw_node_truth[:, :4] = true_peak
    nan_mask = np.concatenate(
        [ds._raw[int(i)].per_grain_aux_nan_mask.numpy() for i in ids]
    )
    assert nan_mask.shape == (len(pn), 15)
    raw_node_truth[:, 1:][nan_mask] = np.nan
    raw_graph = np.column_stack(
        [ds.inverse_pergraph_target(pg64[:, 0])]
        + [
            ds.inverse_aux_pergraph_target(c - 1, pg64[:, c], bucket_ids=buckets)
            for c in range(1, 14)
        ]
    )
    graph_specs = ds.aux_pergraph_targets
    raw_graph_truth = np.column_stack(
        [truth[:, 0]]
        + [
            np.array(
                [
                    ds._pergraph_target_raw_by_source(ds._raw[int(i)], spec["source"])
                    for i in ids
                ]
            )
            for spec in graph_specs
        ]
    )
    assert raw_graph.shape == raw_graph_truth.shape == (len(ids), 14)
    for j in range(len(ids)):
        assert np.array_equal(true_peak[offsets[j] : offsets[j + 1]].max(0), truth[j])
    assert all(
        (
            np.isfinite(x).all()
            for x in (raw_node, true_peak, raw_graph, raw_graph_truth)
        )
    )
    np.testing.assert_allclose(
        raw_graph[:, :4], endpoint["legacy"], rtol=1e-05, atol=1e-08
    )
    size = SIZES[buckets]
    dest.mkdir(parents=True, exist_ok=True)
    npz = dest / f"{subset}_joint_predictions.npz"
    np.savez_compressed(
        npz,
        cases=np.array(CASES),
        repeat=part["repeat"],
        fold=part["fold"],
        subset=subset,
        arm=model.arm,
        dataset_idx=ids,
        size_um=size,
        size_bucket=buckets,
        grain_count=counts,
        offsets=offsets,
        y_true=truth,
        raw_branch_prediction=endpoint["prediction"],
        direct_prediction=endpoint["legacy"],
        pred_peak=raw_node[:, :4],
        true_peak=true_peak,
        grain_dataset_idx=np.repeat(ids, counts),
        grain_id=np.concatenate([np.arange(n, dtype=np.int32) for n in counts]),
        grain_size_um=np.repeat(size, counts),
        node_transformed=pn,
        graph_transformed=pg,
        node_target_transformed=nt,
        graph_target_transformed=gt,
        node_mean_raw=raw_node,
        node_target_raw=raw_node_truth,
        node_aux_source_nan_mask=nan_mask,
        graph_mean_raw=raw_graph,
        graph_target_raw=raw_graph_truth,
        node_mean_names=np.array(
            ["lcf_fs_peak"] + [s["name"] for s in ds.per_grain_aux_targets]
        ),
        graph_mean_names=np.array(["lcf_fs_max"] + [s["name"] for s in graph_specs]),
    )
    rows = [
        dict(
            arm=model.arm,
            route=route,
            repeat=part["repeat"],
            fold=part["fold"],
            subset=subset,
            dataset_idx=int(idx),
            size_um=int(size[j]),
            case=case,
            truth=float(truth[j, c]),
            prediction=float(pred[j, c]),
        )
        for route, pred in (
            ("raw_branch", endpoint["prediction"]),
            ("legacy_direct", endpoint["legacy"]),
        )
        for j, idx in enumerate(ids)
        for c, case in enumerate(CASES)
    ]
    pd.DataFrame(rows).to_csv(dest / f"{subset}_maximum_predictions.csv", index=False)
    with np.load(npz) as z:
        assert np.array_equal(z["dataset_idx"], ids)
        assert np.array_equal(z["raw_branch_prediction"], endpoint["prediction"])
        assert np.array_equal(z["true_peak"], true_peak)
        assert np.array_equal(z["offsets"], offsets)
    report = dict(
        status="complete",
        subset=subset,
        n_sves=len(ids),
        n_grains=int(counts.sum()),
        raw_branch_score=endpoint["score"],
        raw_branch_cells=endpoint["cells"].tolist(),
        negative_raw_branch_counts=endpoint["negative_counts"],
        inverse_clip_counts=endpoint["inverse_clip_counts"],
        source_aux_missing_entries=int(nan_mask.sum()),
        archive_sha256=sha(npz),
        grain_truth_exact=True,
        endpoint_identity_exact=True,
        auxiliary_truth_note="Exact original raw source columns; source auxiliary NaNs restored using saved source mask. Transformed labels are also retained.",
    )
    dump(dest / f"{subset}_export.json", report)
    return report


def validate_release(path):
    release = json.loads(Path(path).read_text())
    assert (
        release["source_manifest"] == source_manifest()
    ), "Code/protocol changed after release"
    assert release["partitions_sha256"] == sha(FROZEN / "partitions.json")
    assert release["arms"] == list(ARMS)
    expected_keys = {f"repeat{p['repeat']}_fold{p['fold']}" for p in load_partitions()}
    assert set(release["preparations"]) == expected_keys
    for p in load_partitions():
        key = f"repeat{p['repeat']}_fold{p['fold']}"
        assert (
            sha(preparation_dir(p) / "preparation.json") == release["preparations"][key]
        )
    return release


def fit(inp, arm, device, release):
    part = inp["part"]
    dest = HERE / arm / f"repeat{part['repeat']}_fold{part['fold']}"
    if (dest / "completion.json").exists() or (dest / "failure.json").exists():
        raise RuntimeError(f"Preserve existing completed/failed run: {dest}")
    if dest.exists():
        raise RuntimeError(f"Partial run preserved; explicit recovery needed: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=False)
    start = time.monotonic()
    try:
        model = construct(inp, arm, device)
        initial = preparation_dir(part) / "initial_state.pt"
        prepared = json.loads(initial.with_name("preparation.json").read_text())
        assert prepared["source_manifest"] == source_manifest()
        assert prepared["preprocessing_sha256"] == sha(
            inp["pair"] / "fitted_preprocessing.pkl"
        )
        assert prepared["source_preparation_sha256"] == sha(
            inp["pair"] / "preparation.json"
        )
        assert prepared["saved_batch_orders_sha256"] == array_sha(inp["permutations"])
        assert prepared["raw_source_sha256"] == inp["raw_source_sha256"]
        assert prepared["initial_file_sha256"] == sha(initial)
        assert prepared["statistics_file_sha256"] == sha(
            initial.with_name("fitting_statistics.npz")
        )
        assert joint.state_hash(model) == prepared["paired_full_state_sha256"]
        optimizer, scheduler = joint.train.build_optimizer_and_scheduler(
            model, inp["cfg"]
        )
        assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
            id(p) for p in model.parameters()
        }
        assert joint.state_hash(model.core) == inp["prep"]["full_initial_state_sha256"]
        dump(
            dest / "protocol.json",
            dict(
                arm=arm,
                partition=part,
                release=release,
                initial_state_sha256=joint.state_hash(model),
                core_state_sha256=joint.state_hash(model.core),
                preprocessing_sha256=sha(inp["pair"] / "fitted_preprocessing.pkl"),
                optimizer_parameters=sum(
                    (p.numel() for g in optimizer.param_groups for p in g["params"])
                ),
                original_source_config=inp["cfg"],
                actual_epoch_eligibility="all epochs starting zero",
                raw_branch_zero_initialization="fitting size/case mean; original outputs unchanged",
                no_outer_transform_or_forward_before_checkpoint_selection=True,
            ),
        )
        history, best, best_epoch = ([], float("inf"), -1)
        stop_graphs, stop_ids, stop_truth, stop_buckets = subset_data(inp, "stopping")
        for epoch in range(MAX_EPOCHS):
            model.train()
            total, count = (np.zeros(5), 0)
            for batch in DataLoader(
                inp["values"][: inp["n"]],
                batch_size=16,
                sampler=inp["permutations"][epoch].tolist(),
                num_workers=0,
                pin_memory=device.type == "cuda",
            ):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = model.losses(model(batch), batch)
                losses[0].backward()
                if not all(
                    (
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                ):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
                total += np.array([v.item() for v in losses]) * batch.num_graphs
                count += batch.num_graphs
            stopping = evaluate(
                model, inp, stop_graphs, stop_ids, stop_truth, stop_buckets
            )
            score = stopping["score"]
            scheduler.step(score)
            if score < best:
                best, best_epoch = (score, epoch)
                torch.save(
                    dict(
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        epoch=epoch,
                        raw_stopping_score=score,
                        arm=arm,
                        full_core_initial_state_sha256=inp["prep"][
                            "full_initial_state_sha256"
                        ],
                        source_manifest=source_manifest(),
                    ),
                    dest / "best.pt",
                )
            row = dict(
                epoch_zero_based=epoch,
                stopping_raw_branch_nmse=score,
                stopping_cells=stopping["cells"].tolist(),
                best_epoch_zero_based=best_epoch,
                fitting_total_loss=float(total[0] / count),
                fitting_joint_loss=float(total[1] / count),
                fitting_node_loss=float(total[2] / count),
                fitting_graph_loss=float(total[3] / count),
                fitting_raw_maximum_loss=float(total[4] / count),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                negative_raw_branch_counts=stopping["negative_counts"],
                inverse_clip_counts=stopping["inverse_clip_counts"],
                elapsed_seconds=time.monotonic() - start,
            )
            history.append(row)
            dump(dest / "history.json", history)
            print(
                json.dumps(
                    dict(arm=arm, repeat=part["repeat"], fold=part["fold"], **row)
                ),
                flush=True,
            )
            if epoch - best_epoch >= PATIENCE:
                break
        torch.save(
            dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch,
            ),
            dest / "last.pt",
        )
        state = torch.load(dest / "best.pt", weights_only=True, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        frozen_hash = sha(dest / "best.pt")
        dump(
            dest / "selected_checkpoint.json",
            dict(
                checkpoint_sha256=frozen_hash,
                epoch_zero_based=best_epoch,
                selected_before_outer_target_extraction=True,
                selected_before_outer_transform_forward=True,
            ),
        )
        exports = [
            export_subset(model, inp, subset, dest)
            for subset in ("fitting", "stopping", "evaluation")
        ]
        assert sha(dest / "best.pt") == frozen_hash
        report = dict(
            status="complete",
            arm=arm,
            repeat=part["repeat"],
            fold=part["fold"],
            actual_epochs=len(history),
            selected_epoch_zero_based=best_epoch,
            stop_reason="budget cap120" if len(history) == 120 else "patience50",
            checkpoint_sha256=frozen_hash,
            selected_stopping_score=best,
            elapsed_seconds=time.monotonic() - start,
            exports=exports,
            all_outer_sves_preserved=True,
            no_implicit_raw_head_fallback=True,
        )
        dump(dest / "completion.json", report)
        return report
    except Exception as e:
        dump(
            dest / "failure.json",
            dict(
                status="failed",
                arm=arm,
                partition=part,
                elapsed_seconds=time.monotonic() - start,
                error=repr(e),
                traceback=traceback.format_exc(),
                no_failed_sves_dropped=True,
                no_automatic_mean_substitution=True,
            ),
        )
        raise


def init_distributed():
    torch.set_num_threads(2)
    if not torch.distributed.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.update(
            MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="1", RANK="0"
        )
        torch.distributed.init_process_group("gloo", init_method="env://")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="0,0", help="repeat,fold or all")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--arm", choices=("both",) + ARMS, default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if not (args.prepare or args.smoke or args.fit):
        parser.error("Choose --prepare/--smoke or --fit")
    release = validate_release(args.release) if args.fit and args.release else None
    if args.fit and release is None:
        parser.error(
            "Scientific fits require --release pointing to the generated training configuration"
        )
    init_distributed()
    try:
        reports = []
        for part in load_partitions():
            if args.pair != "all" and args.pair != f"{part['repeat']},{part['fold']}":
                continue
            inp = fold_inputs(part)
            if args.prepare:
                report = prepare(inp)
                reports.append(report)
                print("PREPARED", part["repeat"], part["fold"], flush=True)
            if args.smoke:
                from smoke import run_smoke

                reports.append(run_smoke(inp, torch.device(args.device)))
            if args.fit:
                for arm in ARMS if args.arm == "both" else (args.arm,):
                    reports.append(fit(inp, arm, torch.device(args.device), release))
                    gc.collect()
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
        if args.prepare:
            prepared_paths = {
                f"repeat{p['repeat']}_fold{p['fold']}": preparation_dir(p)
                / "preparation.json"
                for p in load_partitions()
            }
            if all((p.exists() for p in prepared_paths.values())):
                template = dict(
                    source_manifest=source_manifest(),
                    partitions_sha256=sha(FROZEN / "partitions.json"),
                    arms=list(ARMS),
                    target_fits=20,
                    preparations={k: sha(p) for k, p in prepared_paths.items()},
                )
                template_path = HERE / f"training_config_{manifest_id()}.json"
                if template_path.exists():
                    assert json.loads(template_path.read_text()) == template
                else:
                    dump(template_path, template)
        print(
            "REQUEST_COMPLETE",
            json.dumps(dict(reports=len(reports), scientific_fit_requested=args.fit)),
            flush=True,
        )
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
