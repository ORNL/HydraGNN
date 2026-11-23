"""Matched grain-to-maximum and direct-SVE tree comparisons on frozen SVE splits."""
from fip_release_paths import release_root
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import pickle
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "revision_R2/studies_2026_11_15"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def prepare():
    out = HERE / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "validation.json").exists():
        assert json.loads((out / "validation.json").read_text())["status"] == "pass"
        print("INPUTS ALREADY VERIFIED", flush=True)
        return
    replay = imported("original_replay", OLD / "inference/replay_saved.py")
    import torch

    torch.set_num_threads(1)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(34500 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    p = BASE / "gnn/repeat0_fold0/fitted_preprocessing.pkl"
    with p.open("rb") as f:
        ds = pickle.load(f)
    ds._raw = None
    ds._load()
    assert len(ds._raw) == 940
    with np.load(BASE / "regressors/raw_features940.npz") as z:
        reference = {k: z[k] for k in z.files}
    xs = []
    ys = []
    gids = []
    buckets = []
    counts = []
    hashes = []
    for idx, d in enumerate(ds._raw):
        x = ds._select_x(d.x).numpy().astype(np.float64)
        y = np.column_stack(
            [getattr(d, c + "_fip_dist")[:, 0].numpy() for c in CASES]
        ).astype(np.float64)
        n = len(x)
        b = int(d.size_bucket[0])
        hx = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        assert hx == reference["selected32_sha256"][idx], idx
        assert np.array_equal(y.max(0), reference["raw_truth"][idx]), idx
        assert n == reference["node_counts"][idx] and b == reference["size_bucket"][idx]
        assert (
            x.shape == (n, 32)
            and y.shape == (n, 4)
            and np.isfinite(x).all()
            and np.isfinite(y).all()
        )
        xs.append(x.astype(np.float32))
        ys.append(y)
        gids.append(np.full(n, idx, np.int32))
        buckets.append(np.full(n, b, np.int8))
        counts.append(n)
        hashes.append(hx)
    arrays = dict(
        raw32=np.concatenate(xs),
        grain_truth=np.concatenate(ys),
        grain_dataset_idx=np.concatenate(gids),
        grain_size_bucket=np.concatenate(buckets),
        grain_count=np.array(counts, np.int32),
        offsets=np.r_[0, np.cumsum(counts)],
        size_bucket=reference["size_bucket"],
        sve_truth=reference["raw_truth"],
        raw_extreme249=reference["raw_extreme249"],
        size9=reference["size9"],
        grain_sum32=reference["grain_sum32"],
    )
    for key, a in arrays.items():
        np.save(out / (key + ".npy"), a, allow_pickle=False)
    dump(
        out / "validation.json",
        dict(
            status="pass",
            n_sves=940,
            n_grains=len(arrays["raw32"]),
            source_grain_input_hashes_exact940=True,
            source_maximum_targets_exact940=True,
            no_preprocessing_from_outer_labels=True,
            source_dataset=str(
                release_root() / "examples/cpfem_fip_data/outputs/full_data_v5.pt"
            ),
            source_dataset_sha256=json.loads(
                (OLD / "inference/raw_dataset_provenance.json").read_text()
            )["sha256"],
            raw_features_reference_sha256=sha(BASE / "regressors/raw_features940.npz"),
            partitions_sha256=sha(BASE / "partitions.json"),
            artifacts={p.name: sha(p) for p in out.glob("*.npy")},
            created_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    torch.distributed.destroy_process_group()
    print("INPUT PREPARATION PASS", 940, len(arrays["raw32"]), flush=True)


def input_arrays():
    assert json.loads((HERE / "inputs/validation.json").read_text())["status"] == "pass"
    return {p.stem: np.load(p, mmap_mode="r") for p in (HERE / "inputs").glob("*.npy")}


def model_for(family, settings, seed, jobs):
    cls = RandomForestRegressor if family.startswith("rf") else ExtraTreesRegressor
    return cls(**settings, random_state=seed, n_jobs=jobs)


def center_direct(X, buckets, centers, size_centers):
    x = X.copy()
    for b in range(5):
        mask = buckets == b
        for first in [0, 64, 105, 137, 169]:
            x[mask, first : first + 32] -= centers[b]
        x[mask, 96:105] -= size_centers[b]
    return x


def export_grains(a, ids, pred, direct, repeat, fold, family, subset):
    counts = a["grain_count"][ids]
    offsets = np.r_[0, np.cumsum(counts)]
    gidx = np.concatenate(
        [np.arange(a["offsets"][i], a["offsets"][i + 1]) for i in ids]
    )
    truth = a["grain_truth"][gidx]
    b = a["size_bucket"][ids]
    size = SIZES[b]
    assert len(pred) == len(truth) == offsets[-1] and np.isfinite(pred).all()
    for j, idx in enumerate(ids):
        assert np.array_equal(
            truth[offsets[j] : offsets[j + 1]].max(0), a["sve_truth"][idx]
        )
    out = HERE / family / "data"
    out.mkdir(parents=True, exist_ok=True)
    name = f"grain_predictions_{('' if subset == 'evaluation' else 'stopping_')}repeat{repeat}_fold{fold}"
    path = out / (name + ".npz")
    np.savez_compressed(
        path,
        cases=np.array(CASES),
        repeat=np.array(repeat),
        fold=np.array(fold),
        subset=np.array(subset),
        grain_dataset_idx=np.repeat(ids, counts),
        grain_id=np.concatenate([np.arange(n, dtype=np.int32) for n in counts]),
        grain_size_um=np.repeat(size, counts),
        pred_peak=pred,
        true_peak=truth,
        dataset_idx=ids,
        size_um=size,
        size_bucket=b,
        grain_count=counts,
        y_true=a["sve_truth"][ids],
        direct_prediction=direct,
        offsets=offsets,
    )
    dump(
        path.with_suffix(".json"),
        dict(
            status="pass",
            subset=subset,
            repeat=repeat,
            fold=fold,
            family=family,
            n_sves=len(ids),
            n_grains=len(pred),
            grain_truth_max_equals_sve_truth=True,
            all_predictions_finite=True,
            archive_sha256=sha(path),
            direct_prediction_definition="New matched direct tree fitted on original fitting SVEs only",
        ),
    )


def run_fold(family, repeat, fold, jobs):
    folder = HERE / family / f"repeat{repeat}_fold{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "completion.json").exists():
        assert (
            json.loads((folder / "completion.json").read_text())["status"] == "complete"
        )
        print("ALREADY COMPLETE", family, repeat, fold, flush=True)
        return
    protocol = json.loads((HERE / "protocol.json").read_text())
    a = input_arrays()
    part = json.loads((BASE / f"partition_r{repeat}_f{fold}.json").read_text())
    fit = np.array(part["fitting"])
    stop = np.array(part["stopping"])
    ev = np.array(part["evaluation"])
    union = np.sort(np.r_[fit, stop])
    assert (
        not set(fit) & set(stop)
        and (not set(union) & set(ev))
        and (set(np.r_[union, ev]) == set(range(940)))
    )
    b = a["size_bucket"]
    gb = a["grain_size_bucket"]
    gids = a["grain_dataset_idx"]
    seed = part["initialization_seed"]
    gfit = np.isin(gids, fit)
    infer = np.r_[stop, ev]
    ginf = np.concatenate(
        [np.arange(a["offsets"][i], a["offsets"][i + 1]) for i in infer]
    )
    means = np.stack([a["grain_truth"][gfit & (gb == k)].mean(0) for k in range(5)])
    sd = np.stack([a["grain_truth"][gfit & (gb == k)].std(0, ddof=1) for k in range(5)])
    assert (sd > 0).all()
    centers = np.stack(
        [
            a["grain_sum32"][fit[b[fit] == k]].sum(0)
            / a["grain_count"][fit[b[fit] == k]].sum()
            for k in range(5)
        ]
    )

    def grainX(mask):
        return np.column_stack(
            [a["raw32"][mask] - centers[gb[mask]], np.eye(5)[gb[mask]]]
        ).astype(np.float32)

    X = grainX(gfit)
    y = (a["grain_truth"][gfit] - means[gb[gfit]]) / sd[gb[gfit]]
    base_family = "rf" if family.startswith("rf") else family
    config = protocol["grain_models"][base_family]
    weights = None
    weight_record = dict(mode="uniform grain weights")
    if family in ["rf_sve", "rf_sve_tail"]:
        weights = 1 / a["grain_count"][gids[gfit]].astype(float)
        weight_record = dict(mode="inverse number of grains per SVE")
        if family == "rf_sve_tail":
            q90 = np.stack(
                [
                    np.quantile(a["grain_truth"][gfit & (gb == k)], 0.9, axis=0)
                    for k in range(5)
                ]
            )
            multiplier = 1 + 4 * (a["grain_truth"][gfit] > q90[gb[gfit]]).mean(axis=1)
            weights *= multiplier
            weight_record.update(
                tail_multiplier="1+4*mean of four per-case above-q90 indicators",
                grain_q90_by_size_case=q90.tolist(),
            )
        weights /= weights.mean()
        weight_record.update(
            weight_min=float(weights.min()),
            weight_max=float(weights.max()),
            weight_mean=float(weights.mean()),
        )
    t = time.perf_counter()
    node = model_for(family, config, seed, jobs)
    node.fit(X, y, sample_weight=weights)
    fit_seconds = time.perf_counter() - t
    pred = node.predict(grainX(ginf)) * sd[gb[ginf]] + means[gb[ginf]]
    assert np.isfinite(pred).all()
    joblib.dump(node, folder / "grain_model.joblib", compress=3)
    print(
        "GRAIN FIT",
        family,
        repeat,
        fold,
        "seconds",
        round(fit_seconds, 2),
        "grains",
        int(gfit.sum()),
        flush=True,
    )
    del X, y, node
    rows = []
    direct_outputs = {}
    direct_stats = []
    for budget, tr in [("fitting", fit), ("outer_training", union)]:
        mu = np.stack([a["sve_truth"][tr[b[tr] == k]].mean(0) for k in range(5)])
        var = np.stack(
            [a["sve_truth"][tr[b[tr] == k]].var(0, ddof=1) for k in range(5)]
        )
        floor = np.maximum(
            np.maximum(var, 1e-06 * a["sve_truth"][tr].var(0, ddof=1)), 1e-30
        )
        ss = np.sqrt(floor)
        xc = np.stack(
            [
                a["grain_sum32"][tr[b[tr] == k]].sum(0)
                / a["grain_count"][tr[b[tr] == k]].sum()
                for k in range(5)
            ]
        )
        sc = np.stack([a["size9"][tr[b[tr] == k]].mean(0) for k in range(5)])
        xx = center_direct(a["raw_extreme249"], b, xc, sc)
        m = model_for(family, protocol["direct_models"][base_family], seed, jobs)
        t = time.perf_counter()
        m.fit(xx[tr], (a["sve_truth"][tr] - mu[b[tr]]) / ss[b[tr]])
        d = m.predict(xx[infer]) * ss[b[infer]] + mu[b[infer]]
        assert np.isfinite(d).all()
        direct_outputs[budget] = d
        joblib.dump(m, folder / f"direct_{budget}.joblib", compress=3)
        direct_stats.append(
            dict(
                budget=budget,
                ids=tr.tolist(),
                mean=mu.tolist(),
                sd=ss.tolist(),
                feature_center=xc.tolist(),
                size_center=sc.tolist(),
                seconds=time.perf_counter() - t,
            )
        )
        for j, idx in enumerate(infer):
            for c, case in enumerate(CASES):
                rows.append(
                    dict(
                        model=f"{family} direct {budget}",
                        repeat=repeat,
                        fold=fold,
                        dataset_idx=int(idx),
                        size_um=int(SIZES[b[idx]]),
                        case=case,
                        truth=float(a["sve_truth"][idx, c]),
                        prediction=float(d[j, c]),
                        subset="stopping" if j < len(stop) else "evaluation",
                        target="SVE maximum",
                        units="MPa" if case.endswith("wp") else "dimensionless",
                    )
                )
        if family == "rf":
            matched = model_for(family, config, seed, jobs)
            matched.fit(xx[tr], (a["sve_truth"][tr] - mu[b[tr]]) / ss[b[tr]])
            dp = matched.predict(xx[infer]) * ss[b[infer]] + mu[b[infer]]
            joblib.dump(
                matched, folder / f"direct_grain_settings_{budget}.joblib", compress=3
            )
            for j, idx in enumerate(infer):
                for c, case in enumerate(CASES):
                    rows.append(
                        dict(
                            model=f"rf direct grain-settings {budget}",
                            repeat=repeat,
                            fold=fold,
                            dataset_idx=int(idx),
                            size_um=int(SIZES[b[idx]]),
                            case=case,
                            truth=float(a["sve_truth"][idx, c]),
                            prediction=float(dp[j, c]),
                            subset="stopping" if j < len(stop) else "evaluation",
                            target="SVE maximum",
                            units="MPa" if case.endswith("wp") else "dimensionless",
                        )
                    )
    stop_n = int(a["grain_count"][stop].sum())
    export_grains(
        a,
        stop,
        pred[:stop_n],
        direct_outputs["fitting"][: len(stop)],
        repeat,
        fold,
        family,
        "stopping",
    )
    export_grains(
        a,
        ev,
        pred[stop_n:],
        direct_outputs["fitting"][len(stop) :],
        repeat,
        fold,
        family,
        "evaluation",
    )
    pd.DataFrame(rows).to_csv(
        folder / "direct_predictions.csv", index=False, float_format="%.17g"
    )
    dump(
        folder / "fitted_preprocessing.json",
        dict(
            grain_mean_by_size=means.tolist(),
            grain_sd_by_size=sd.tolist(),
            grain_feature_center=centers.tolist(),
            grain_weights=weight_record,
            direct=direct_stats,
        ),
    )
    dump(
        folder / "completion.json",
        dict(
            status="complete",
            family=family,
            repeat=repeat,
            fold=fold,
            seed=seed,
            fitting_sve_ids=fit.tolist(),
            stopping_sve_ids=stop.tolist(),
            evaluation_sve_ids=ev.tolist(),
            n_fitting_grains=int(gfit.sum()),
            n_fitting_sves=len(fit),
            n_stopping_sves=len(stop),
            n_evaluation_sves=len(ev),
            grain_fit_seconds=fit_seconds,
            grain_model=sha(folder / "grain_model.joblib"),
            script_sha256=sha(__file__),
            protocol_sha256=sha(HERE / "protocol.json"),
            partition_sha256=sha(BASE / f"partition_r{repeat}_f{fold}.json"),
            full_grain_fitting_labels_used=True,
            grain_bootstrap_sampling=config.get("max_samples", "all"),
            label_budget_note="Grain fits use all grain labels in fitting SVEs as the sampling population. Direct fitting uses fitting maxima; direct outer_training uses fitting+stopping maxima. Stopping predictions of the outer_training direct model are in-sample and must never fit a second-stage calibrator.",
        ),
    )
    print("FOLD COMPLETE", family, repeat, fold, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "fit"])
    p.add_argument("--family", choices=["rf", "rf_sve", "rf_sve_tail", "extra_trees"])
    p.add_argument("--repeat", type=int)
    p.add_argument("--fold", type=int)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        run_fold(args.family, args.repeat, args.fold, args.jobs)
