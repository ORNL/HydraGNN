"""Fixed input-only mechanical baselines on the established repeated940 cohort."""
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
from datetime import datetime, timezone
import hashlib, json, pickle, platform, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
R2 = HERE.parents[1]
BASE = R2.parent / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
OLD = BASE / "regressors"
SOURCE = R2 / "studies_2026_11_16/hierarchy_output_probe"
MECH = R2 / "studies_2026_11_16/model_redevelopment/mechanical_connectivity"
SIZES = np.array([125, 177, 250, 354, 500])
FIELDS = [
    "yield_stress",
    "hardening_modulus",
    "Wp_lcf",
    "Wp_hcf",
    "sigma_amp_lcf",
    "sigma_amp_hcf",
]
REFERENCES = ["original", "no_message_degree", "no_message_no_degree"]
BUDGETS = ["fitting", "outer_training"]
MODELS = REFERENCES + [m + "_" + b for b in BUDGETS for m in ["RF", "HGB", "SizeMean"]]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ah = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def dump(name, value):
    (HERE / name).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def framewrite(name, frame):
    frame.to_csv(HERE / name, index=False, float_format="%.17g")


def load_inputs():
    config = json.loads((OLD / "config.json").read_text())
    assert config["raw_feature_artifact_sha256"] == sha(OLD / "raw_features940.npz")
    assert config["partitions_sha256"] == sha(BASE / "partitions.json")
    protocol = json.loads(
        (R2 / "studies_2026_11_15/label_matched_trees/protocol.json").read_text()
    )
    assert config["RF"] == protocol["RF"] and config["HGB"] == protocol["HGB"]
    with np.load(OLD / "raw_features940.npz") as z:
        raw = {k: z[k].copy() for k in z.files}
    with np.load(SOURCE / "true_auxiliary_cache.npz") as z:
        truth = z["mechanical_true"].copy()
        fields = z["mechanical_names"].astype(str)
        buckets = z["size_bucket"].copy()
        counts = z["grain_count"].copy()
    assert (
        list(fields) == FIELDS and truth.shape == (940, 6) and np.isfinite(truth).all()
    )
    assert np.array_equal(raw["dataset_idx"], np.arange(940))
    assert np.array_equal(raw["size_bucket"], buckets) and np.array_equal(
        raw["node_counts"], counts
    )
    assert (
        raw["raw_extreme249"].shape == (940, 249)
        and np.isfinite(raw["raw_extreme249"]).all()
    )
    parts = json.loads((BASE / "partitions.json").read_text())
    assert len(parts) == 10
    sources = [
        Path(__file__),
        HERE / "PRESPEC.txt",
        OLD / "config.json",
        OLD / "run_regressors.py",
        OLD / "preparation_report.json",
        OLD / "raw_features940.npz",
        BASE / "partitions.json",
        SOURCE / "true_auxiliary_cache.npz",
        MECH / "metrics_by_repeat.csv",
        MECH / "mechanical_arrays.npz",
        R2 / "studies_2026_11_15/label_matched_trees/protocol.json",
    ]
    for p in parts:
        for encoder in REFERENCES:
            sources.append(
                SOURCE
                / encoder
                / f'repeat{p["repeat"]}_fold{p["fold"]}'
                / "evaluation.npz"
            )
    dump(
        "input_manifest.json",
        dict(
            recorded_utc=datetime.now(timezone.utc).isoformat(),
            before_new_fitting=True,
            hashes={str(p.relative_to(R2.parent)): sha(p) for p in sources},
            RF=config["RF"],
            HGB=config["HGB"],
            fields=FIELDS,
            models=MODELS,
            budgets=BUDGETS,
        ),
    )
    grain_schema = json.loads((OLD / "preparation_report.json").read_text())[
        "selected32_schema"
    ]
    names = [
        f'{stat}__{g["name"]}' for stat in ["mean", "sd0", "max"] for g in grain_schema
    ]
    names += ["size_um", "log_size", "grain_count", "log_grain_count"] + [
        f"size_indicator_{s}" for s in SIZES
    ]
    names += [
        f'{stat}__{g["name"]}'
        for stat in ["q90", "q99", "top3_mean"]
        for g in grain_schema
    ]
    interactions = [
        "max_Schmid",
        "relative_volume",
        "max_Schmid_x_relative_volume",
        "max_Schmid_x_relative_volume_cuberoot",
        "max_Schmid_x_relative_degree",
        "max_Schmid_x_relative_surface",
    ]
    names += [
        f"{stat}__{v}"
        for stat in ["mean", "sd0", "max", "q50", "q90", "q95", "q99", "top3_mean"]
        for v in interactions
    ]
    assert len(names) == 249
    dump(
        "feature_schema.json",
        dict(
            features=names,
            grain32=grain_schema,
            source_artifact_sha256=sha(OLD / "raw_features940.npz"),
            scope="Input-only statistical summaries, not full ODF or spatial texture representation. Degree is graph-derived.",
        ),
    )
    return config, raw, truth, buckets, parts


def scores(y, p):
    e = p - y
    sy = y.std(ddof=1)
    sp = p.std(ddof=1)
    return dict(
        r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mae=float(np.mean(abs(e))),
        bias=float(e.mean()),
        truth_sd=float(sy),
        prediction_sd=float(sp),
        prediction_to_truth_sd=float(sp / sy),
        pearson=float(np.corrcoef(y, p)[0, 1]) if sp > 0 else 0.0,
    )


def main():
    assert not (
        HERE / "completion.json"
    ).exists(), "Preserve completed results; no silent refit."
    started = time.perf_counter()
    config, raw, truth, buckets, parts = load_inputs()
    X = raw["raw_extreme249"]
    sums = raw["grain_sum32"]
    counts = raw["node_counts"]
    sv = raw["size9"]
    sizes = SIZES[buckets]
    pred = np.full((len(MODELS), 2, 940, 6), np.nan)
    seen = np.zeros((len(MODELS), 2, 940), int)
    folds = np.full((2, 940), -1, int)
    fit_records = []
    stats = []
    (HERE / "models").mkdir(exist_ok=True)
    for part in parts:
        r, f, seed = part["repeat"], part["fold"], part["initialization_seed"]
        fitting = np.array(part["fitting"], int)
        stopping = np.array(part["stopping"], int)
        ev = np.array(part["evaluation"], int)
        assert not set(fitting) & set(stopping) and not set(ev) & set(
            np.r_[fitting, stopping]
        )
        assert sorted(np.r_[fitting, stopping, ev]) == list(range(940))
        folds[r, ev] = f
        for mi, encoder in enumerate(REFERENCES):
            path = SOURCE / encoder / f"repeat{r}_fold{f}" / "evaluation.npz"
            with np.load(path) as z:
                assert np.array_equal(z["dataset_idx"], ev) and np.array_equal(
                    z["size_um"], sizes[ev]
                )
                assert np.array_equal(z["mechanical_true"], truth[ev])
                assert (
                    z["mechanical_true_valid"].all()
                    and z["mechanical_pred_valid"].all()
                )
                pred[mi, r, ev] = z["mechanical_pred"]
                seen[mi, r, ev] += 1
        for budget in BUDGETS:
            tr = (
                fitting.copy()
                if budget == "fitting"
                else np.sort(np.r_[fitting, stopping])
            )
            centers = np.stack(
                [
                    sums[tr[buckets[tr] == b]].sum(0)
                    / counts[tr[buckets[tr] == b]].sum()
                    for b in range(5)
                ]
            )
            szcenters = np.stack([sv[tr[buckets[tr] == b]].mean(0) for b in range(5)])
            xx = X.copy()
            for b in range(5):
                mask = buckets == b
                for first in [0, 64, 105, 137, 169]:
                    xx[mask, first : first + 32] -= centers[b]
                xx[mask, 96:105] -= szcenters[b]
            means = np.stack([truth[tr[buckets[tr] == b]].mean(0) for b in range(5)])
            variances = np.stack(
                [truth[tr[buckets[tr] == b]].var(0, ddof=1) for b in range(5)]
            )
            pooled = truth[tr].var(0, ddof=1)
            denom = np.maximum(np.maximum(variances, 1e-6 * pooled), 1e-30)
            sd = np.sqrt(denom)
            target = (truth[tr] - means[buckets[tr]]) / sd[buckets[tr]]
            stats.append(
                dict(
                    repeat=r,
                    fold=f,
                    budget=budget,
                    seed=seed,
                    training_ids=tr.tolist(),
                    original_fitting_ids=fitting.tolist(),
                    original_stopping_ids=stopping.tolist(),
                    evaluation_ids=ev.tolist(),
                    feature_centers=centers.tolist(),
                    size9_centers=szcenters.tolist(),
                    target_means=means.tolist(),
                    target_variances=variances.tolist(),
                    target_scale_squared=denom.tolist(),
                    floor_active=bool(np.any(denom != variances)),
                )
            )
            for family in ["RF", "HGB", "SizeMean"]:
                model_name = family + "_" + budget
                mi = MODELS.index(model_name)
                tic = time.perf_counter()
                if family == "SizeMean":
                    prediction = means[buckets[ev]]
                    model = None
                elif family == "RF":
                    model = RandomForestRegressor(**config["RF"], random_state=seed)
                    with threadpool_limits(1):
                        model.fit(xx[tr], target)
                        p = model.predict(xx[ev])
                    prediction = means[buckets[ev]] + sd[buckets[ev]] * p
                else:
                    model = [
                        HistGradientBoostingRegressor(
                            **config["HGB"], random_state=seed
                        )
                        for _ in FIELDS
                    ]
                    with threadpool_limits(1):
                        for ci, m in enumerate(model):
                            m.fit(xx[tr], target[:, ci])
                        p = np.column_stack([m.predict(xx[ev]) for m in model])
                    prediction = means[buckets[ev]] + sd[buckets[ev]] * p
                assert (
                    prediction.shape == (len(ev), 6) and np.isfinite(prediction).all()
                )
                pred[mi, r, ev] = prediction
                seen[mi, r, ev] += 1
                model_file = None
                model_sha = None
                if model is not None:
                    path = HERE / "models" / f"{model_name}_r{r}_f{f}.pkl"
                    with path.open("wb") as stream:
                        pickle.dump(model, stream)
                    model_file = str(path.relative_to(HERE))
                    model_sha = sha(path)
                fit_records.append(
                    dict(
                        model=model_name,
                        budget=budget,
                        repeat=r,
                        fold=f,
                        seed=seed,
                        n_train=len(tr),
                        n_eval=len(ev),
                        estimator_fits=0
                        if model is None
                        else 6
                        if isinstance(model, list)
                        else 1,
                        seconds=time.perf_counter() - tic,
                        model_file=model_file,
                        model_sha256=model_sha,
                    )
                )
            print(
                f"completed repeat={r} fold={f} budget={budget} n_train={len(tr)}",
                flush=True,
            )
            dump("preprocessing.json", stats)
            framewrite("fit_records.csv", pd.DataFrame(fit_records))
    assert (seen == 1).all() and np.isfinite(pred).all() and (folds >= 0).all()
    np.savez_compressed(
        HERE / "prediction_arrays.npz",
        prediction=pred,
        truth=truth,
        size_um=sizes,
        models=np.array(MODELS),
        fields=np.array(FIELDS),
        dataset_idx=np.arange(940),
        fold=folds,
    )
    rows = []
    metrics = []
    byfold = []
    for mi, model in enumerate(MODELS):
        for r in range(2):
            for ci, field in enumerate(FIELDS):
                for idx in range(940):
                    rows.append(
                        dict(
                            model=model,
                            repeat=r,
                            fold=int(folds[r, idx]),
                            dataset_idx=idx,
                            size_um=int(sizes[idx]),
                            field=field,
                            truth=float(truth[idx, ci]),
                            prediction=float(pred[mi, r, idx, ci]),
                            units="MPa",
                        )
                    )
                for size in ["pooled", *map(str, SIZES)]:
                    ix = (
                        np.arange(940)
                        if size == "pooled"
                        else np.flatnonzero(sizes == int(size))
                    )
                    metrics.append(
                        dict(
                            model=model,
                            repeat=r,
                            field=field,
                            size_um=size,
                            n_sve=len(ix),
                            **scores(truth[ix, ci], pred[mi, r, ix, ci]),
                        )
                    )
                    for f in range(5):
                        sub = ix[folds[r, ix] == f]
                        byfold.append(
                            dict(
                                model=model,
                                repeat=r,
                                fold=f,
                                field=field,
                                size_um=size,
                                n_sve=len(sub),
                                **scores(truth[sub, ci], pred[mi, r, sub, ci]),
                            )
                        )
    df = pd.DataFrame(rows)
    assert len(df) == 101520
    framewrite("predictions_long.csv", df)
    by = pd.DataFrame(metrics)
    framewrite("metrics_by_repeat.csv", by)
    framewrite("metrics_by_fold.csv", pd.DataFrame(byfold))
    cols = [
        "r2",
        "rmse",
        "mae",
        "bias",
        "truth_sd",
        "prediction_sd",
        "prediction_to_truth_sd",
        "pearson",
    ]
    summary = (
        by.groupby(["model", "field", "size_um"], sort=False)[cols]
        .mean()
        .add_suffix("_mean")
        .reset_index()
    )
    framewrite("metrics_summary.csv", summary)
    prior = pd.read_csv(
        MECH / "metrics_by_repeat.csv",
        dtype={"size_um": str},
        float_precision="round_trip",
    ).rename(columns={"encoder": "model"})
    compared = by[by.model.isin(REFERENCES)].merge(
        prior,
        on=["model", "repeat", "field", "size_um"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    differences = {
        c: float(np.max(abs(compared[c + "_new"] - compared[c + "_old"])))
        for c in ["r2", "rmse", "mae", "bias", "pearson"]
    }
    assert len(compared) == 216 and max(differences.values()) < 1e-11, differences
    losses = np.mean((pred - truth[None, None, :, :]) ** 2, axis=1)
    paired = []
    weight_hashes = {}
    for size in ["pooled", *map(str, SIZES)]:
        ix = np.arange(940) if size == "pooled" else np.flatnonzero(sizes == int(size))
        rng = np.random.default_rng(20261117)
        w = np.zeros((4000, len(ix)), np.int32)
        for s in np.unique(sizes[ix]):
            colsix = np.flatnonzero(sizes[ix] == s)
            n = len(colsix)
            draws = rng.integers(0, n, size=(4000, n))
            w[:, colsix] = np.array([np.bincount(d, minlength=n) for d in draws])
        weight_hashes[size] = ah(w)
        cells = losses[:, ix, :].transpose(1, 0, 2).reshape(len(ix), len(MODELS) * 6)
        mse = cells.mean(0).reshape(len(MODELS), 6)
        bmse = (w @ cells / len(ix)).reshape(4000, len(MODELS), 6)
        y = truth[ix]
        vy = y.var(0, ddof=0)
        bvy = w @ (y * y) / len(ix) - (w @ y / len(ix)) ** 2
        for mi, ref in enumerate(MODELS[1:], 1):
            dr2 = (mse[mi] - mse[0]) / vy
            br2 = (bmse[:, mi] - bmse[:, 0]) / bvy
            drmse = np.sqrt(mse[0]) - np.sqrt(mse[mi])
            brmse = np.sqrt(bmse[:, 0]) - np.sqrt(bmse[:, mi])
            pct = 100 * (1 - np.sqrt(mse[0] / mse[mi]))
            bpct = 100 * (1 - np.sqrt(bmse[:, 0] / bmse[:, mi]))
            lo_r2, hi_r2 = np.quantile(br2, [0.025, 0.975], axis=0)
            lo_rmse, hi_rmse = np.quantile(brmse, [0.025, 0.975], axis=0)
            lo_pct, hi_pct = np.quantile(bpct, [0.025, 0.975], axis=0)
            for ci, field in enumerate(FIELDS):
                paired.append(
                    dict(
                        model="original",
                        reference=ref,
                        field=field,
                        size_um=size,
                        n_sve=len(ix),
                        graph_r2=float(1 - mse[0, ci] / vy[ci]),
                        reference_r2=float(1 - mse[mi, ci] / vy[ci]),
                        delta_r2=float(dr2[ci]),
                        delta_r2_ci_low=float(lo_r2[ci]),
                        delta_r2_ci_high=float(hi_r2[ci]),
                        graph_rmse=float(np.sqrt(mse[0, ci])),
                        reference_rmse=float(np.sqrt(mse[mi, ci])),
                        delta_rmse=float(drmse[ci]),
                        delta_rmse_ci_low=float(lo_rmse[ci]),
                        delta_rmse_ci_high=float(hi_rmse[ci]),
                        graph_rmse_reduction_percent=float(pct[ci]),
                        reduction_ci_low=float(lo_pct[ci]),
                        reduction_ci_high=float(hi_pct[ci]),
                        units="MPa",
                    )
                )
    framewrite("paired_graph_comparisons.csv", pd.DataFrame(paired))
    dump(
        "completion.json",
        dict(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            physical_sves=940,
            repeats=2,
            prediction_rows=len(df),
            models=MODELS,
            estimator_fits=int(sum(x["estimator_fits"] for x in fit_records)),
            fit_records=len(fit_records),
            all_predictions_finite=True,
            prior_metric_max_abs_difference=differences,
            bootstrap_draws=4000,
            bootstrap_seed=20261117,
            bootstrap_weight_sha256=weight_hashes,
            script_sha256=sha(Path(__file__)),
            prespec_sha256=sha(HERE / "PRESPEC.txt"),
            total_seconds=time.perf_counter() - started,
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
            files={p.name: sha(p) for p in HERE.glob("*.csv")},
        ),
    )
    print(
        summary[summary.size_um == "pooled"]
        .pivot(index="field", columns="model", values="r2_mean")
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
