"""Fixed tree comparators trained on the complete outer-training partitions."""
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
import hashlib
import json
import pickle
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
ORIGINAL = BASE / "regressors"
BRIDGE = HERE.parent / "bridge"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
KEYS = ["repeat", "fold", "dataset_idx", "size_um", "case"]
LABELS = ["Outer-training Random forest", "Outer-training Histogram boosting"]
CANDIDATES = [
    "Graph Affine bridge calibration",
    "evt_original_recut__size_mean_fallback",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def center_features(X, sv, buckets, centers, szcenters):
    # Exact column operations from the original audited runner.
    xx = X.copy()
    for b in range(5):
        mask = buckets == b
        for first in [0, 64, 105, 137, 169]:
            xx[mask, first : first + 32] -= centers[b]
        xx[mask, 96:105] -= szcenters[b]
    return xx


def metric(frame):
    y = frame.truth.to_numpy()
    p = frame.prediction.to_numpy()
    e = p - y
    return dict(
        n_predictions=len(y),
        n_sves=frame.dataset_idx.nunique(),
        r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(e * e))),
        mae=float(np.mean(abs(e))),
        bias=float(e.mean()),
        prediction_sd=float(p.std(ddof=1)),
        truth_sd=float(y.std(ddof=1)),
    )


def fit():
    assert not (
        HERE / "predictions_long.csv"
    ).exists(), "Preserve completed results; do not silently refit."
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = json.loads((ORIGINAL / "config.json").read_text())
    assert protocol["RF"] == config["RF"] and protocol["HGB"] == config["HGB"]
    assert config["partitions_sha256"] == sha(BASE / "partitions.json")
    assert config["raw_feature_artifact_sha256"] == sha(
        ORIGINAL / "raw_features940.npz"
    )
    inputs = [
        HERE / "protocol.json",
        Path(__file__),
        ORIGINAL / "config.json",
        ORIGINAL / "run_regressors.py",
        ORIGINAL / "raw_features940.npz",
        BASE / "partitions.json",
        BRIDGE / "predictions_long.csv",
    ]
    inputs += sorted(BASE.glob("partition_r*_f*.json"))
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    dump(
        "input_manifest.json",
        dict(recorded_utc=datetime.now(timezone.utc).isoformat(), hashes=input_hashes),
    )
    with np.load(ORIGINAL / "raw_features940.npz", allow_pickle=False) as z:
        X = z["raw_extreme249"]
        y = z["raw_truth"]
        sv = z["size9"]
        buckets = z["size_bucket"]
        sums = z["grain_sum32"]
        counts = z["node_counts"]
        assert np.array_equal(z["dataset_idx"], np.arange(940))
    assert (
        X.shape == (940, 249)
        and y.shape == (940, 4)
        and np.isfinite(X).all()
        and np.isfinite(y).all()
    )
    start = time.perf_counter()
    allrows = []
    stats = []
    records = []
    for path in sorted(BASE.glob("partition_r*_f*.json")):
        part = json.loads(path.read_text())
        r = part["repeat"]
        f = part["fold"]
        seed = part["initialization_seed"]
        oldfit = np.array(part["fitting"], int)
        stop = np.array(part["stopping"], int)
        ev = np.array(part["evaluation"], int)
        tr = np.sort(np.r_[oldfit, stop])
        assert not set(oldfit) & set(stop) and not set(tr) & set(ev)
        assert len(tr) + len(ev) == 940 and len(np.unique(tr)) == len(tr)
        assert set(np.r_[tr, ev]) == set(range(940))
        centers = np.stack(
            [
                sums[tr[buckets[tr] == b]].sum(0) / counts[tr[buckets[tr] == b]].sum()
                for b in range(5)
            ]
        )
        szcenters = np.stack([sv[tr[buckets[tr] == b]].mean(0) for b in range(5)])
        xx = center_features(X, sv, buckets, centers, szcenters)
        means = np.stack([y[tr[buckets[tr] == b]].mean(0) for b in range(5)])
        variances = np.stack([y[tr[buckets[tr] == b]].var(0, ddof=1) for b in range(5)])
        pooled = y[tr].var(0, ddof=1)
        denominators = np.maximum(np.maximum(variances, 1e-6 * pooled), 1e-30)
        sd = np.sqrt(denominators)
        targets = (y[tr] - means[buckets[tr]]) / sd[buckets[tr]]
        stats.append(
            dict(
                repeat=r,
                fold=f,
                initialization_seed=seed,
                original_fitting_ids=oldfit.tolist(),
                original_stopping_ids=stop.tolist(),
                combined_fitting_ids=tr.tolist(),
                outer_ids=ev.tolist(),
                grain32_center_by_size=centers.tolist(),
                size9_center_by_size=szcenters.tolist(),
                target_mean_by_size=means.tolist(),
                target_sample_variance_by_size=variances.tolist(),
                target_variance_floor_by_size=denominators.tolist(),
            )
        )
        models = [
            (LABELS[0], RandomForestRegressor(**config["RF"], random_state=seed)),
            (
                LABELS[1],
                [
                    HistGradientBoostingRegressor(**config["HGB"], random_state=seed)
                    for c in CASES
                ],
            ),
        ]
        for label, model in models:
            t = time.perf_counter()
            with threadpool_limits(1):
                if isinstance(model, list):
                    for j, m in enumerate(model):
                        m.fit(xx[tr], targets[:, j])
                    raw = np.column_stack([m.predict(xx[ev]) for m in model])
                else:
                    model.fit(xx[tr], targets)
                    raw = model.predict(xx[ev])
            prediction = means[buckets[ev]] + sd[buckets[ev]] * raw
            assert prediction.shape == (len(ev), 4) and np.isfinite(prediction).all()
            modelpath = (
                HERE / f'model_r{r}_f{f}_{"rf" if label==LABELS[0] else "hgb"}.pkl'
            )
            with modelpath.open("wb") as stream:
                pickle.dump(model, stream)
            seconds = time.perf_counter() - t
            records.append(
                dict(
                    model=label,
                    repeat=r,
                    fold=f,
                    seed=seed,
                    n_fitting=len(tr),
                    n_original_fitting=len(oldfit),
                    n_added_stopping=len(stop),
                    n_outer=len(ev),
                    optimized_estimators=4 if isinstance(model, list) else 1,
                    seconds=seconds,
                    model_sha256=sha(modelpath),
                )
            )
            for i, idx in enumerate(ev):
                for j, case in enumerate(CASES):
                    allrows.append(
                        dict(
                            model=label,
                            repeat=r,
                            fold=f,
                            dataset_idx=int(idx),
                            size_um=SIZES[int(buckets[idx])],
                            case=case,
                            truth=float(y[idx, j]),
                            prediction=float(prediction[i, j]),
                            target="SVE maximum",
                            units="MPa" if case.endswith("wp") else "dimensionless",
                        )
                    )
            print(
                f"FIT {label} repeat={r} fold={f} training={len(tr)} seconds={seconds:.2f}",
                flush=True,
            )
        pd.DataFrame(allrows).to_csv(
            HERE / "predictions_long.partial.csv", index=False, float_format="%.17g"
        )
        pd.DataFrame(records).to_csv(HERE / "fit_records.csv", index=False)
        dump("fitted_preprocessing.json", stats)
    p = pd.DataFrame(allrows)
    assert (
        len(p) == 15040
        and not p.duplicated(["model", "repeat", "dataset_idx", "case"]).any()
    )
    assert p.groupby(["model", "repeat", "case"]).dataset_idx.nunique().eq(940).all()
    p.to_csv(HERE / "predictions_long.csv", index=False, float_format="%.17g")
    dump(
        "fit_report.json",
        dict(
            status="complete",
            n_predictions=len(p),
            n_models=2,
            n_repeats=2,
            n_outer_folds=5,
            distinct_sves_per_case_repeat=940,
            optimized_estimator_fits=50,
            total_seconds=time.perf_counter() - start,
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
            all_finite=True,
            all_transforms_outer_training_only=True,
            no_outer_label_fitting=True,
            no_hyperparameter_tuning=True,
            no_neural_retraining=True,
            prediction_sha256=sha(HERE / "predictions_long.csv"),
        ),
    )
    return p


def analyze(trees):
    reference = pd.read_csv(BRIDGE / "predictions_long.csv", low_memory=False)
    include = CANDIDATES + [
        "Random forest",
        "Histogram boosting",
        "Outer-training size mean",
        "size_mean",
        "direct_graph_head",
    ]
    reference = reference[reference.model.isin(include)][
        KEYS + ["model", "truth", "prediction"]
    ]
    allpred = pd.concat(
        [trees[KEYS + ["model", "truth", "prediction"]], reference], ignore_index=True
    )
    for (model, repeat, case), g in allpred.groupby(["model", "repeat", "case"]):
        assert (
            len(g) == 940
            and g.dataset_idx.nunique() == 940
            and np.isfinite(g.prediction).all()
        ), (model, repeat, case)
    allpred.to_csv(
        HERE / "all_comparison_predictions.csv", index=False, float_format="%.17g"
    )
    rows = []
    for (model, case), g in allpred.groupby(["model", "case"]):
        for repeat in ["both", 0, 1]:
            sub = g if repeat == "both" else g[g.repeat.eq(repeat)]
            for size in ["pooled"] + SIZES:
                frame = sub if size == "pooled" else sub[sub.size_um.eq(size)]
                rows.append(
                    dict(
                        model=model,
                        case=case,
                        repeat=repeat,
                        size_um=size,
                        **metric(frame),
                    )
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(HERE / "metrics.csv", index=False, float_format="%.17g")
    per = metrics[metrics.repeat.ne("both")]
    summary = per.groupby(["model", "case", "size_um"])[
        ["r2", "rmse", "mae", "bias"]
    ].agg(["mean", "min", "max", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(
        HERE / "metrics_summary.csv", index=False, float_format="%.17g"
    )
    output = []
    for ref in LABELS:
        rng = np.random.default_rng(20261115)
        b = trees[trees.model.eq(ref)][KEYS + ["truth", "prediction"]]
        for candidate in sorted(CANDIDATES):
            a = reference[reference.model.eq(candidate)]
            paired = a.merge(
                b, on=KEYS, validate="one_to_one", suffixes=("", "_reference")
            )
            assert len(paired) == 7520 and np.allclose(
                paired.truth, paired.truth_reference, rtol=1e-12, atol=1e-14
            )
            for case, g in paired.groupby("case", sort=True):
                for size in ["pooled"] + SIZES:
                    q = g if size == "pooled" else g[g.size_um.eq(size)]
                    q = q.assign(
                        loss=(q.prediction - q.truth) ** 2,
                        base_loss=(q.prediction_reference - q.truth) ** 2,
                    )
                    agg = (
                        q.groupby(["dataset_idx", "size_um"])
                        .agg(
                            loss=("loss", "mean"),
                            base_loss=("base_loss", "mean"),
                            n_repeat=("repeat", "nunique"),
                        )
                        .reset_index()
                    )
                    assert agg.n_repeat.eq(2).all()
                    mb = np.zeros(2000)
                    bb = np.zeros(2000)
                    for _, stratum in agg.groupby("size_um"):
                        n = len(stratum)
                        draw = rng.integers(0, n, size=(2000, n))
                        mb += stratum.loss.to_numpy()[draw].sum(axis=1)
                        bb += stratum.base_loss.to_numpy()[draw].sum(axis=1)
                    change = 100 * (1 - np.sqrt(mb / bb))
                    low, high = np.quantile(change, [0.025, 0.975])
                    output.append(
                        dict(
                            candidate=candidate,
                            reference=ref,
                            case=case,
                            size_um=size,
                            n_sves=len(agg),
                            n_boot=2000,
                            rmse_reduction_pct=float(
                                100
                                * (1 - np.sqrt(agg.loss.mean() / agg.base_loss.mean()))
                            ),
                            reduction_ci_low=float(low),
                            reduction_ci_high=float(high),
                            bridge_rmse=float(np.sqrt(agg.loss.mean())),
                            tree_rmse=float(np.sqrt(agg.base_loss.mean())),
                            interval_scope="pointwise exploratory conditional fixed-prediction SVE bootstrap",
                        )
                    )
    paired = pd.DataFrame(output)
    assert len(paired) == 96
    paired.to_csv(
        HERE / "paired_bridge_comparisons.csv", index=False, float_format="%.17g"
    )
    final = dict(
        status="complete",
        n_tree_predictions=len(trees),
        n_paired_comparisons=len(paired),
        n_metrics_rows=len(metrics),
        all_tree_predictions_finite=True,
        primary_graph_predictions_unchanged=True,
        files={p.name: sha(p) for p in HERE.glob("*.csv") if ".partial." not in p.name},
    )
    dump("analysis_manifest.json", final)
    print(
        paired[
            paired.candidate.eq(CANDIDATES[0])
            & paired.case.str.startswith("hcf")
            & paired.size_um.astype(str).isin(["pooled", "125"])
        ][
            [
                "reference",
                "case",
                "size_um",
                "rmse_reduction_pct",
                "reduction_ci_low",
                "reduction_ci_high",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    analyze(fit())
