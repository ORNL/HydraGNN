"""Complete-cohort analysis of frozen literature-threshold feature maps.

No partial performance table is produced: all eight extraction manifests must
be complete and source-valid before either fitting or scoring begins.
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "revision_R2/studies_2026_11_15"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
SCOPES = ["pooled", *map(str, SIZES)]
FAMILIES = ["fixed", "eqd", "diagnostics", "predictive"]
SELECTORS = [
    "eqd",
    "ad_forwardstop",
    "ad_strongstop",
    "shape_stability",
    "predictive_select",
    "predictive_average",
]
METHODS = [
    "fixed_q0.5",
    "fixed_q0.75",
    "fixed_q0.9",
    "fixed_mean_sd",
    "grain_maximum",
    *SELECTORS,
]
KEY = ["repeat", "fold", "subset", "dataset_idx", "case"]
PREDKEY = ["repeat", "fold", "dataset_idx", "case"]
BOOTSTRAPS = 4000
BOOTSTRAP_SEED = 2026111607


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def verify_manifests(directory):
    manifests = []
    checked = {}
    for kind, total in [("predicted", 10520), ("observed", 3760)]:
        for family in FAMILIES:
            path = directory / f"{kind}_{family}_manifest.json"
            if not path.exists():
                raise RuntimeError(f"Missing required manifest: {path.name}")
            manifest = json.loads(path.read_text())
            assert manifest.get("status") == "complete", (path.name, "incomplete")
            assert manifest["tasks"] == manifest["completed"] == total
            assert manifest["limit"] == 0
            output = directory / f"{kind}_{family}.jsonl"
            assert sha(output) == manifest["output_sha256"], (
                output,
                "output hash mismatch",
            )
            for source, digest in manifest["sources"].items():
                if source not in checked:
                    checked[source] = sha(source)
                assert checked[source] == digest, (source, "source hash mismatch")
            for source, digest in manifest["code_hashes"].items():
                path_code = HERE / source
                key = str(path_code)
                if key not in checked:
                    checked[key] = sha(path_code)
                assert checked[key] == digest, (source, "code hash mismatch")
            manifests.append(dict(path=str(path), sha256=sha(path), **manifest))
    return manifests, checked


def read_fields(directory, kind):
    expected = 10520 if kind == "predicted" else 3760
    reference = {}
    rows = []
    diagnostics = []
    candidates = []
    timings = []
    quadrature = []
    for family in FAMILIES:
        seen = set()
        with (directory / f"{kind}_{family}.jsonl").open() as handle:
            for line in handle:
                d = json.loads(line)
                key = d["key"]
                assert key not in seen
                seen.add(key)
                meta = {
                    k: d[k]
                    for k in [
                        "kind",
                        "repeat",
                        "fold",
                        "subset",
                        "dataset_idx",
                        "case",
                        "size_um",
                        "grain_count",
                        "truth",
                        "field_sha256",
                    ]
                }
                if family == "fixed":
                    reference[key] = meta
                else:
                    assert reference[key] == meta, (family, key, "metadata mismatch")
                cs = {int(c["index"]): c for c in d["candidates"]}
                timings.append(
                    dict(
                        **{k: meta[k] for k in ["kind", "subset", "case", "size_um"]},
                        family=family,
                        seconds=d["seconds"],
                    )
                )
                if family == "fixed":
                    for c in cs.values():
                        candidates.append(
                            dict(
                                **meta,
                                candidate_index=c["index"],
                                quantile=c["quantile"],
                                threshold=c["threshold"],
                                count=c["count"],
                                shape=c["fit"]["shape"],
                                scale=c["fit"]["scale"],
                                fit_success=c["fit"]["success"],
                                boundary=c["fit"]["boundary"],
                            )
                        )
                for method, result in d["methods"].items():
                    assert method in METHODS and np.isfinite(result["raw_maximum"])
                    rows.append(
                        dict(**meta, method=method, raw_maximum=result["raw_maximum"])
                    )
                    status = str(result.get("status", ""))
                    map_status = str(result.get("map_status", ""))
                    fallback = (
                        "fallback" in status
                        or "fallback" in map_status
                        or status
                        in [
                            "no_valid_candidate_predicted_maximum",
                            "all_quadratures_failed_predicted_maximum",
                        ]
                    )
                    count = float(result.get("selected_count", 0))
                    boundary = float(result.get("fit", {}).get("boundary", False))
                    j = result.get("selected_index")
                    if j is not None and j >= 0 and j in cs:
                        boundary = float(cs[j]["fit"]["boundary"])
                    details = result.get("diagnostics", {})
                    qrecords = details.get("candidates", [])
                    if (
                        method == "predictive_average"
                        and status == "predictive_threshold_average"
                    ):
                        resolved = [q for q in qrecords if "weight" in q]
                        assert abs(sum(q["weight"] for q in resolved) - 1) < 1e-10
                        count = float(
                            sum(
                                q["weight"] * cs[int(q["index"])]["count"]
                                for q in resolved
                            )
                        )
                        boundary = float(
                            sum(
                                q["weight"] * cs[int(q["index"])]["fit"]["boundary"]
                                for q in resolved
                            )
                        )
                        assert count >= 10 and not fallback
                    diagnostics.append(
                        dict(
                            **meta,
                            method=method,
                            status=status,
                            map_status=map_status,
                            fallback=fallback,
                            selected_count=count,
                            selected_quantile=result.get("selected_quantile"),
                            selected_threshold=result.get("selected_threshold"),
                            selected_boundary_weight=boundary,
                            candidate_count=len(cs),
                            candidate_fit_failures=sum(
                                not c["fit"]["success"] for c in cs.values()
                            ),
                            candidate_boundary_count=sum(
                                c["fit"]["boundary"] for c in cs.values()
                            ),
                            bootstrap_failures=result.get("bootstrap_failures", 0),
                            bootstrap_boundary_fits=result.get("boundary_fits", 0),
                            bootstrap_fits=sum(
                                s.get("bootstrap_count", 0)
                                for s in result.get("scores", [])
                            ),
                            predictive_resolved=details.get("resolved_count"),
                            predictive_candidates=details.get("candidate_count"),
                            diagnostic_invalid_tests=sum(
                                not t.get("valid", True)
                                for t in result.get("tests", [])
                            ),
                        )
                    )
                    if method == "predictive_select":
                        for q in qrecords:
                            qq = q.get("quadrature", {})
                            full = qq.get("full_posterior", {})
                            deleted = qq.get("maximum_deleted_posterior", {})
                            quadrature.append(
                                dict(
                                    **meta,
                                    candidate_index=q["index"],
                                    score=q.get("score"),
                                    resolved=q.get("score") is not None,
                                    weight=q.get("weight", 0.0),
                                    full_ess=full.get("effective_sample_size"),
                                    maximum_deleted_ess=deleted.get(
                                        "effective_sample_size"
                                    ),
                                    minimum_nonmaximum_loo_ess=qq.get(
                                        "minimum_nonmaximum_loo_ess"
                                    ),
                                    full_nodes=full.get("nodes"),
                                    maximum_deleted_nodes=deleted.get("nodes"),
                                    full_mode_converged=full.get("mode_converged"),
                                    maximum_deleted_mode_converged=deleted.get(
                                        "mode_converged"
                                    ),
                                    full_max_weight=full.get("max_normalized_weight"),
                                    maximum_deleted_max_weight=deleted.get(
                                        "max_normalized_weight"
                                    ),
                                )
                            )
        assert len(seen) == expected, (kind, family, len(seen))
        assert seen == set(reference), (kind, family, "field identities differ")
    fields = pd.DataFrame(rows)
    assert not fields.duplicated(["method", *KEY]).any()
    assert set(fields.method) == set(METHODS)
    assert fields.groupby("method").size().eq(expected).all()
    return (
        fields,
        pd.DataFrame(diagnostics),
        pd.DataFrame(candidates),
        pd.DataFrame(timings),
        pd.DataFrame(quadrature),
    )


def validate_partition_rows(fields, parts, observed=False):
    reference = fields[fields.method.eq("grain_maximum")]
    for part in parts:
        if observed and part["repeat"] != 0:
            continue
        for subset in ["evaluation"] if observed else ["stopping", "evaluation"]:
            for case in CASES:
                g = reference[
                    reference.repeat.eq(part["repeat"])
                    & reference.fold.eq(part["fold"])
                    & reference.subset.eq(subset)
                    & reference.case.eq(case)
                ]
                assert set(g.dataset_idx) == set(part[subset]), (
                    part["repeat"],
                    part["fold"],
                    subset,
                    case,
                )
                assert not g.dataset_idx.duplicated().any()
        assert not set(part["evaluation"]) & (
            set(part["stopping"]) | set(part["fitting"])
        )
    if observed:
        assert reference.groupby("case").size().eq(940).all()
        np.testing.assert_allclose(
            reference.raw_maximum, reference.truth, rtol=2e-6, atol=1e-12
        )
    else:
        assert (
            reference[reference.subset.eq("evaluation")]
            .groupby(["repeat", "case"])
            .size()
            .eq(940)
            .all()
        )


def add_completed_gumbel(fields):
    path = OLD / "baselines/graph_tail_summaries.csv"
    summaries = pd.read_csv(path)
    assert len(summaries) == 10520 and summaries.model.eq("Graph").all()
    source = fields[fields.method.eq("grain_maximum")].drop(
        columns=["method", "raw_maximum"]
    )
    combined = source.merge(
        summaries.drop(columns=["model", "size_um"]), on=KEY, validate="one_to_one"
    )
    value = combined.tail_mean + np.sqrt(6) / np.pi * combined.sd_log_count
    valid = np.isfinite(value)
    completed = np.where(valid, value, combined.max_predicted_grain)
    assert np.isfinite(completed).all()
    extra = source.copy()
    extra["method"] = "gumbel_completed"
    # Merge does not necessarily preserve source order; use the merged frame.
    extra = combined[fields.columns.drop(["method", "raw_maximum"])].copy()
    extra["method"] = "gumbel_completed"
    extra["raw_maximum"] = completed
    ledger = combined[KEY + ["size_um", "selected_count"]].copy()
    ledger["fallback"] = ~valid
    return pd.concat([fields, extra[fields.columns]], ignore_index=True), ledger, path


def calibrate(fields, parts):
    rows = []
    fits = []
    for (method, repeat, fold, case), g in fields.groupby(
        ["method", "repeat", "fold", "case"], sort=True
    ):
        part = next(p for p in parts if p["repeat"] == repeat and p["fold"] == fold)
        tr = g[g.subset.eq("stopping")].sort_values("dataset_idx")
        ev = g[g.subset.eq("evaluation")].sort_values("dataset_idx")
        assert len(tr) == 75 and set(tr.dataset_idx) == set(part["stopping"])
        assert set(ev.dataset_idx) == set(part["evaluation"])
        x = tr[["raw_maximum"]].to_numpy()
        v = ev[["raw_maximum"]].to_numpy()
        y = tr.truth.to_numpy()
        scaler = StandardScaler().fit(x)
        estimator = Ridge(alpha=10.0, fit_intercept=True).fit(scaler.transform(x), y)
        prediction = estimator.predict(scaler.transform(v))
        assert np.isfinite(prediction).all()
        # Independent scalar closed form checks the actual fitted convention.
        z = scaler.transform(x)[:, 0]
        zv = scaler.transform(v)[:, 0]
        coefficient = float(z @ (y - y.mean()) / (z @ z + 10.0))
        np.testing.assert_allclose(
            prediction, y.mean() + zv * coefficient, rtol=2e-12, atol=1e-13
        )
        fits.append(
            dict(
                method=method,
                repeat=repeat,
                fold=fold,
                case=case,
                alpha=10.0,
                calibration_ids=tr.dataset_idx.tolist(),
                evaluation_ids=ev.dataset_idx.tolist(),
                n_calibration=len(tr),
                x_mean=float(scaler.mean_[0]),
                x_scale=float(scaler.scale_[0]),
                coefficient=float(estimator.coef_[0]),
                intercept=float(estimator.intercept_),
                input_sha256=hashlib.sha256(
                    np.column_stack([tr.dataset_idx, x[:, 0], y]).tobytes()
                ).hexdigest(),
            )
        )
        for stage, p in [("raw", v[:, 0]), ("calibrated", prediction)]:
            out = ev[PREDKEY + ["size_um", "truth"]].copy()
            out["model"] = stage + "/" + method
            out["prediction"] = p
            rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    return result, fits


def imports():
    paths = [
        OLD / "baselines/graph_calibration_predictions.csv",
        OLD / "baselines/outer_training_size_baseline_predictions.csv",
        OLD / "label_matched_trees/predictions_long.csv",
    ]
    mapping = {
        "Graph Affine bridge calibration": "Original calibrated bridge",
        "Graph Affine direct calibration": "Original calibrated direct head",
        "Graph Affine maximum calibration": "Original calibrated grain maximum",
        "Outer-training size mean": "Training size mean",
        "Outer-training Within-size count": "Training within-size count",
        "Outer-training Random forest": "Direct RF",
    }
    frames = []
    for path in paths:
        d = pd.read_csv(path)
        for name in d.model.unique():
            if name.startswith("Outer-training Within-size count"):
                mapping[name] = "Training within-size count"
        d = d[d.model.isin(mapping)].copy()
        d["model"] = d.model.map(mapping)
        frames.append(d[["model", *PREDKEY, "size_um", "truth", "prediction"]])
    out = pd.concat(frames, ignore_index=True)
    assert set(out.model) == {
        "Original calibrated bridge",
        "Original calibrated direct head",
        "Original calibrated grain maximum",
        "Training size mean",
        "Training within-size count",
        "Direct RF",
    }, set(out.model)
    assert out.groupby("model").size().eq(7520).all()
    return out, paths


def validate_predictions(predictions):
    assert np.isfinite(predictions[["truth", "prediction"]]).all().all()
    assert not predictions.duplicated(["model", "repeat", "case", "dataset_idx"]).any()
    assert predictions.groupby(["model", "repeat", "case"]).size().eq(940).all()
    reference = predictions[predictions.model.eq("Training size mean")][
        PREDKEY + ["size_um", "truth"]
    ]
    for model, d in predictions.groupby("model"):
        merged = d.merge(
            reference, on=PREDKEY, suffixes=("", "_reference"), validate="one_to_one"
        )
        assert (
            len(merged) == 7520 and (merged.size_um == merged.size_um_reference).all()
        ), model
        np.testing.assert_allclose(
            merged.truth, merged.truth_reference, rtol=2e-12, atol=1e-12
        )


def score(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    error = p - y
    denom = np.sum((y - y.mean()) ** 2)
    assert denom > 0
    variable = np.ptp(p) > 0
    return dict(
        n=len(y),
        r2=1 - np.sum(error ** 2) / denom,
        rmse=np.sqrt(np.mean(error ** 2)),
        mae=np.mean(abs(error)),
        bias=np.mean(error),
        truth_sd=np.std(y, ddof=1),
        pred_sd=np.std(p, ddof=1),
        pearson=np.corrcoef(y, p)[0, 1] if variable else 0.0,
        spearman=spearmanr(y, p).statistic if variable else 0.0,
        constant_prediction=not variable,
    )


def metrics(predictions):
    rows = []
    for (model, case, repeat), g in predictions.groupby(["model", "case", "repeat"]):
        for scope in SCOPES:
            q = g if scope == "pooled" else g[g.size_um.eq(int(scope))]
            rows.append(
                dict(
                    model=model,
                    case=case,
                    repeat=repeat,
                    size_um=scope,
                    **score(q.truth, q.prediction),
                )
            )
    by_repeat = pd.DataFrame(rows)
    columns = [
        "r2",
        "rmse",
        "mae",
        "bias",
        "truth_sd",
        "pred_sd",
        "pearson",
        "spearman",
    ]
    summary = by_repeat.groupby(["model", "case", "size_um"])[columns].agg(
        ["mean", "min", "max"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return by_repeat, summary.reset_index()


def observed_metrics(fields):
    rows = []
    for (method, case), g in fields.groupby(["method", "case"]):
        for scope in SCOPES:
            q = g if scope == "pooled" else g[g.size_um.eq(int(scope))]
            rows.append(
                dict(
                    method=method,
                    case=case,
                    size_um=scope,
                    **score(q.truth, q.raw_maximum),
                )
            )
    return pd.DataFrame(rows)


def bootstrap(predictions, out):
    d = predictions.copy()
    d["squared_error"] = (d.prediction - d.truth) ** 2
    mse = d.groupby(["model", "case", "dataset_idx"]).squared_error.mean()
    reference = d[
        d.model.eq("Training size mean") & d.case.eq(CASES[0]) & d.repeat.eq(0)
    ].sort_values("dataset_idx")
    ids = reference.dataset_idx.to_numpy()
    sizes = reference.size_um.to_numpy()
    assert np.array_equal(ids, np.arange(940))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # The same parent-SVE draws preserve covariance across all four cases and
    # all comparisons. Pooled draws concatenate the same five size-specific draws.
    drawings = {
        str(size): rng.choice(
            np.flatnonzero(sizes == size),
            size=(BOOTSTRAPS, np.sum(sizes == size)),
            replace=True,
        )
        for size in SIZES
    }
    drawings["pooled"] = np.concatenate([drawings[str(s)] for s in SIZES], axis=1)
    np.savez_compressed(
        out / "bootstrap_parent_indices.npz",
        **{f"size_{s}": x.astype(np.int16) for s, x in drawings.items()},
        seed=np.array(BOOTSTRAP_SEED),
        dataset_idx=ids,
    )
    models = ["calibrated/" + s for s in SELECTORS]
    references = [
        "Original calibrated bridge",
        "calibrated/fixed_q0.5",
        "Training size mean",
    ]
    all_models = sorted(set(models + references))
    arrays = {}
    observed = {}
    for case in CASES:
        for model in all_models:
            error = mse.loc[(model, case)].reindex(ids).to_numpy()
            assert np.isfinite(error).all()
            for scope in SCOPES:
                ix = (
                    np.arange(940)
                    if scope == "pooled"
                    else np.flatnonzero(sizes == int(scope))
                )
                observed[(model, case, scope)] = float(np.sqrt(error[ix].mean()))
                arrays[(model, case, scope)] = np.sqrt(
                    error[drawings[scope]].mean(axis=1)
                )
    records = []
    draws = []
    for model in models:
        for case in CASES:
            for scope in SCOPES:
                for ref in references:
                    estimate = 100 * (
                        1
                        - observed[(model, case, scope)] / observed[(ref, case, scope)]
                    )
                    sample = 100 * (
                        1 - arrays[(model, case, scope)] / arrays[(ref, case, scope)]
                    )
                    lo, hi = np.quantile(sample, [0.025, 0.975])
                    simultaneous = ref != "Training size mean"
                    records.append(
                        dict(
                            model=model,
                            reference=ref,
                            case=case,
                            size_um=scope,
                            rmse_reduction_pct=estimate,
                            ci_low=lo,
                            ci_high=hi,
                            bootstrap_se=float(sample.std(ddof=1)),
                            simultaneous_family=simultaneous,
                            n_sves=len(drawings[scope][0]),
                            model_rmse=observed[(model, case, scope)],
                            reference_rmse=observed[(ref, case, scope)],
                        )
                    )
                    draws.append(sample)
    frame = pd.DataFrame(records)
    draws = np.column_stack(draws)
    family = frame.simultaneous_family.to_numpy()
    assert family.sum() == 288 and len(frame) == 432
    se = frame.bootstrap_se.to_numpy()
    centered = draws[:, family] - frame.rmse_reduction_pct.to_numpy()[None, family]
    standardized = np.divide(
        abs(centered),
        se[None, family],
        out=np.zeros_like(centered),
        where=se[None, family] > 0,
    )
    maxima = standardized.max(axis=1)
    critical = float(np.quantile(maxima, 0.95))
    frame["simultaneous_ci_low"] = np.where(
        family, frame.rmse_reduction_pct - critical * se, np.nan
    )
    frame["simultaneous_ci_high"] = np.where(
        family, frame.rmse_reduction_pct + critical * se, np.nan
    )
    np.savez_compressed(
        out / "bootstrap_contrast_draws.npz",
        draws=draws,
        family_indices=np.flatnonzero(family),
        max_standardized_deviation=maxima,
        critical=np.array(critical),
    )
    info = dict(
        replicates=BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
        physical_sves=940,
        repeat_squared_errors_averaged_before_resampling=True,
        shared_resamples_across_cases_and_contrasts=True,
        pooled_stratified_by_size=True,
        pointwise_contrasts=432,
        simultaneous_contrasts=288,
        critical_max_standardized_deviation=critical,
        simultaneous_rule="theta +/- q95(max_j(abs(theta_boot_j-theta_j)/sd_boot_j))*sd_boot_j",
        family="6 calibrated selectors x4cases x6scopes x2references(original bridge and calibrated q50)",
        uncertainty_scope="Fixed predictions on reused cohort; excludes training and prior exploratory-search variability",
    )
    return frame, info


def summarize_diagnostics(diagnostics, timings, quadrature):
    rows = []
    for (kind, subset, method, case), g in diagnostics.groupby(
        ["kind", "subset", "method", "case"]
    ):
        for scope in SCOPES:
            q = g if scope == "pooled" else g[g.size_um.eq(int(scope))]
            good = q[~q.fallback]
            rows.append(
                dict(
                    kind=kind,
                    subset=subset,
                    method=method,
                    case=case,
                    size_um=scope,
                    n=len(q),
                    fallback_count=int(q.fallback.sum()),
                    fallback_fraction=float(q.fallback.mean()),
                    selected_count_mean=float(q.selected_count.mean()),
                    selected_count_median=float(q.selected_count.median()),
                    selected_count_valid_mean=float(good.selected_count.mean())
                    if len(good)
                    else None,
                    selected_quantile_mean=float(q.selected_quantile.mean()),
                    selected_boundary_weight_mean=float(
                        q.selected_boundary_weight.mean()
                    ),
                    candidate_count_mean=float(q.candidate_count.mean()),
                    candidate_fit_failures=int(q.candidate_fit_failures.sum()),
                    candidate_boundary_count=int(q.candidate_boundary_count.sum()),
                    candidate_fits=int(q.candidate_count.sum()),
                    bootstrap_failures=int(q.bootstrap_failures.sum()),
                    bootstrap_boundary_fits=int(q.bootstrap_boundary_fits.sum()),
                    bootstrap_fits=int(q.bootstrap_fits.sum()),
                )
            )
    runtime = (
        timings.groupby(["kind", "family"])
        .seconds.agg(["count", "sum", "mean", "median", "max"])
        .reset_index()
    )
    qsummary = []
    for (kind, subset), g in quadrature.groupby(["kind", "subset"]):
        qsummary.append(
            dict(
                kind=kind,
                subset=subset,
                candidates=len(g),
                failures=int((~g.resolved).sum()),
                full_ess_min=float(g.full_ess.min()),
                full_ess_median=float(g.full_ess.median()),
                maximum_deleted_ess_min=float(g.maximum_deleted_ess.min()),
                minimum_nonmaximum_loo_ess=float(g.minimum_nonmaximum_loo_ess.min()),
                maximum_full_weight=float(g.full_max_weight.max()),
                maximum_deleted_weight=float(g.maximum_deleted_max_weight.max()),
                full_mode_nonconverged=int(g.full_mode_converged.eq(False).sum()),
                maximum_deleted_mode_nonconverged=int(
                    g.maximum_deleted_mode_converged.eq(False).sum()
                ),
                maximum_nodes=float(
                    max(g.full_nodes.max(), g.maximum_deleted_nodes.max())
                ),
            )
        )
    return pd.DataFrame(rows), runtime, pd.DataFrame(qsummary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "field_results")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    start = time.time()
    manifests, source_hashes = verify_manifests(args.input)
    if args.check_only:
        print(
            "All eight extraction manifests complete and source hashes verified. No performance scores computed."
        )
        return
    args.output.mkdir(exist_ok=True, parents=True)
    parts = json.loads((BASE / "partitions.json").read_text())
    predicted, dp, cp, tp, qp = read_fields(args.input, "predicted")
    observed, do, co, to, qo = read_fields(args.input, "observed")
    validate_partition_rows(predicted, parts)
    validate_partition_rows(observed, parts, observed=True)
    # Shared true targets and observed fields are identical across repetitions.
    a = predicted[
        predicted.method.eq("grain_maximum")
        & predicted.subset.eq("evaluation")
        & predicted.repeat.eq(0)
    ]
    b = observed[observed.method.eq("grain_maximum")]
    merged = a.merge(
        b,
        on=["repeat", "fold", "dataset_idx", "case"],
        suffixes=("_pred", "_obs"),
        validate="one_to_one",
    )
    np.testing.assert_array_equal(merged.truth_pred, merged.truth_obs)
    predicted, completion_ledger, gumbel_path = add_completed_gumbel(predicted)
    new, fits = calibrate(predicted, parts)
    old, old_paths = imports()
    predictions = pd.concat([new, old], ignore_index=True)
    validate_predictions(predictions)
    by_repeat, summary = metrics(predictions)
    observed_scores = observed_metrics(observed)
    diagnostics = pd.concat([dp, do], ignore_index=True)
    candidate_rows = pd.concat([cp, co], ignore_index=True)
    timings = pd.concat([tp, to], ignore_index=True)
    quadrature = pd.concat([qp, qo], ignore_index=True)
    diagnostic_summary, runtime, qsummary = summarize_diagnostics(
        diagnostics, timings, quadrature
    )
    execution = pd.DataFrame(
        [
            dict(
                kind=m["kind"],
                family=m["family"],
                wall_seconds=m["elapsed_seconds"],
                workers=m["workers"],
            )
            for m in manifests
        ]
    )
    runtime = runtime.merge(execution, on=["kind", "family"], validate="one_to_one")
    comparisons, bootstrap_info = bootstrap(predictions, args.output)
    outputs = {
        "predictions.csv.gz": predictions,
        "metrics_by_repeat.csv": by_repeat,
        "metrics_summary.csv": summary,
        "observed_predictions.csv.gz": observed,
        "observed_metrics.csv": observed_scores,
        "selector_diagnostics.csv.gz": diagnostics,
        "selector_diagnostics_summary.csv": diagnostic_summary,
        "candidate_fits.csv.gz": candidate_rows,
        "runtime_summary.csv": runtime,
        "predictive_quadrature.csv.gz": quadrature,
        "predictive_quadrature_summary.csv": qsummary,
        "paired_comparisons.csv": comparisons,
        "gumbel_completion_ledger.csv": completion_ledger,
    }
    for name, frame in outputs.items():
        temporary = args.output / ("temporary_" + name)
        frame.to_csv(temporary, index=False)
        temporary.replace(args.output / name)
    dump(args.output / "calibration_coefficients.json", fits)
    dump(args.output / "bootstrap_protocol.json", bootstrap_info)
    source_hashes.update(
        {
            str(p): sha(p)
            for p in [*old_paths, gumbel_path, BASE / "partitions.json", Path(__file__)]
        }
    )
    verification = dict(
        status="complete",
        seconds=time.time() - start,
        predicted_fields=10520,
        observed_fields=3760,
        evaluated_fields=7520,
        selector_methods=len(METHODS),
        new_scalar_routes=12,
        predictive_models=predictions.model.nunique(),
        predictive_rows=len(predictions),
        finite_predictions=True,
        no_omitted_sves=True,
        identical_evaluation_identities=True,
        source_hashes=source_hashes,
        extraction_manifests=[
            dict(path=m["path"], sha256=m["sha256"]) for m in manifests
        ],
        calibration="StandardScaler + scalar Ridge10, all75stopping, freeintercept, no clipping",
        calibrator_count=len(fits),
        calibration_closed_form_check_passed=True,
        all_reference_predictions_imported_unchanged=True,
        bootstrap=bootstrap_info,
        outputs={name: sha(args.output / name) for name in outputs},
    )
    dump(args.output / "verification.json", verification)
    names = [
        "Original calibrated bridge",
        "calibrated/gumbel_completed",
        "Training size mean",
        "Direct RF",
        "calibrated/fixed_q0.5",
        *["calibrated/" + s for s in SELECTORS],
    ]
    small_names = [
        "Original calibrated bridge",
        "Training size mean",
        "Direct RF",
        "calibrated/fixed_q0.5",
        "calibrated/gumbel_completed",
        *["calibrated/" + s for s in SELECTORS],
    ]
    small_raw = [
        "raw/gumbel_completed",
        "raw/grain_maximum",
        "raw/fixed_q0.5",
        *["raw/" + s for s in SELECTORS],
    ]
    report = [
        "LITERATURE THRESHOLD COMPARISON: COMPLETE NUMERICAL ANALYSIS",
        "",
        "All 940 physical SVEs, two preserved outer repeats, and all four cases. All methods and scopes are reported.",
        "Every new scalar map uses the same stopping-only standardized Ridge10 calibration.",
        "Observed-field reconstructions are separate and are not deployable predictions or a ceiling.",
        "",
        "Pooled R2 (mean over two full-cohort repeats):",
        summary[summary.model.isin(names) & summary.size_um.eq("pooled")]
        .pivot(index="model", columns="case", values="r2_mean")[CASES]
        .to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "125 micrometre calibrated R2:",
        summary[summary.model.isin(small_names) & summary.size_um.eq("125")]
        .pivot(index="model", columns="case", values="r2_mean")[CASES]
        .to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "125 micrometre raw R2, all using the SAME predicted GNN grain fields:",
        summary[summary.model.isin(small_raw) & summary.size_um.eq("125")]
        .pivot(index="model", columns="case", values="r2_mean")[CASES]
        .to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "125 micrometre observed-input reconstruction R2:",
        observed_scores[observed_scores.size_um.eq("125")]
        .pivot(index="method", columns="case", values="r2")[CASES]
        .to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "Interpretation of the small-size comparison:",
        "The automatic threshold procedures and the original bridge share the same GNN grain predictions. This is a comparison of aggregation maps, not GNN versus EVT.",
        "The original calibrated bridge has the strongest HCF/FS result among these methods at 125 micrometres. Fixed-q50 GPD slightly exceeds it for HCF/WP. The LCF/WP limitation remains.",
        "At 125 micrometres, observed fixed-q75, fixed-q90 and mean-plus-SD controls all return the observed maximum because fewer than ten excesses remain. Their R2=1 is an identity fallback, not a successful tail fit.",
        "EQD has no observed-input fallback at that size. Its observed reconstruction R2 ranges from about .78 to .89, whereas its raw predicted-field performance is much weaker.",
        "Many small-size GPD fits reach the imposed shape boundary. Therefore these results apply to the specified regularized procedures; they do not establish that every possible literature estimator fails or validate a Gumbel assumption.",
        "Fallback rates, actual selected counts and boundary frequencies are in selector_diagnostics_summary.csv.",
        "",
        "500 micrometre R2:",
        summary[summary.model.isin(names) & summary.size_um.eq("500")]
        .pivot(index="model", columns="case", values="r2_mean")[CASES]
        .to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "Paired RMSE gains over original bridge (pooled):",
        comparisons[
            comparisons.reference.eq("Original calibrated bridge")
            & comparisons.size_um.eq("pooled")
        ][
            [
                "model",
                "case",
                "rmse_reduction_pct",
                "ci_low",
                "ci_high",
                "simultaneous_ci_low",
                "simultaneous_ci_high",
            ]
        ].to_string(
            index=False, float_format=lambda x: f"{x:.5f}"
        ),
        "",
        "The simultaneous family contains 288 prespecified contrasts; critical standardized deviation="
        + str(bootstrap_info["critical_max_standardized_deviation"]),
        "Intervals describe fixed-prediction cohort uncertainty; they exclude prior adaptive development and neural retraining variability.",
        "No manuscript method is selected or changed by this analysis.",
        "",
    ]
    (args.output / "REPORT.txt").write_text("\n".join(report))
    print(
        json.dumps(
            {
                k: verification[k]
                for k in [
                    "status",
                    "seconds",
                    "predictive_models",
                    "predictive_rows",
                    "calibrator_count",
                ]
            },
            indent=2,
        )
    )
    print(report[7])


if __name__ == "__main__":
    main()
