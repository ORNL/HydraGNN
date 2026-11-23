"""All-arm sensitivity analysis; no best-arm selection or original-file edits."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import analyze_results as analysis

ARMS = ["lo049_n10", "lo049_n4", "lo090_n10", "lo090_n4"]
METHODS = ["eqd", "fixed_q0.5", "fixed_mean_sd"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze_analysis_protocol():
    protocol = dict(
        arms=ARMS,
        methods=METHODS,
        calibration="same standardized scalar Ridge10, all75stopping, no clipping",
        bootstrap_draws=str(BASE / "analysis/bootstrap_parent_indices.npz"),
        bootstrap_draws_sha256=sha(BASE / "analysis/bootstrap_parent_indices.npz"),
        references=[
            "Original calibrated bridge",
            "Training size mean",
            "same-method lo049_n10 for three changed arms",
        ],
        simultaneous_contrasts=792,
        simultaneous_family="12 routes x4cases x6scopes x2externalreferences plus9changedroutes x4cases x6scopes versus matchedbase",
        rule="same fixed-standard-error max-deviation bootstrap as primary analysis",
        all_variants_reported=True,
        no_outer_selection=True,
    )
    path = HERE / "ANALYSIS_PROTOCOL.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        analysis.dump(path, protocol)
    return protocol


def verify():
    checked = {}
    manifests = []
    for arm in ARMS:
        path = HERE / f"fields_{arm}_manifest.json"
        assert path.exists(), f"Not complete: {arm}"
        m = json.loads(path.read_text())
        assert m.get("status") == "complete"
        assert m["tasks"] == m["completed"] == 14280
        assert m["protocol_sha256"] == sha(HERE / "PROTOCOL.txt")
        assert m["runner_sha256"] == sha(HERE / "run_sensitivity.py")
        assert m["numerical_verification_sha256"] == sha(
            HERE / "numerical_verification.json"
        )
        assert m["output_sha256"] == sha(HERE / f"fields_{arm}.jsonl")
        for file, digest in m["clone"]["code_hashes"].items():
            assert sha(HERE / arm / file) == digest
        for file, digest in m["sources"].items():
            if file not in checked:
                checked[file] = sha(file)
            assert checked[file] == digest
        manifests.append(m)
    originals = json.loads((HERE / "clones_manifest.json").read_text())[
        "original_sources"
    ]
    for file, digest in originals.items():
        assert sha(BASE / file) == digest, ("original modified", file)
    return manifests


def original_fields():
    result = {}
    for kind in ["predicted", "observed"]:
        for family in ["fixed", "eqd"]:
            with (BASE / "field_results" / f"{kind}_{family}.jsonl").open() as file:
                for line in file:
                    d = json.loads(line)
                    for method in METHODS:
                        if method in d["methods"]:
                            result[(d["key"], method)] = d["methods"][method]
    assert len(result) == 14280 * 3
    return result


def read():
    frozen = original_fields()
    rows = []
    diagnostics = []
    seenbase = {}
    reproduced = 0
    for arm in ARMS:
        low = -0.49 if arm.startswith("lo049") else -0.90
        seen = set()
        with (HERE / f"fields_{arm}.jsonl").open() as file:
            for line in file:
                d = json.loads(line)
                assert d["key"] not in seen
                seen.add(d["key"])
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
                if arm == "lo049_n10":
                    seenbase[d["key"]] = meta
                else:
                    assert seenbase[d["key"]] == meta
                for method, value in d["methods"].items():
                    assert method in METHODS and np.isfinite(value["raw_maximum"])
                    if arm == "lo049_n10":
                        assert value == frozen[(d["key"], method)], (
                            "base arm differs",
                            d["key"],
                            method,
                        )
                        reproduced += 1
                    rows.append(
                        dict(
                            **meta,
                            arm=arm,
                            scalar_method=method,
                            method=arm + "/" + method,
                            raw_maximum=value["raw_maximum"],
                        )
                    )
                    fit = value.get("fit", {})
                    shape = fit.get("shape")
                    status = value["status"]
                    fallback = "fallback" in status
                    diagnostics.append(
                        dict(
                            **meta,
                            arm=arm,
                            method=method,
                            status=status,
                            fallback=fallback,
                            selected_count=value.get("selected_count", 0),
                            selected_quantile=value.get("selected_quantile"),
                            selected_shape=shape,
                            lower_bound=low,
                            lower_boundary=(shape is not None and shape <= low + 1e-6),
                            upper_boundary=(shape is not None and shape >= 0.95 - 1e-6),
                            candidate_count=len(d["candidates"]),
                            candidate_lower_boundary=sum(
                                c["fit"]["shape"] <= low + 1e-6 for c in d["candidates"]
                            ),
                            candidate_upper_boundary=sum(
                                c["fit"]["shape"] >= 0.95 - 1e-6
                                for c in d["candidates"]
                            ),
                            candidate_fit_failures=sum(
                                not c["fit"]["success"] for c in d["candidates"]
                            ),
                            bootstrap_failures=value.get("bootstrap_failures", 0),
                            field_seconds=d["seconds"],
                        )
                    )
        assert len(seen) == 14280 and seen == set(seenbase)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), reproduced


def paired(predictions, out):
    errors = predictions.copy()
    errors["se"] = (errors.prediction - errors.truth) ** 2
    mse = errors.groupby(["model", "case", "dataset_idx"]).se.mean()
    with np.load(BASE / "analysis/bootstrap_parent_indices.npz") as z:
        drawings = {scope: z["size_" + scope] for scope in analysis.SCOPES}
        ids = z["dataset_idx"]
    ref = predictions[
        predictions.model.eq("Training size mean")
        & predictions.case.eq(analysis.CASES[0])
        & predictions.repeat.eq(0)
    ].sort_values("dataset_idx")
    sizes = ref.size_um.to_numpy()
    assert np.array_equal(ids, ref.dataset_idx)
    model_names = [
        "calibrated/" + arm + "/" + method for arm in ARMS for method in METHODS
    ]
    all_models = model_names + ["Original calibrated bridge", "Training size mean"]
    rmse = {}
    draw = {}
    for model in all_models:
        for case in analysis.CASES:
            e = mse.loc[(model, case)].reindex(ids).to_numpy()
            assert np.isfinite(e).all()
            for scope in analysis.SCOPES:
                ix = (
                    np.arange(940)
                    if scope == "pooled"
                    else np.flatnonzero(sizes == int(scope))
                )
                rmse[(model, case, scope)] = np.sqrt(e[ix].mean())
                draw[(model, case, scope)] = np.sqrt(e[drawings[scope]].mean(axis=1))
    records = []
    samples = []
    for arm in ARMS:
        for method in METHODS:
            model = "calibrated/" + arm + "/" + method
            refs = ["Original calibrated bridge", "Training size mean"]
            if arm != "lo049_n10":
                refs.append("calibrated/lo049_n10/" + method)
            for case in analysis.CASES:
                for scope in analysis.SCOPES:
                    for reference in refs:
                        estimate = 100 * (
                            1
                            - rmse[(model, case, scope)]
                            / rmse[(reference, case, scope)]
                        )
                        sample = 100 * (
                            1
                            - draw[(model, case, scope)]
                            / draw[(reference, case, scope)]
                        )
                        lo, hi = np.quantile(sample, [0.025, 0.975])
                        records.append(
                            dict(
                                model=model,
                                arm=arm,
                                method=method,
                                reference=reference,
                                case=case,
                                size_um=scope,
                                rmse_reduction_pct=estimate,
                                ci_low=lo,
                                ci_high=hi,
                                bootstrap_se=sample.std(ddof=1),
                            )
                        )
                        samples.append(sample)
    table = pd.DataFrame(records)
    samples = np.column_stack(samples)
    assert len(table) == 792
    se = table.bootstrap_se.to_numpy()
    standardized = np.divide(
        abs(samples - table.rmse_reduction_pct.to_numpy()[None, :]),
        se[None, :],
        out=np.zeros_like(samples),
        where=se[None, :] > 0,
    )
    critical = float(np.quantile(standardized.max(axis=1), 0.95))
    table["simultaneous_ci_low"] = table.rmse_reduction_pct - critical * se
    table["simultaneous_ci_high"] = table.rmse_reduction_pct + critical * se
    np.savez_compressed(
        out / "bootstrap_contrasts.npz", draws=samples, critical=np.array(critical)
    )
    return table, critical


def main():
    protocol = freeze_analysis_protocol()
    if len(sys.argv) > 1 and sys.argv[1] == "--freeze-only":
        return
    start = time.time()
    manifests = verify()
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    fields, diagnostics, reproduced = read()
    parts = json.loads((analysis.BASE / "partitions.json").read_text())
    predicted = fields[fields.kind.eq("predicted")]
    observed = fields[fields.kind.eq("observed")]
    assert predicted.groupby(["arm", "scalar_method"]).size().eq(10520).all()
    assert observed.groupby(["arm", "scalar_method"]).size().eq(3760).all()
    new, fits = analysis.calibrate(predicted, parts)
    old = pd.read_csv(
        BASE / "analysis/predictions.csv.gz", float_precision="round_trip"
    )
    refs = [
        "Original calibrated bridge",
        "Training size mean",
        "calibrated/gumbel_completed",
        "Direct RF",
    ]
    # In addition to exact field-map replay, compare all recomputed base-arm
    # raw/calibrated predictions with their originally saved counterparts.
    replayed = 0
    for method in METHODS:
        for stage in ["raw", "calibrated"]:
            a = new[new.model.eq(stage + "/lo049_n10/" + method)].sort_values(
                analysis.PREDKEY
            )
            b = old[old.model.eq(stage + "/" + method)].sort_values(analysis.PREDKEY)
            assert len(a) == len(b) == 7520
            np.testing.assert_array_equal(
                a[analysis.PREDKEY].to_numpy(), b[analysis.PREDKEY].to_numpy()
            )
            np.testing.assert_array_equal(
                a.prediction.to_numpy(), b.prediction.to_numpy()
            )
            replayed += len(a)
    predictions = pd.concat([new, old[old.model.isin(refs)]], ignore_index=True)
    analysis.validate_predictions(predictions)
    by_repeat, summary = analysis.metrics(predictions)
    observed_scores = analysis.observed_metrics(observed)
    comparisons, critical = paired(predictions, out)
    ds = []
    for (kind, subset, arm, method, case), g in diagnostics.groupby(
        ["kind", "subset", "arm", "method", "case"]
    ):
        for scope in analysis.SCOPES:
            q = g if scope == "pooled" else g[g.size_um.eq(int(scope))]
            ds.append(
                dict(
                    kind=kind,
                    subset=subset,
                    arm=arm,
                    method=method,
                    case=case,
                    size_um=scope,
                    n=len(q),
                    fallback_fraction=q.fallback.mean(),
                    fallback_count=int(q.fallback.sum()),
                    selected_count_mean=q.selected_count.mean(),
                    selected_count_median=q.selected_count.median(),
                    selected_quantile_mean=q.selected_quantile.mean(),
                    lower_boundary_fraction=q.lower_boundary.mean(),
                    upper_boundary_fraction=q.upper_boundary.mean(),
                    candidate_count_mean=q.candidate_count.mean(),
                    candidate_fit_failures=int(q.candidate_fit_failures.sum()),
                    bootstrap_failures=int(q.bootstrap_failures.sum()),
                )
            )
    outputs = {
        "predictions.csv.gz": predictions,
        "metrics_by_repeat.csv": by_repeat,
        "metrics_summary.csv": summary,
        "observed_predictions.csv.gz": observed,
        "observed_metrics.csv": observed_scores,
        "diagnostics.csv.gz": diagnostics,
        "diagnostics_summary.csv": pd.DataFrame(ds),
        "paired_comparisons.csv": comparisons,
    }
    for file, frame in outputs.items():
        frame.to_csv(out / file, index=False)
    analysis.dump(out / "calibration_coefficients.json", fits)
    verification = dict(
        status="complete",
        seconds=time.time() - start,
        predictions=len(predictions),
        models=predictions.model.nunique(),
        numerical_checks=sha(HERE / "numerical_verification.json"),
        exact_base_map_reproductions=reproduced,
        exact_base_prediction_reproductions=replayed,
        all_finite=True,
        no_omitted_sves=True,
        calibrators=len(fits),
        pointwise_and_simultaneous_contrasts=792,
        simultaneous_critical=critical,
        original_sources_unchanged=True,
        analysis_source_sha256=sha(__file__),
        primary_analysis_sha256=sha(analysis.__file__),
        protocol=protocol,
        extraction_manifests={
            arm: sha(HERE / f"fields_{arm}_manifest.json") for arm in ARMS
        },
        outputs={file: sha(out / file) for file in outputs},
    )
    analysis.dump(out / "verification.json", verification)
    lines = [
        "SHAPE-BOUND AND MINIMUM-COUNT SENSITIVITY",
        "",
        "All four prescribed arms and all three methods are reported. No arm was chosen from evaluation performance.",
        "Base-arm field maps and calibrated predictions reproduced the frozen main comparison exactly.",
        "Negative shapes below -0.5 are used only for the bounded EQD/fixed-map sensitivity, without score-test or GOF calibration.",
        "",
    ]
    for scope in ["pooled", "125", "500"]:
        lines.extend(
            [
                f"Calibrated R2: {scope}",
                summary[
                    (
                        summary.model.str.startswith("calibrated/")
                        | summary.model.isin(refs)
                    )
                    & summary.size_um.eq(scope)
                ]
                .pivot(index="model", columns="case", values="r2_mean")[analysis.CASES]
                .to_string(float_format=lambda x: f"{x:.6f}"),
                "",
            ]
        )
    lines.extend(
        [
            "125 micrometre raw predictive R2:",
            summary[summary.model.str.startswith("raw/") & summary.size_um.eq("125")]
            .pivot(index="model", columns="case", values="r2_mean")[analysis.CASES]
            .to_string(float_format=lambda x: f"{x:.6f}"),
            "",
            "125 micrometre observed-input reconstruction R2:",
            observed_scores[observed_scores.size_um.eq("125")]
            .pivot(index="method", columns="case", values="r2")[analysis.CASES]
            .to_string(float_format=lambda x: f"{x:.6f}"),
            "",
            "Observed fields contain the maximum. Their map reconstruction is not deployable prediction or a ceiling.",
            "Diagnostics report fallbacks and shape-boundary frequencies explicitly.",
            "",
        ]
    )
    (out / "REPORT.txt").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                k: verification[k]
                for k in [
                    "status",
                    "seconds",
                    "predictions",
                    "models",
                    "exact_base_map_reproductions",
                    "exact_base_prediction_reproductions",
                ]
            },
            indent=2,
        )
    )
    print(lines[7])


if __name__ == "__main__":
    main()
