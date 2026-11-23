"""Exploratory predicted-input conditional MLE bridge; no label-based fitting."""
import os

os.environ.update(
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="",
    PYTHONDONTWRITEBYTECODE="1",
)
import sys

sys.dont_write_bytecode = True
import hashlib, importlib.util, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
OLD = ROOT / "revision_restart_2026_10_05/new_studies"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
KEYS = ["repeat", "fold", "dataset_idx", "size_um", "case"]
PRIMARY = "Conditional MLE graph bridge"
FALLBACK = " (fitting-size-mean fallback)"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


uncertainty = load("audited_uncertainty", STUDY / "uncertainty/run_uncertainty.py")
bridge = load("audited_bridge", STUDY / "bridge/evaluate_bridges.py")
fitter = uncertainty.load_fitter()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_bound(y, lower_log_t):
    """Archived objective and starts, changed lower numerical bound only."""
    m = float(np.mean(y))
    z = np.asarray(y) / m
    n = len(z)

    def obj(q):
        logt, logb = q
        t = np.exp(logt)
        b = np.exp(logb)
        v = z / b
        e = np.exp(-v)
        logsf = np.log(-np.expm1(-t))
        ll = n * (logt - logb) - v.sum() - t * e.sum() - n * logsf
        ratio = t / np.expm1(t) if t < 700 else 0.0
        grad = np.array([n - t * e.sum() - n * ratio, -n + v.sum() - t * (v * e).sum()])
        return -ll, -grad

    answers = [
        minimize(
            obj,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(lower_log_t, 8), (-4, 4)],
            options={"maxiter": 120, "ftol": 1e-10},
        )
        for start in [[-2, 0], [1, 0]]
    ]
    f = min(answers, key=lambda q: q.fun)
    t, b = np.exp(f.x)
    b *= m
    return {
        "t": float(t),
        "scale": float(b),
        "a": float(-f.x[0]),
        "boundary": bool(
            f.x[0] < lower_log_t + 0.01 or f.x[0] > 7.99 or abs(f.x[1]) > 3.99
        ),
        "exponential_limit_boundary": bool(f.x[0] < lower_log_t + 0.01),
        "optimization_success": bool(f.success),
        "log_likelihood_scaled": float(-f.fun),
    }


def estimate(values, lower_log_t=-12.0):
    """Receive predicted values only; return scalar and complete fitting ledger."""
    selected, u, rec = bridge.selected_population(values, "original_recut")
    rec = {
        **rec,
        "lower_log_t_bound": float(lower_log_t),
        "fit_attempted": False,
        "optimization_success": False,
        "boundary": False,
        "exponential_limit_boundary": False,
        "attempted_prediction": np.nan,
        "exception_type": "",
    }
    if rec["failure"] != "none":
        return np.nan, rec
    if len(selected) < 4:
        rec["failure"] = "fewer_than_four_selected"
        return np.nan, rec
    if not selected.std(ddof=1) > 0:
        rec["failure"] = "nonpositive_selected_sd"
        return np.nan, rec
    excess = selected - u
    if not (np.isfinite(excess).all() and np.all(excess > 0) and excess.mean() > 0):
        rec["failure"] = "nonpositive_or_nonfinite_excess"
        return np.nan, rec
    rec["fit_attempted"] = True
    try:
        f = fitter.fit(excess) if lower_log_t == -12 else fit_bound(excess, lower_log_t)
        rec.update({k: v for k, v in f.items() if k != "pit"})
        rec["conditional_location"] = u - f["a"] * f["scale"]
        rec["attempted_prediction"] = u + uncertainty.conditional_max_mean(
            len(selected), f["t"], f["scale"]
        )
        if not f["optimization_success"]:
            rec["failure"] = "optimizer_unsuccessful"
        elif not (
            f["t"] > 0
            and f["scale"] > 0
            and np.isfinite(rec["attempted_prediction"])
            and np.isfinite(rec["conditional_location"])
        ):
            rec["failure"] = "nonfinite_fit_or_scalar"
        else:
            rec["failure"] = "none"
    except Exception as e:
        rec["failure"] = "fitting_exception"
        rec["exception_type"] = type(e).__name__
    return (
        float(rec["attempted_prediction"]) if rec["failure"] == "none" else np.nan
    ), rec


def verify_fit(y):
    f = fitter.fit(y)
    g = fit_bound(y, -12.0)
    for k in ["t", "scale", "a", "log_likelihood_scaled"]:
        assert np.isclose(f[k], g[k], rtol=1e-13, atol=1e-13), (k, f[k], g[k])
    for k in ["boundary", "exponential_limit_boundary", "optimization_success"]:
        assert f[k] == g[k]


def main():
    start = time.perf_counter()
    protocol = HERE / "protocol.json"
    assert protocol.exists(), "Write the follow-up protocol before any execution."
    assert not (
        HERE / "predictions.csv"
    ).exists(), "Preserve original run; do not silently overwrite."
    sourcepaths = [
        protocol,
        Path(__file__),
        STUDY / "uncertainty/run_uncertainty.py",
        STUDY / "bridge/evaluate_bridges.py",
        OLD / "threshold_gof/run_independent_threshold_tail.py",
    ]
    sourcehashes = {str(p): sha(p) for p in sourcepaths}
    (HERE / "run_started.json").write_text(
        json.dumps(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha(protocol),
                "source_hashes": sourcehashes,
            },
            indent=2,
        )
        + "\n"
    )
    rng = np.random.default_rng(2026111506)
    for k in [4, 6, 48, 394]:
        for a in [-2.0, 0.0, 2.0]:
            verify_fit(uncertainty.conditional_q(rng.random(k), np.exp(-a)))
    rows = []
    ledger = []
    inputs = []
    sample_checks = 12
    for repeat in range(2):
        for fold in range(5):
            path = STUDY / "data" / f"grain_predictions_repeat{repeat}_fold{fold}.npz"
            with np.load(path, allow_pickle=False) as archive:
                # Truth is loaded solely to label the completed scalar outputs below.
                z = {
                    k: archive[k]
                    for k in [
                        "dataset_idx",
                        "size_um",
                        "grain_count",
                        "offsets",
                        "cases",
                        "pred_peak",
                        "y_true",
                    ]
                }
            assert list(z["cases"]) == CASES and np.isfinite(z["pred_peak"]).all()
            assert np.array_equal(np.diff(z["offsets"]), z["grain_count"])
            inputs.append({"path": str(path), "sha256": sha(path)})
            checked = False
            for j, (idx, size) in enumerate(zip(z["dataset_idx"], z["size_um"])):
                for c, case in enumerate(CASES):
                    values = z["pred_peak"][z["offsets"][j] : z["offsets"][j + 1], c]
                    selected, u, _ = bridge.selected_population(
                        values, "original_recut"
                    )
                    if not checked and len(selected) >= 4 and selected.std(ddof=1) > 0:
                        verify_fit(selected - u)
                        sample_checks += 1
                        checked = True
                    key = dict(
                        repeat=repeat,
                        fold=fold,
                        dataset_idx=int(idx),
                        size_um=int(size),
                        case=case,
                    )
                    # Neither estimate call receives any target/observed statistics.
                    for lower in [-12.0, -20.0]:
                        pred, rec = estimate(values, lower)
                        model = (
                            PRIMARY
                            if lower == -12
                            else PRIMARY + " (lower log-t bound -20)"
                        )
                        ledger.append(
                            {**key, "model": model, "prediction": pred, **rec}
                        )
                        rows.append(
                            {
                                **key,
                                "model": model,
                                "prediction": pred,
                                "failure": rec["failure"],
                                "used_fallback": False,
                                "truth": float(z["y_true"][j, c]),
                                "target": "SVE maximum",
                                "units": "dimensionless"
                                if case.endswith("fs")
                                else "MPa",
                            }
                        )
            print(
                "FITTED",
                repeat,
                fold,
                len(z["dataset_idx"]),
                "elapsed",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
    df = pd.DataFrame(rows)
    ld = pd.DataFrame(ledger)
    assert len(df) == 2 * 940 * 4 * 2 and not df.duplicated(KEYS + ["model"]).any()
    counts = df.groupby(["model", "repeat"]).dataset_idx.nunique()
    assert (counts == 940).all()
    refpath = OLD / "final_repeated_evaluation/all_outer_predictions.csv"
    ref = pd.read_csv(refpath)
    ref = ref[ref.model == "Fitting-size mean"][KEYS + ["prediction", "truth"]].rename(
        columns={"prediction": "fitting_size_mean", "truth": "reference_truth"}
    )
    assert len(ref) == 7520 and not ref.duplicated(KEYS).any()
    merged = df.merge(ref, on=KEYS, validate="many_to_one")
    assert np.allclose(merged.truth, merged.reference_truth, rtol=2e-6, atol=1e-12)
    operational = merged.copy()
    operational["used_fallback"] = ~np.isfinite(operational.prediction)
    operational.loc[operational.used_fallback, "prediction"] = operational.loc[
        operational.used_fallback, "fitting_size_mean"
    ]
    operational["model"] += FALLBACK
    assert np.isfinite(operational.prediction).all()
    df = pd.concat([df, operational[df.columns]], ignore_index=True)
    df.to_csv(HERE / "predictions.csv", index=False)
    ld.to_csv(HERE / "fit_ledger.csv", index=False)
    coverage = (
        ld.groupby(["model", "size_um", "case"])
        .agg(
            n=("dataset_idx", "size"),
            fit_attempts=("fit_attempted", "sum"),
            valid=("prediction", lambda x: int(np.isfinite(x).sum())),
            optimizer_success=("optimization_success", "sum"),
            boundaries=("boundary", "sum"),
            exponential_boundaries=("exponential_limit_boundary", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(HERE / "coverage_by_size_case.csv", index=False)
    ld.groupby(["model", "size_um", "case", "failure"]).size().rename(
        "count"
    ).reset_index().to_csv(HERE / "failure_counts.csv", index=False)
    a = ld[ld.lower_log_t_bound == -12].set_index(KEYS)
    b = ld[ld.lower_log_t_bound == -20].set_index(KEYS)
    comparison = pd.DataFrame(
        {
            "primary_prediction": a.prediction,
            "minus20_prediction": b.prediction,
            "primary_location": a.conditional_location,
            "minus20_location": b.conditional_location,
            "primary_scale": a.scale,
            "minus20_scale": b.scale,
            "prediction_difference": b.prediction - a.prediction,
            "location_difference": b.conditional_location - a.conditional_location,
            "primary_failure": a.failure,
            "minus20_failure": b.failure,
        }
    ).reset_index()
    comparison.to_csv(HERE / "bound_sensitivity_per_sve.csv", index=False)
    summary = (
        comparison.groupby(["size_um", "case"])
        .agg(
            n=("dataset_idx", "size"),
            both_valid=("prediction_difference", lambda x: int(np.isfinite(x).sum())),
            rms_prediction_difference=(
                "prediction_difference",
                lambda x: float(np.sqrt(np.nanmean(x * x))),
            ),
            max_abs_prediction_difference=(
                "prediction_difference",
                lambda x: float(np.nanmax(np.abs(x))),
            ),
            rms_parent_location_difference=(
                "location_difference",
                lambda x: float(np.sqrt(np.nanmean(x * x))),
            ),
        )
        .reset_index()
    )
    summary.to_csv(HERE / "bound_sensitivity_summary.csv", index=False)
    report = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - start,
        "scalar_fit_rows": len(ld),
        "exported_rows_with_two_numerical_variants_and_fallbacks": len(df),
        "unique_sves": 940,
        "repeats": 2,
        "archived_fitter_parity_samples": sample_checks,
        "cases": CASES,
        "models": df.model.unique().tolist(),
        "source_hashes": sourcehashes,
        "input_archives": inputs,
        "fallback_source_sha256": sha(refpath),
        "protocol": "Exploratory follow-up; not part of original prespecified set. No outcome-based selection.",
        "interpretation": "Conditional working-model scalar fits to predicted grain peaks; parent boundary estimates remain flagged. Targets attach only after scalar computation. All estimators, failures and numerical variants exported for independent scoring.",
        "coverage": coverage.to_dict("records"),
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "REPORT.txt").write_text(
        "Predicted-input conditional maximum-likelihood bridge\n\n"
        + report["protocol"]
        + "\n"
        + report["interpretation"]
        + "\n\n"
        + "Primary log(t) bounds [-12,8], mean-scaled scale log bounds [-4,4], two fixed starts. "
        + "The separately labeled lower-bound -20 rerun is a numerical sensitivity, not a selected estimator. "
        + "Optimizer failure leaves an undefined primary prediction; finite attempted values and status remain in fit_ledger.csv. "
        + "Near-exponential and other parameter boundaries are flagged but remain valid if the fit succeeds.\n\n"
        + "Coverage and boundary counts (both repeats combined)\n"
        + coverage.to_string(index=False)
        + "\n\n"
        + "Numerical bound sensitivity in raw case units\n"
        + summary.to_string(index=False)
        + "\n\n"
        + "Independent bridge evaluator will score all variants. These predictions do not establish an iid physical tail law or resolve dependence; the adaptive threshold and count are conditioned upon. "
        + "This method was motivated by earlier results in the same dataset and is exploratory.\n",
        encoding="utf-8",
    )
    (HERE / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "predictions_sha256": sha(HERE / "predictions.csv"),
                "fit_ledger_sha256": sha(HERE / "fit_ledger.csv"),
                "prediction_rows": len(df),
                "scalar_fit_rows": len(ld),
                "models": df.model.unique().tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "elapsed_seconds",
                    "scalar_fit_rows",
                    "exported_rows_with_two_numerical_variants_and_fallbacks",
                    "archived_fitter_parity_samples",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
