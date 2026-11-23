"""Frozen 940-SVE bridge audit with fixed and documented follow-up estimators.

Writes only its own directory. Neural checkpoints stay frozen and threshold
variants are fixed. Secondary controls include stopping-trained calibrators and
per-SVE distribution fits to predicted peaks; no outer-label fitting or tuning
occurs. Repeated predictions remain clustered by SVE when comparing errors.
"""
from __future__ import annotations
from fip_release_paths import release_root
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import argparse
import hashlib
import importlib.util
import json
import math
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import digamma
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
BENCH = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
DATA = STUDY / "data"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
GAMMA = 0.5772156649015329
C = math.sqrt(6) / math.pi
KEYS = ["repeat", "fold", "dataset_idx", "size_um", "case"]
SELECTORS = [
    "original_recut",
    "literal_c1",
    "literal_c0p5",
    "literal_c1p5",
    "quantile075",
    "quantile090",
]


def digest(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def load_original():
    path = Path(
        str(
            release_root()
            / "examples/fip_unified/sim_evt_validation/code/_evt_mixture.py"
        )
    )
    if not path.exists():
        return (None, None)
    spec = importlib.util.spec_from_file_location("historical_evt_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module, path)


class ConditionalGumbel:
    """Existing 256-node moment inversion, retaining inadmissible ratios as failures."""

    def __init__(self, nodes=256):
        z, w = leggauss(nodes)
        self.p, self.w = ((z + 1) / 2, w / 2)
        self.a = np.linspace(-12, 12, 2401)
        fa = np.exp(-np.exp(-self.a))
        prob = fa[:, None] + (1 - fa[:, None]) * self.p
        q = -np.log(-np.log(np.minimum(prob, np.nextafter(1.0, 0.0))))
        excess = q - self.a[:, None]
        self.mean = excess @ self.w
        self.sd = np.sqrt((excess - self.mean[:, None]) ** 2 @ self.w)
        self.ratio = self.sd / self.mean
        assert np.all(np.diff(self.ratio) > 0)

    def estimate(self, selected, threshold):
        k = len(selected)
        if k < 4:
            return (np.nan, {"failure": "fewer_than_four_selected"})
        excess = selected.mean() - threshold
        ratio = selected.std(ddof=1) / excess if excess > 0 else np.nan
        record = {"conditional_ratio": ratio}
        if not self.ratio[0] < ratio < self.ratio[-1]:
            return (
                np.nan,
                {**record, "failure": "conditional_moment_ratio_outside_lookup"},
            )
        a = np.interp(ratio, self.ratio, self.a)
        scale = excess / np.interp(a, self.a, self.mean)
        loc = threshold - a * scale
        fa = np.exp(-np.exp(-a))
        prob = fa + (1 - fa) * self.p ** (1 / k)
        pred = np.sum(
            (loc - scale * np.log(-np.log(np.minimum(prob, np.nextafter(1.0, 0.0)))))
            * self.w
        )
        return (
            float(pred),
            {
                **record,
                "failure": "none",
                "conditional_a": a,
                "conditional_scale": scale,
                "conditional_location": loc,
                "conditional_lookup_near_boundary": abs(a) >= 11.99,
            },
        )


def selected_population(values, selector):
    values = np.asarray(values, dtype=np.float64)
    x = values[np.isfinite(values)]
    if len(x) < 6:
        return (
            x[:0],
            np.nan,
            {"failure": "fewer_than_six_finite_grains", "n_finite": len(x)},
        )
    mean, sd = (float(x.mean()), float(x.std(ddof=1)))
    literal = x > mean + sd
    q = min(max(1 - float(literal.mean()) if sd > 0 else 1.0, 0.0), 0.99)
    if selector == "original_recut":
        threshold = float(np.quantile(x, q, method="linear"))
    elif selector.startswith("literal_"):
        coefficient = {"literal_c1": 1.0, "literal_c0p5": 0.5, "literal_c1p5": 1.5}[
            selector
        ]
        threshold = mean + coefficient * sd
    else:
        threshold = float(
            np.quantile(
                x, {"quantile075": 0.75, "quantile090": 0.9}[selector], method="linear"
            )
        )
    mask = x > threshold
    selected = x[mask]
    return (
        selected,
        threshold,
        {
            "failure": "none",
            "n_grains": len(values),
            "n_finite": len(x),
            "negative_predicted_grains": int((x < 0).sum()),
            "full_mean": mean,
            "full_sd_ddof1": sd,
            "literal_c1_threshold": mean + sd,
            "literal_c1_count": int(literal.sum()),
            "fitted_quantile": q,
            "threshold": threshold,
            "selected_count": len(selected),
            "selected_mean": float(selected.mean()) if len(selected) else np.nan,
            "selected_sd_ddof1": float(selected.std(ddof=1))
            if len(selected) > 1
            else np.nan,
            "same_membership_as_literal_c1": bool(np.array_equal(mask, literal)),
        },
    )


def untruncated(selected, geometry_count, n_finite):
    if len(selected) < 4:
        return (np.nan, "fewer_than_four_selected")
    sd = float(selected.std(ddof=1))
    if not sd > 0:
        return (np.nan, "nonpositive_selected_sd")
    n_eff = geometry_count * (len(selected) / n_finite)
    if n_eff < 2:
        return (np.nan, "effective_count_below_two")
    scale = sd * C
    loc = float(selected.mean()) - GAMMA * scale
    return (loc + scale * (math.log(n_eff) + GAMMA), "none")


def make_protocol():
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Two five-fold repeats, 940 unique SVEs each; neural checkpoints are frozen during evaluation",
        "cases": CASES,
        "size_um": SIZES,
        "selectors": SELECTORS,
        "scalar_estimators": [
            "maximum of raw predicted grain peaks",
            "historical untruncated Gumbel expected-maximum map at six fixed selectors",
            "conditional-Gumbel moment inversion and finite-count expected maximum at original recut",
            "exponential exceedance expected maximum u + mean_excess H_k at original recut",
        ],
        "negative_predictions": "Retain raw signed predictions; no abs, clipping, or positive-only filtering",
        "failure_policy": "Original routes retain NaN and disclose valid counts. Separately named operational variants substitute the identical fold fitting-size mean only on invalid rows.",
        "comparison": "All saved direct/tree/size-mean baselines use identical repeat/fold/SVE/case keys. No oracle moments enter predictive estimators.",
        "conditional_map": "256-node Gauss-Legendre quadrature; 2401 standardized thresholds [-12,12]. Strict out-of-domain ratio guard. Threshold estimated from predicted sample; conditional iid model remains a working approximation.",
        "exponential_map": "Exploratory fixed alternative, exponential excess tail with mean estimated from the same selected predicted peaks. Conditional finite count k uses exact harmonic number H_k. Not a validated distribution claim.",
        "upper_tail": "Observed outer maximum exceeds the fitting-only per-size/case 90th percentile. Descriptive subgroup, not threshold tuning.",
        "bootstrap": "2000 paired resamples, seed 20261115. Average two squared losses within SVE; use SVEs valid in both repeats, stratify pooled resamples by physical size. Fixed-prediction evaluation uncertainty, not full training/model-selection uncertainty.",
        "selection_scope": "Recipe was developed using this data pool. No tuning or selection on the newly computed outer scores.",
        "script": digest(Path(__file__)),
    }


def build_predictions():
    original, original_path = load_original()
    if original is None:
        raise RuntimeError(
            "Historical source unavailable: exact parity check is required"
        )
    conditional = ConditionalGumbel()
    rows, ledger, checks, sources = ([], [], [], [digest(original_path)])
    parity_max_abs = 0.0
    cases_seen = []
    for repeat in range(2):
        for fold in range(5):
            path = DATA / f"grain_predictions_repeat{repeat}_fold{fold}.npz"
            with np.load(path, allow_pickle=False) as archive:
                z = {name: archive[name] for name in archive.files}
            assert list(z["cases"]) == CASES
            sources.append(digest(path))
            ids, sizes, counts, offsets = (
                z[k] for k in ["dataset_idx", "size_um", "grain_count", "offsets"]
            )
            assert len(set(ids)) == len(ids)
            assert np.all(np.isfinite(z["pred_peak"]))
            assert z["pred_peak"].shape == z["true_peak"].shape
            assert len(offsets) == len(ids) + 1 and offsets[-1] == len(z["pred_peak"])
            fold_rows = 0
            for j, (idx, size, n) in enumerate(zip(ids, sizes, counts)):
                sl = slice(offsets[j], offsets[j + 1])
                assert offsets[j + 1] - offsets[j] == n
                assert np.all(z["grain_dataset_idx"][sl] == idx)
                assert np.array_equal(z["grain_id"][sl], np.arange(n))
                assert np.all(z["grain_size_um"][sl] == size)
                for c, case in enumerate(CASES):
                    x, t = (z["pred_peak"][sl, c], z["true_peak"][sl, c])
                    truth = float(z["y_true"][j, c])
                    assert np.isclose(t.max(), truth, rtol=2e-06, atol=1e-12)
                    key = dict(
                        repeat=repeat,
                        fold=fold,
                        dataset_idx=int(idx),
                        size_um=int(size),
                        case=case,
                    )
                    base = {
                        **key,
                        "truth": truth,
                        "units": "dimensionless" if case.endswith("fs") else "MPa",
                    }
                    rows.append(
                        {
                            **base,
                            "model": "direct_current_encoder",
                            "prediction": float(z["direct_prediction"][j, c]),
                            "failure": "none",
                        }
                    )
                    rows.append(
                        {
                            **base,
                            "model": "maximum_predicted_grain",
                            "prediction": float(x.max()),
                            "failure": "none",
                        }
                    )
                    for selector in SELECTORS:
                        sel, threshold, record = selected_population(x, selector)
                        prediction, failure = untruncated(sel, int(n), len(x))
                        name = "evt_" + selector
                        rows.append(
                            {
                                **base,
                                "model": name,
                                "prediction": prediction,
                                "failure": failure,
                            }
                        )
                        ledger.append(
                            {
                                **key,
                                "selector": selector,
                                **record,
                                "map_failure": failure,
                            }
                        )
                        if selector == "original_recut":
                            old = original.evt_mixture_fitted_q(
                                x, int(n), method="plastic_frac"
                            )["M_S_pred"]
                            assert np.isfinite(old) == np.isfinite(prediction)
                            if np.isfinite(old):
                                assert np.isclose(
                                    old, prediction, rtol=1e-13, atol=1e-15
                                )
                                parity_max_abs = max(
                                    parity_max_abs, abs(old - prediction)
                                )
                            value, cond_record = conditional.estimate(sel, threshold)
                            rows.append(
                                {
                                    **base,
                                    "model": "conditional_gumbel_original_recut",
                                    "prediction": value,
                                    "failure": cond_record["failure"],
                                }
                            )
                            ledger.append(
                                {
                                    **key,
                                    "selector": "conditional_gumbel_original_recut",
                                    **record,
                                    **cond_record,
                                }
                            )
                            ex = float(sel.mean() - threshold) if len(sel) else np.nan
                            exp_value = (
                                float(threshold + ex * (digamma(len(sel) + 1) + GAMMA))
                                if len(sel) >= 4 and ex > 0
                                else np.nan
                            )
                            rows.append(
                                {
                                    **base,
                                    "model": "exponential_excess_original_recut",
                                    "prediction": exp_value,
                                    "failure": "none"
                                    if np.isfinite(exp_value)
                                    else "fewer_than_four_or_nonpositive_excess",
                                }
                            )
                    fold_rows += 1
                cases_seen.append((repeat, int(idx)))
            checks.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_sves": len(ids),
                    "n_grains": int(counts.sum()),
                    "n_sve_case": fold_rows,
                }
            )
            print(f"mapped repeat={repeat} fold={fold} SVEs={len(ids)}", flush=True)
    assert len(cases_seen) == 1880 and len(set(cases_seen)) == 1880
    assert len({idx for _, idx in cases_seen}) == 940
    pred = pd.DataFrame(rows)
    reference = pd.read_csv(BENCH / "all_outer_predictions.csv")
    extra_baselines = STUDY / "baselines/size_baseline_predictions.csv"
    if extra_baselines.exists():
        reference = pd.concat(
            [reference, pd.read_csv(extra_baselines)], ignore_index=True
        )
        sources.append(digest(extra_baselines))
    calibration_path = STUDY / "baselines/graph_calibration_predictions.csv"
    if calibration_path.exists():
        reference = pd.concat(
            [reference, pd.read_csv(calibration_path)], ignore_index=True
        )
        sources.append(digest(calibration_path))
    ref = reference[reference.model == "Fitting-size mean"][
        KEYS + ["truth", "prediction"]
    ].rename(columns={"prediction": "size_mean_prediction", "truth": "baseline_truth"})
    assert len(ref) == 7520 and (not ref.duplicated(KEYS).any())
    pred = pred.merge(ref, on=KEYS, how="left", validate="many_to_one")
    assert np.all(np.isfinite(pred.size_mean_prediction))
    assert np.allclose(pred.truth, pred.baseline_truth, rtol=2e-06, atol=1e-12)
    pred["used_fallback"] = False
    operational = pred[
        ~pred.model.isin(["maximum_predicted_grain", "direct_current_encoder"])
    ].copy()
    operational["used_fallback"] = ~np.isfinite(operational.prediction)
    operational["prediction"] = operational.prediction.where(
        ~operational.used_fallback, operational.size_mean_prediction
    )
    operational["model"] += "__size_mean_fallback"
    reference["failure"] = "none"
    reference["used_fallback"] = False
    calibration_ledger = STUDY / "baselines/graph_calibration_ledger.csv"
    if calibration_ledger.exists():
        cal = pd.read_csv(calibration_ledger)
        join_keys = ["model", "repeat", "fold", "dataset_idx", "case"]
        reference = reference.merge(
            cal[join_keys + ["fallback"]],
            on=join_keys,
            how="left",
            validate="one_to_one",
        )
        reference["used_fallback"] = reference.fallback.fillna(False).astype(bool)
        reference = reference.drop(columns="fallback")
    reference = reference.merge(
        ref.drop(columns="baseline_truth"), on=KEYS, validate="many_to_one"
    )
    reference["model"] = reference.model.replace(
        {"baseline": "direct_graph_head", "Fitting-size mean": "size_mean"}
    )
    pred = pd.concat(
        [
            pred.drop(columns="baseline_truth"),
            operational.drop(columns="baseline_truth"),
            reference,
        ],
        ignore_index=True,
    )
    assert not pred.duplicated(KEYS + ["model"]).any()
    thresholds = []
    for repeat in range(2):
        for fold in range(5):
            fitting = pd.read_csv(
                BENCH
                / f"gnn/repeat{repeat}_fold{fold}/baseline/fitting_predictions.csv"
            )
            for (size, case), g in fitting.groupby(["size_um", "case"]):
                thresholds.append(
                    dict(
                        repeat=repeat,
                        fold=fold,
                        size_um=size,
                        case=case,
                        training_q90=float(g.truth.quantile(0.9)),
                    )
                )
    pred = pred.merge(
        pd.DataFrame(thresholds),
        on=["repeat", "fold", "size_um", "case"],
        validate="many_to_one",
    )
    pred["upper_tail_training_q90"] = pred.truth > pred.training_q90
    pred.to_csv(HERE / "predictions_long.csv", index=False, float_format="%.17g")
    pd.DataFrame(ledger).to_csv(
        HERE / "tail_parameter_ledger.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(thresholds).to_csv(
        HERE / "training_upper_tail_thresholds.csv", index=False, float_format="%.17g"
    )
    json_write(
        HERE / "input_validation.json",
        {
            "status": "pass",
            "n_unique_sves": 940,
            "n_repeats": 2,
            "parity_comparisons": 7520,
            "original_max_abs_difference": parity_max_abs,
            "all_grain_predictions_finite": True,
            "all_grain_truth_maxima_match": True,
            "folds": checks,
            "sources": sources,
        },
    )
    return pred


def metrics(y, p, baseline):
    y, p, baseline = (np.asarray(y), np.asarray(p), np.asarray(baseline))
    n = len(y)
    if n < 2:
        return {
            k: np.nan
            for k in [
                "r2",
                "mae",
                "rmse",
                "bias",
                "spread_ratio",
                "spearman",
                "skill_vs_size_mean",
                "rmse_change_pct",
            ]
        }
    e = p - y
    den = np.sum((y - y.mean()) ** 2)
    base_mse = np.mean((baseline - y) ** 2)
    mse = np.mean(e * e)
    return {
        "r2": 1 - np.sum(e * e) / den if den > 0 else np.nan,
        "mae": np.mean(abs(e)),
        "rmse": np.sqrt(mse),
        "bias": np.mean(e),
        "spread_ratio": np.std(p) / np.std(y) if np.std(y) > 0 else np.nan,
        "spearman": spearmanr(y, p).statistic if np.ptp(p) > 0 else np.nan,
        "skill_vs_size_mean": 1 - mse / base_mse if base_mse > 0 else np.nan,
        "rmse_change_pct": 100 * (np.sqrt(mse / base_mse) - 1)
        if base_mse > 0
        else np.nan,
    }


def refresh_external_calibrations(pred):
    """Append available, independently specified calibration controls by model name."""
    existing = set(pred.model)
    base = pred[pred.model == "size_mean"][
        KEYS + ["size_mean_prediction", "training_q90", "upper_tail_training_q90"]
    ]
    for path in sorted((STUDY / "baselines").glob("*_calibration_predictions.csv")):
        external = pd.read_csv(path)
        external = external[~external.model.isin(existing)].copy()
        if external.empty:
            continue
        assert not external.duplicated(KEYS + ["model"]).any()
        external = external.merge(base, on=KEYS, validate="many_to_one")
        external["failure"] = "none"
        external["used_fallback"] = False
        ledger = path.with_name(path.name.replace("_predictions.csv", "_ledger.csv"))
        if ledger.exists():
            lk = ["model", "repeat", "fold", "dataset_idx", "case"]
            external = external.merge(
                pd.read_csv(ledger)[lk + ["fallback"]],
                on=lk,
                how="left",
                validate="one_to_one",
            )
            external["used_fallback"] = external.fallback.fillna(False).astype(bool)
            external = external.drop(columns="fallback")
        assert np.isfinite(external.prediction).all()
        pred = pd.concat([pred, external], ignore_index=True)
        existing.update(external.model.unique())
    assert not pred.duplicated(KEYS + ["model"]).any()
    return pred


def refresh_external_studies(pred):
    """Ingest explicitly completed follow-up estimators, retaining their failure flags."""
    source = STUDY / "conditional_mle/predictions.csv"
    marker = STUDY / "conditional_mle/completion.json"
    if not source.exists() or not marker.exists():
        return pred
    if json.loads(marker.read_text()).get("status") != "complete":
        return pred
    external = pd.read_csv(source)
    external = external[~external.model.isin(set(pred.model))].copy()
    if external.empty:
        return pred
    assert not external.duplicated(KEYS + ["model"]).any()
    assert all((n == 7520 for n in external.groupby("model").size()))
    base = pred[pred.model == "size_mean"][
        KEYS
        + ["truth", "size_mean_prediction", "training_q90", "upper_tail_training_q90"]
    ]
    external = external.merge(
        base.rename(columns={"truth": "reference_truth"}),
        on=KEYS,
        validate="many_to_one",
    )
    assert np.allclose(external.truth, external.reference_truth, rtol=2e-06, atol=1e-12)
    external = external.drop(columns="reference_truth")
    if "failure" not in external:
        external["failure"] = np.where(
            np.isfinite(external.prediction), "none", "undefined_external_estimator"
        )
    if "used_fallback" not in external:
        external["used_fallback"] = False
    pred = pd.concat([pred, external], ignore_index=True)
    assert not pred.duplicated(KEYS + ["model"]).any()
    return pred


def refresh_outer_training_baselines(pred):
    """Keep equal-label-budget size/count controls distinct from historical controls."""
    source = STUDY / "baselines/outer_training_size_baseline_predictions.csv"
    if not source.exists():
        return pred
    external = pd.read_csv(source)
    external = external[~external.model.isin(set(pred.model))].copy()
    if external.empty:
        return pred
    assert len(external) % 7520 == 0
    assert all((n == 7520 for n in external.groupby("model").size()))
    base = pred[pred.model == "size_mean"][
        KEYS
        + ["truth", "size_mean_prediction", "training_q90", "upper_tail_training_q90"]
    ]
    external = external.merge(
        base.rename(columns={"truth": "reference_truth"}),
        on=KEYS,
        validate="many_to_one",
    )
    assert np.allclose(external.truth, external.reference_truth, rtol=2e-06, atol=1e-12)
    external = external.drop(columns="reference_truth")
    external["failure"] = "none"
    external["used_fallback"] = False
    pred = pd.concat([pred, external], ignore_index=True)
    assert not pred.duplicated(KEYS + ["model"]).any()
    return pred


def score_predictions(pred):
    scores = []
    for (model, repeat, case), g in pred.groupby(
        ["model", "repeat", "case"], sort=True
    ):
        for size in ["pooled"] + SIZES:
            cell = g if size == "pooled" else g[g.size_um == size]
            for subgroup in ["all", "above_training_q90"]:
                h = cell if subgroup == "all" else cell[cell.upper_tail_training_q90]
                finite = np.isfinite(h.prediction.to_numpy())
                for scope in ["complete_cohort", "valid_subset"]:
                    valid = h[finite]
                    ok = len(valid) == len(h) or scope == "valid_subset"
                    result = (
                        metrics(
                            valid.truth, valid.prediction, valid.size_mean_prediction
                        )
                        if ok
                        else metrics([], [], [])
                    )
                    scores.append(
                        dict(
                            model=model,
                            repeat=repeat,
                            case=case,
                            size_um=size,
                            subgroup=subgroup,
                            scope=scope,
                            n_expected=len(h),
                            n_valid=len(valid),
                            n_fallback=int(h.used_fallback.sum()),
                            availability=len(valid) / len(h) if len(h) else np.nan,
                            **result,
                        )
                    )
    frame = pd.DataFrame(scores)
    frame.to_csv(HERE / "metrics_by_repeat.csv", index=False, float_format="%.17g")
    keys = ["model", "case", "size_um", "subgroup", "scope"]
    measure = [
        "r2",
        "mae",
        "rmse",
        "bias",
        "spread_ratio",
        "spearman",
        "skill_vs_size_mean",
        "rmse_change_pct",
        "n_expected",
        "n_valid",
        "n_fallback",
        "availability",
    ]
    summary = frame.groupby(keys, dropna=False)[measure].agg(
        ["mean", "min", "max", "count"]
    )
    summary.columns = [f"{x}_{stat}" for x, stat in summary.columns]
    summary = summary.reset_index()
    incomplete = (summary.scope == "complete_cohort") & (summary.r2_count < 2)
    for field in [
        "r2",
        "mae",
        "rmse",
        "bias",
        "spread_ratio",
        "spearman",
        "skill_vs_size_mean",
        "rmse_change_pct",
    ]:
        summary.loc[
            incomplete, [field + "_" + stat for stat in ["mean", "min", "max"]]
        ] = np.nan
    summary.to_csv(HERE / "metrics_summary.csv", index=False, float_format="%.17g")
    return (frame, summary)


def bootstrap_comparisons(
    pred,
    n_boot=2000,
    filename="paired_sve_bootstrap_vs_size_mean.csv",
    reference_name="size_mean",
):
    """Paired errors averaged over repeats, then SVE-cluster resampling."""
    rng = np.random.default_rng(20261115)
    outputs = []
    for (model, case), group in pred.groupby(["model", "case"], sort=True):
        if model == "size_mean":
            continue
        for size in ["pooled"] + SIZES:
            g = group if size == "pooled" else group[group.size_um == size]
            g = g.copy()
            g["loss"] = (g.prediction - g.truth) ** 2
            g["base_loss"] = (g.size_mean_prediction - g.truth) ** 2
            g["finite"] = np.isfinite(g.prediction)
            agg = g.groupby(["dataset_idx", "size_um"]).agg(
                loss=("loss", "mean"),
                base_loss=("base_loss", "mean"),
                n_repeat=("repeat", "nunique"),
                n_finite=("finite", "sum"),
            )
            assert np.all(agg.n_repeat == 2)
            expected = len(agg)
            agg = agg[agg.n_finite == 2].reset_index()
            if len(agg) < 3:
                continue
            model_boot = np.zeros(n_boot)
            baseline_boot = np.zeros(n_boot)
            for _, stratum in agg.groupby("size_um"):
                n = len(stratum)
                draw = rng.integers(0, n, size=(n_boot, n))
                model_boot += stratum.loss.to_numpy()[draw].sum(axis=1)
                baseline_boot += stratum.base_loss.to_numpy()[draw].sum(axis=1)
            model_boot /= len(agg)
            baseline_boot /= len(agg)
            skill_boot = 1 - model_boot / baseline_boot
            delta_boot = np.sqrt(model_boot) - np.sqrt(baseline_boot)
            pct_boot = 100 * (np.sqrt(model_boot / baseline_boot) - 1)
            ml, bl = (agg.loss.mean(), agg.base_loss.mean())
            out = dict(
                model=model,
                case=case,
                size_um=size,
                n_expected_unique_sves=expected,
                n_both_repeats_valid=len(agg),
                mask="both_repeats_valid",
                n_boot=n_boot,
                reference_model=reference_name,
                interval_scope="pointwise exploratory conditional fixed-prediction interval",
                skill=1 - ml / bl,
                delta_rmse=np.sqrt(ml) - np.sqrt(bl),
                rmse_change_pct=100 * (np.sqrt(ml / bl) - 1),
            )
            for name, values in [
                ("skill", skill_boot),
                ("delta_rmse", delta_boot),
                ("rmse_change_pct", pct_boot),
            ]:
                out[name + "_ci_low"], out[name + "_ci_high"] = np.quantile(
                    values, [0.025, 0.975]
                )
            outputs.append(out)
    result = pd.DataFrame(outputs)
    if filename:
        result.to_csv(HERE / filename, index=False, float_format="%.17g")
    return result


def bootstrap_original_vs_controls(pred, n_boot=2000):
    chosen = ["evt_original_recut", "evt_original_recut__size_mean_fallback"]
    controls = [
        "Log count linear",
        "Log count quadratic",
        "Within-size count ridge",
        "Random forest",
        "Histogram boosting",
        "Outer-training size mean",
        "Outer-training Log count linear",
        "Outer-training Log count quadratic",
        "Outer-training Within-size count ridge",
        "Graph Affine maximum calibration",
        "Graph Affine bridge calibration",
        "Graph Affine direct calibration",
        "Graph Tail-moment ridge",
        "Graph Size-residual tail ridge",
        "direct_graph_head",
        "direct_current_encoder",
    ]
    actual = set(pred.model)
    controls = [
        name
        if name in actual
        else next((v for v in actual if v.startswith(name)), name)
        for name in controls
    ]
    result = []
    for control in controls:
        if control not in actual:
            continue
        base = pred[pred.model == control][KEYS + ["prediction"]].rename(
            columns={"prediction": "reference_prediction"}
        )
        pair = pred[pred.model.isin(chosen)].merge(
            base, on=KEYS, validate="many_to_one"
        )
        assert np.isfinite(pair.reference_prediction).all()
        pair["size_mean_prediction"] = pair.reference_prediction
        result.append(
            bootstrap_comparisons(pair, n_boot, filename=None, reference_name=control)
        )
    combined = pd.concat(result, ignore_index=True)
    combined.to_csv(
        HERE / "original_bridge_paired_comparisons.csv",
        index=False,
        float_format="%.17g",
    )
    return combined


def plot_and_report(summary, intervals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [
        "size_mean",
        "direct_graph_head",
        "maximum_predicted_grain",
        "evt_original_recut__size_mean_fallback",
        "conditional_gumbel_original_recut__size_mean_fallback",
        "exponential_excess_original_recut__size_mean_fallback",
    ]
    labels = [
        "Size mean",
        "Direct head",
        "Maximum predicted grain",
        "Original bridge + fallback",
        "Conditional bridge + fallback",
        "Exponential excess + fallback",
    ]
    cols = ["#777777", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for case, ax in zip(["hcf_fs", "hcf_wp", "lcf_fs", "lcf_wp"], axes.flat):
        for model, label, color in zip(primary, labels, cols):
            g = summary[
                (summary.model == model)
                & (summary.case == case)
                & (summary.scope == "complete_cohort")
                & (summary.subgroup == "all")
                & (summary.size_um.astype(str) != "pooled")
            ].copy()
            g["size_um"] = g.size_um.astype(int)
            g = g.sort_values("size_um")
            ax.plot(g.size_um, g.r2_mean, "o-", label=label, color=color, markersize=4)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(case.upper().replace("_", "/"))
        ax.set_xlabel("SVE edge length (micrometers)")
        ax.set_ylabel("Within-size maximum R squared")
        ax.set_xticks(SIZES)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(HERE / "within_size_r2.png", dpi=180)
    fig.savefig(HERE / "within_size_r2.pdf")
    plt.close(fig)
    detail_models = [primary[i] for i in [0, 1, 3, 4]]
    detail_labels = [labels[i] for i in [0, 1, 3, 4]]
    detail_colors = [cols[i] for i in [0, 1, 3, 4]]
    if "Graph Affine bridge calibration" in set(summary.model):
        detail_models.append("Graph Affine bridge calibration")
        detail_labels.append("Affine bridge calibration")
        detail_colors.append("#7B3294")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for case, ax in zip(["hcf_fs", "hcf_wp"], axes):
        for model, label, color in zip(detail_models, detail_labels, detail_colors):
            g = summary[
                (summary.model == model)
                & (summary.case == case)
                & (summary.scope == "complete_cohort")
                & (summary.subgroup == "all")
                & (summary.size_um.astype(str) != "pooled")
            ].copy()
            g["size_um"] = g.size_um.astype(int)
            g = g.sort_values("size_um")
            ax.plot(g.size_um, g.r2_mean, "o-", label=label, color=color, markersize=4)
            if model in [
                "evt_original_recut__size_mean_fallback",
                "Graph Affine bridge calibration",
            ]:
                ax.fill_between(
                    g.size_um.to_numpy(),
                    g.r2_min.to_numpy(),
                    g.r2_max.to_numpy(),
                    color=color,
                    alpha=0.13,
                )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(case.upper().replace("_", "/"))
        ax.set_xlabel("SVE edge length (micrometers)")
        ax.set_ylabel("Within-size maximum R squared")
        ax.set_xticks(SIZES)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Mean of two complete out-of-fold repeats; bands show their range", fontsize=10
    )
    fig.savefig(HERE / "hcf_primary_detail.png", dpi=180)
    fig.savefig(HERE / "hcf_primary_detail.pdf")
    plt.close(fig)
    keep = summary[
        summary.model.isin(primary)
        & (summary.scope == "complete_cohort")
        & (summary.subgroup == "all")
    ]
    keep.to_csv(HERE / "primary_comparison.csv", index=False, float_format="%.17g")
    compact = keep[
        [
            "model",
            "case",
            "size_um",
            "r2_mean",
            "rmse_mean",
            "skill_vs_size_mean_mean",
            "n_fallback_mean",
        ]
    ]
    lines = [
        "940 SVE frozen checkpoint bridge evaluation",
        "",
        "All results are mean scores over two complete five-fold repeats unless noted.",
        "Each repeat contains 940 unique SVEs. The repeat predictions are not independent validation datasets.",
        "Neural checkpoints remain frozen and threshold variants are fixed. Secondary comparisons include stopping-trained calibrators and per-SVE distribution fits to predicted peaks. No outer-label fitting or tuning occurs.",
        "The __size_mean_fallback suffix identifies an operational composite estimator, distinct from the historical bridge.",
        "Original route metrics are retained separately, including every failure and conditional-subset denominator.",
        "Bootstrap intervals average repeated squared losses within SVE and resample SVE identities within size.",
        "They are conditional on the fixed fitted models and do not capture full model-selection/training uncertainty.",
        "",
        "Primary pooled results",
        compact[compact.size_um.astype(str) == "pooled"].to_string(index=False),
        "",
        "Primary within-size results",
        compact[compact.size_um.astype(str) != "pooled"].to_string(index=False),
        "",
        "Files",
        "predictions_long.csv: every original/composite/baseline SVE prediction and validity/fallback flag",
        "tail_parameter_ledger.csv: raw-signed predicted grain selection and conditional fit diagnostics",
        "metrics_by_repeat.csv and metrics_summary.csv: complete and valid-subset scores, including upper tail",
        "paired_sve_bootstrap_vs_size_mean.csv: paired loss/skill and conditional bootstrap intervals",
        "input_validation.json: historical parity and population checks",
        "protocol.json: prespecified estimators and information-access limits",
    ]
    (HERE / "RESULTS.txt").write_text("\n".join(lines) + "\n")


def main():
    global HERE, DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-subdir")
    args = parser.parse_args()
    if args.data:
        DATA = args.data.resolve()
    if args.output_subdir:
        base = HERE
        HERE = (base / args.output_subdir).resolve()
        assert HERE.is_relative_to(
            base
        ), "Output must remain within the bridge-owned directory"
    HERE.mkdir(parents=True, exist_ok=True)
    if not (HERE / "protocol.json").exists():
        json_write(HERE / "protocol.json", make_protocol())
    pred = (
        pd.read_csv(HERE / "predictions_long.csv", low_memory=False)
        if args.reuse_predictions
        else build_predictions()
    )
    pred = refresh_external_calibrations(pred)
    pred = refresh_external_studies(pred)
    pred = refresh_outer_training_baselines(pred)
    pred.to_csv(HERE / "predictions_long.csv", index=False, float_format="%.17g")
    _, summary = score_predictions(pred)
    intervals = bootstrap_comparisons(pred, args.bootstrap)
    bootstrap_original_vs_controls(pred, args.bootstrap)
    plot_and_report(summary, intervals)
    json_write(
        HERE / "execution_manifest.json",
        {
            "script": digest(Path(__file__)),
            "input_directory": str(DATA),
            "output_directory": str(HERE),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "external_calibration_sources": [
                digest(p)
                for p in sorted(
                    (STUDY / "baselines").glob("*_calibration_predictions.csv")
                )
            ],
            "reference_calibrations": "Secondary controls fitted on original stopping partitions; reused for checkpoint selection, not an independent calibration cohort.",
            "intervals": "Pointwise exploratory comparisons without multiplicity correction; not confirmatory tests.",
        },
    )
    print("COMPLETE", HERE, flush=True)


if __name__ == "__main__":
    main()
