"""Prespecified aluminum-informed frozen-checkpoint steel probe. See protocol.json."""
from __future__ import annotations
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import hashlib
import json
import platform
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import gumbel_r

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
KEYS = ["repeat", "fold", "dataset_idx", "size_um", "case"]
MODELS = ["block16_raw", "block16_affine", "bridge_logcount_affine"]


def digest(p):
    return {
        "path": str(p.relative_to(ROOT)),
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize_predictions(data, repeat, fold, subset):
    """Only prediction/geometric arrays accepted; no observed-response arrays."""
    assert set(data) == {
        "dataset_idx",
        "size_um",
        "grain_count",
        "offsets",
        "pred_peak",
    }
    rows, partitions = [], []
    for i, idx in enumerate(data["dataset_idx"]):
        n = int(data["grain_count"][i])
        size = int(data["size_um"][i])
        grain = data["pred_peak"][data["offsets"][i] : data["offsets"][i + 1]]
        assert len(grain) == n
        for c, case in enumerate(CASES):
            key = dict(
                repeat=repeat,
                fold=fold,
                subset=subset,
                dataset_idx=int(idx),
                size_um=size,
                case=case,
            )
            x = np.sort(grain[:, c])
            b, k = np.nan, 0
            if len(x) >= 6 and np.isfinite(x).all():
                q = np.clip(1 - np.mean(x > x.mean() + x.std(ddof=1)), 0, 0.99)
                tail = x[x > np.quantile(x, q, method="linear")]
                k = len(tail)
                if k >= 4 and tail.std(ddof=1) > 0:
                    b = float(
                        tail.mean() + np.sqrt(6) / np.pi * tail.std(ddof=1) * np.log(k)
                    )
            values = []
            m = n // 16
            for p in range(5):
                rec = dict(
                    **key,
                    partition=p,
                    n_grains=n,
                    block_count=16,
                    block_size=m,
                    remainder=n - 16 * m,
                    valid=False,
                    prediction=np.nan,
                    location=np.nan,
                    scale=np.nan,
                    warning="",
                    failure="none",
                )
                if n < 16 or not np.isfinite(x).all():
                    rec["failure"] = "insufficient_or_nonfinite_grains"
                else:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([101, size, int(idx), p])
                    )
                    maxima = x[rng.permutation(n)[: 16 * m]].reshape(16, m).max(axis=1)
                    shift, unit = float(maxima.mean()), float(maxima.std(ddof=1))
                    if not unit > 0:
                        rec["failure"] = "nonpositive_group_sd"
                    else:
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                loc0, scale0 = gumbel_r.fit((maxima - shift) / unit)
                            loc, scale = shift + unit * loc0, unit * scale0
                            pred = float(loc + scale * (np.log(n / m) + np.euler_gamma))
                            good = bool(np.isfinite(pred) and scale > 0)
                            rec.update(
                                valid=good,
                                prediction=pred,
                                location=float(loc),
                                scale=float(scale),
                                warning=" | ".join(str(w.message) for w in caught),
                                failure="none" if good else "invalid_fit",
                            )
                        except Exception as exc:
                            rec["failure"] = repr(exc)
                partitions.append(rec)
                values.append(rec["prediction"] if rec["valid"] else np.nan)
            raw = float(np.mean(values)) if np.isfinite(values).all() else np.nan
            rows.append(
                dict(
                    **key,
                    n_grains=n,
                    selected_count=k,
                    bridge=b,
                    block16_raw=raw,
                    log_count=float(np.log(n)),
                )
            )
    return pd.DataFrame(rows), partitions


def calibrate(stop, test, labels, fields, fallback):
    a, b = stop[fields].to_numpy(), test[fields].to_numpy()
    okfit, oktest = np.isfinite(a).all(1), np.isfinite(b).all(1)
    prediction = fallback.copy()
    if okfit.sum() < 10:
        return (
            prediction,
            np.ones(len(b), bool),
            dict(status="insufficient_stopping_points"),
        )
    mean, sd = a[okfit].mean(0), a[okfit].std(0)
    sd[sd == 0] = 1
    design = np.column_stack([np.ones(okfit.sum()), (a[okfit] - mean) / sd])
    coeff, _, rank, _ = np.linalg.lstsq(design, labels[okfit], rcond=None)
    if rank != design.shape[1]:
        return (
            prediction,
            np.ones(len(b), bool),
            dict(status="rank_deficient_stopping_design"),
        )
    prediction[oktest] = (
        np.column_stack([np.ones(oktest.sum()), (b[oktest] - mean) / sd]) @ coeff
    )
    assert np.isfinite(prediction).all()
    return (
        prediction,
        ~oktest,
        dict(
            status="fitted",
            fields=fields,
            feature_mean=mean.tolist(),
            feature_sd=sd.tolist(),
            standardized_coefficients=coeff.tolist(),
            raw_intercept=float(coeff[0] - np.sum(coeff[1:] * mean / sd)),
            raw_coefficients=(coeff[1:] / sd).tolist(),
            n_stopping=int(okfit.sum()),
            stopping_ids=stop.dataset_idx.to_numpy()[okfit].tolist(),
        ),
    )


def metrics(frame):
    y, p = frame.truth.to_numpy(), frame.prediction.to_numpy()
    err = p - y
    return dict(
        n_predictions=len(y),
        n_sves=frame.dataset_idx.nunique(),
        r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(abs(err))),
        bias=float(err.mean()),
        prediction_sd=float(p.std(ddof=1)),
        truth_sd=float(y.std(ddof=1)),
    )


def main():
    started = datetime.now(timezone.utc).isoformat()
    sources = [
        digest(HERE / "protocol.json"),
        digest(BASE / "partitions.json"),
        digest(BASE / "regressors/raw_features940.npz"),
    ]
    parts = json.loads((BASE / "partitions.json").read_text())
    with np.load(BASE / "regressors/raw_features940.npz", allow_pickle=False) as z:
        all_y, all_b = z["raw_truth"], z["size_bucket"]
    outputs, features, partition_rows, fitted, checks = [], [], [], [], []
    for part in parts:
        r, f = part["repeat"], part["fold"]
        stopping, evaluation = {}, {}
        ids, labels, buckets = {}, {}, {}
        for subset, destination, prefix in [
            ("stopping", stopping, "stopping_"),
            ("evaluation", evaluation, ""),
        ]:
            path = STUDY / "data" / f"grain_predictions_{prefix}repeat{r}_fold{f}.npz"
            sources.append(digest(path))
            with np.load(path, allow_pickle=False) as z:
                assert list(z["cases"]) == CASES
                for k in [
                    "dataset_idx",
                    "size_um",
                    "grain_count",
                    "offsets",
                    "pred_peak",
                ]:
                    destination[k] = z[k]
                ids[subset], buckets[subset] = z["dataset_idx"], z["size_bucket"]
                # Labels are kept separate from the response-free estimator inputs.
                labels[subset] = z["y_true"]
            assert set(ids[subset]) == set(part[subset])
        assert not set(ids["evaluation"]) & (
            set(part["fitting"]) | set(ids["stopping"])
        )
        assert not set(part["fitting"]) & set(ids["stopping"])
        xs, ps = summarize_predictions(stopping, r, f, "stopping")
        xt, pt = summarize_predictions(evaluation, r, f, "evaluation")
        features.extend([xs, xt])
        partition_rows.extend(ps + pt)
        train = np.asarray(part["fitting"])
        means = np.stack([all_y[train[all_b[train] == b]].mean(0) for b in range(5)])
        for c, case in enumerate(CASES):
            stop, test = xs[xs.case == case].copy(), xt[xt.case == case].copy()
            assert np.array_equal(stop.dataset_idx, ids["stopping"])
            assert np.array_equal(test.dataset_idx, ids["evaluation"])
            fallback = means[buckets["evaluation"], c]
            for model in MODELS:
                if model == "block16_raw":
                    p = test.block16_raw.to_numpy().copy()
                    use_fallback = ~np.isfinite(p)
                    p[use_fallback] = fallback[use_fallback]
                else:
                    fields = (
                        ["block16_raw"]
                        if model == "block16_affine"
                        else ["bridge", "log_count"]
                    )
                    p, use_fallback, fit = calibrate(
                        stop, test, labels["stopping"][:, c], fields, fallback
                    )
                    fitted.append(dict(model=model, repeat=r, fold=f, case=case, **fit))
                for j, (_, rec) in enumerate(test.iterrows()):
                    outputs.append(
                        {
                            **{k: rec[k] for k in KEYS},
                            "model": model,
                            "truth": float(labels["evaluation"][j, c]),
                            "prediction": float(p[j]),
                            "used_fallback": bool(use_fallback[j]),
                        }
                    )
        checks.append(
            dict(
                repeat=r,
                fold=f,
                fitting=len(train),
                stopping=len(ids["stopping"]),
                evaluation=len(ids["evaluation"]),
                partitions_disjoint=True,
            )
        )
        print(f"Finished repeat {r}, fold {f}", flush=True)
    pred = pd.DataFrame(outputs)
    assert len(pred) == 940 * 2 * 4 * 3
    assert not pred.duplicated(["model", "repeat", "dataset_idx", "case"]).any()
    assert np.isfinite(pred.prediction).all()
    feat = pd.concat(features, ignore_index=True)
    feat.to_csv(HERE / "prediction_only_features.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(HERE / "block_partition_fits.csv", index=False)
    pred.to_csv(HERE / "candidate_predictions.csv", index=False)
    write_json("calibration_fits.json", fitted)
    reference_path = STUDY / "bridge/predictions_long.csv"
    sources.append(digest(reference_path))
    reference = pd.read_csv(reference_path, low_memory=False)
    comparator_names = [
        "evt_original_recut__size_mean_fallback",
        "Graph Affine bridge calibration",
        "direct_current_encoder",
        "size_mean",
        "Outer-training size mean",
        "Outer-training Log count linear",
    ]
    reference = reference[reference.model.isin(comparator_names)][
        KEYS + ["model", "truth", "prediction"]
    ].copy()
    for model, frame in reference.groupby("model"):
        assert len(frame) == 940 * 2 * 4 and np.isfinite(frame.prediction).all()
        match = pred[pred.model == "block16_raw"].merge(
            frame, on=KEYS, suffixes=("_candidate", "_reference"), validate="one_to_one"
        )
        assert len(match) == 940 * 2 * 4
        assert np.allclose(
            match.truth_candidate, match.truth_reference, rtol=1e-12, atol=1e-14
        )
    # Confirm the current scalar was reconstructed exactly, including failure rows.
    raw = pd.read_csv(reference_path, low_memory=False)
    raw = raw[raw.model == "evt_original_recut"][KEYS + ["prediction"]]
    parity = feat[feat.subset == "evaluation"].merge(
        raw, on=KEYS, validate="one_to_one"
    )
    assert np.allclose(
        parity.bridge, parity.prediction, rtol=1e-12, atol=1e-14, equal_nan=True
    )
    allpred = pd.concat([pred, reference], ignore_index=True)
    allpred.to_csv(HERE / "all_comparison_predictions.csv", index=False)
    scores = []
    for (model, case), frame in allpred.groupby(["model", "case"]):
        for repeat in ["both", 0, 1]:
            repeatframe = frame if repeat == "both" else frame[frame.repeat == repeat]
            for size in ["pooled", 125, 177, 250, 354, 500]:
                sub = (
                    repeatframe
                    if size == "pooled"
                    else repeatframe[repeatframe.size_um == size]
                )
                scores.append(
                    dict(
                        model=model,
                        case=case,
                        repeat=repeat,
                        size_um=size,
                        **metrics(sub),
                    )
                )
    score = pd.DataFrame(scores)
    score.to_csv(HERE / "metrics.csv", index=False)
    contrasts = []
    boot = np.random.default_rng(20261115)
    for case in CASES:
        for size in ["pooled", 125, 177, 250, 354, 500]:
            sub = allpred[allpred.case == case].copy()
            if size != "pooled":
                sub = sub[sub.size_um == size]
            sub["loss"] = (sub.prediction - sub.truth) ** 2
            wide = (
                sub.groupby(["dataset_idx", "size_um", "model"])
                .loss.mean()
                .unstack("model")
                .reset_index()
            )
            strata = [
                np.flatnonzero(wide.size_um.to_numpy() == s)
                for s in sorted(wide.size_um.unique())
            ]
            resamples = np.hstack(
                [boot.choice(ix, size=(2000, len(ix)), replace=True) for ix in strata]
            )
            for model in MODELS:
                for ref in [
                    "Graph Affine bridge calibration",
                    "Outer-training size mean",
                    "size_mean",
                    "Outer-training Log count linear",
                    "direct_current_encoder",
                ]:
                    a, b = wide[model].to_numpy(), wide[ref].to_numpy()
                    changes = 100 * (
                        1 - np.sqrt(a[resamples].mean(1) / b[resamples].mean(1))
                    )
                    contrasts.append(
                        dict(
                            case=case,
                            size_um=size,
                            model=model,
                            reference=ref,
                            n_sves=len(wide),
                            rmse_reduction_percent=float(
                                100 * (1 - np.sqrt(a.mean() / b.mean()))
                            ),
                            reduction_ci_low=float(np.quantile(changes, 0.025)),
                            reduction_ci_high=float(np.quantile(changes, 0.975)),
                            mean_squared_error_difference=float(a.mean() - b.mean()),
                        )
                    )
    pd.DataFrame(contrasts).to_csv(HERE / "paired_comparisons.csv", index=False)
    pp = pd.DataFrame(partition_rows)
    write_json(
        "execution_manifest.json",
        dict(
            status="complete",
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            script=digest(Path(__file__)),
            inputs=sources,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pandas=pd.__version__,
            n_candidate_predictions=len(pred),
            n_partition_fits=len(pp),
            n_failed_partition_fits=int((~pp.valid).sum()),
            n_warning_partition_fits=int(pp.warning.ne("").sum()),
            fallback_counts=pred.groupby("model")
            .used_fallback.sum()
            .astype(int)
            .to_dict(),
            current_bridge_parity_max_abs=float(
                np.nanmax(abs(parity.bridge - parity.prediction))
            ),
            partition_checks=checks,
            no_observed_grain_values_read=True,
            no_outer_labels_used_for_fitting=True,
            stopping_reused_for_checkpoint_selection=True,
            same_ensemble_exploratory_analysis=True,
        ),
    )
    print(
        score[(score.repeat == "both") & (score.size_um == "pooled")]
        .pivot(index="model", columns="case", values="r2")
        .to_string()
    )


if __name__ == "__main__":
    main()
