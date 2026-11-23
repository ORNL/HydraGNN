"""Fixed alpha=10 comparisons prompted by a calibration implementation audit."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
STUDY = PARENT.parent
BASE = (
    STUDY.parent.parent
    / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
)
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
KEYS = ["repeat", "fold", "dataset_idx", "size_um", "case"]
MODELS = ["block16_ridge10", "bridge_logcount_ridge10"]


def ridge_fit(a, b, y):
    mean, sd = a.mean(0), a.std(0, ddof=0)
    sd[sd == 0] = 1
    za, zb = (a - mean) / sd, (b - mean) / sd
    ym = y.mean()
    slope = np.linalg.solve(za.T @ za + 10 * np.eye(a.shape[1]), za.T @ (y - ym))
    pred = ym + zb @ slope
    scaler = StandardScaler().fit(a)
    reference = Ridge(alpha=10).fit(scaler.transform(a), y).predict(scaler.transform(b))
    return pred, dict(
        mean=mean.tolist(),
        sd=sd.tolist(),
        standardized_intercept=float(ym),
        standardized_slopes=slope.tolist(),
        raw_intercept=float(ym - np.sum(slope * mean / sd)),
        raw_slopes=(slope / sd).tolist(),
        alpha=10,
        max_absolute_difference_from_sklearn=float(np.max(abs(pred - reference))),
    )


def metric(frame):
    y, p = frame.truth.to_numpy(), frame.prediction.to_numpy()
    e = p - y
    return dict(
        n_predictions=len(y),
        n_sves=frame.dataset_idx.nunique(),
        r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mae=float(np.mean(abs(e))),
        bias=float(e.mean()),
        prediction_sd=float(p.std(ddof=1)),
        truth_sd=float(y.std(ddof=1)),
    )


def main():
    start = datetime.now(timezone.utc).isoformat()
    f = pd.read_csv(
        PARENT / "prediction_only_features.csv", float_precision="round_trip"
    )
    parts = json.loads((BASE / "partitions.json").read_text())
    old = pd.read_csv(
        STUDY / "baselines/graph_calibration_predictions.csv",
        float_precision="round_trip",
    )
    old = old[old.model == "Graph Affine bridge calibration"]
    with np.load(BASE / "regressors/raw_features940.npz") as z:
        y, bucket = z["raw_truth"], z["size_bucket"]
    rows, fits, parity = [], [], []
    for part in parts:
        r, fold = part["repeat"], part["fold"]
        train = np.asarray(part["fitting"])
        means = np.stack([y[train[bucket[train] == b]].mean(0) for b in range(5)])
        scales = np.stack(
            [y[train[bucket[train] == b]].std(0, ddof=1) for b in range(5)]
        )
        with np.load(
            STUDY / "data" / f"grain_predictions_stopping_repeat{r}_fold{fold}.npz"
        ) as z:
            stopids, stoplabels = z["dataset_idx"], z["y_true"]
        with np.load(
            STUDY / "data" / f"grain_predictions_repeat{r}_fold{fold}.npz"
        ) as z:
            testids, testlabels, testbuckets = (
                z["dataset_idx"],
                z["y_true"],
                z["size_bucket"],
            )
        assert set(stopids) == set(part["stopping"]) and set(testids) == set(
            part["evaluation"]
        )
        assert not set(testids) & (set(train) | set(stopids))
        for c, case in enumerate(CASES):
            sub = f[(f["repeat"] == r) & (f.fold == fold) & (f.case == case)]
            stop = (
                sub[sub.subset == "stopping"]
                .set_index("dataset_idx")
                .loc[stopids]
                .reset_index()
            )
            test = (
                sub[sub.subset == "evaluation"]
                .set_index("dataset_idx")
                .loc[testids]
                .reset_index()
            )
            for model, fields in [
                ("reconstructed_current_ridge10", ["bridge"]),
                ("block16_ridge10", ["block16_raw"]),
                ("bridge_logcount_ridge10", ["bridge", "log_count"]),
            ]:
                a, b = stop[fields].to_numpy(), test[fields].to_numpy()
                okfit, oktest = np.isfinite(a).all(1), np.isfinite(b).all(1)
                assert okfit.sum() >= 10
                assert (
                    np.linalg.matrix_rank(
                        np.column_stack([np.ones(okfit.sum()), a[okfit]])
                    )
                    == len(fields) + 1
                )
                p = means[testbuckets, c].copy()
                pp, fit = ridge_fit(a[okfit], b[oktest], stoplabels[okfit, c])
                p[oktest] = pp
                fits.append(
                    dict(
                        model=model,
                        repeat=r,
                        fold=fold,
                        case=case,
                        fields=fields,
                        n_stopping=int(okfit.sum()),
                        stopping_ids=stopids[okfit].tolist(),
                        **fit,
                    )
                )
                if model == "reconstructed_current_ridge10":
                    ref = (
                        old[
                            (old["repeat"] == r)
                            & (old.fold == fold)
                            & (old.case == case)
                        ]
                        .set_index("dataset_idx")
                        .loc[testids]
                    )
                    delta = abs(p - ref.prediction.to_numpy())
                    parity.append(
                        dict(
                            repeat=r,
                            fold=fold,
                            case=case,
                            n_predictions=len(p),
                            max_absolute_error=float(delta.max()),
                            max_error_in_fitting_sd=float(
                                (delta / scales[testbuckets, c]).max()
                            ),
                        )
                    )
                else:
                    for j, (_, rec) in enumerate(test.iterrows()):
                        rows.append(
                            {
                                **{k: rec[k] for k in KEYS},
                                "model": model,
                                "truth": float(testlabels[j, c]),
                                "prediction": float(p[j]),
                                "used_fallback": bool(not oktest[j]),
                            }
                        )
    pred = pd.DataFrame(rows)
    assert len(pred) == 940 * 2 * 4 * 2 and np.isfinite(pred.prediction).all()
    assert max(v["max_error_in_fitting_sd"] for v in parity) < 1e-10
    reference = pd.read_csv(
        PARENT / "all_comparison_predictions.csv", float_precision="round_trip"
    )
    allpred = pd.concat([pred, reference], ignore_index=True)
    pred.to_csv(HERE / "candidate_predictions.csv", index=False)
    scores = []
    for (model, case), frame in allpred.groupby(["model", "case"]):
        for rep in ["both", 0, 1]:
            rf = frame if rep == "both" else frame[frame["repeat"] == rep]
            for size in ["pooled", 125, 177, 250, 354, 500]:
                sf = rf if size == "pooled" else rf[rf.size_um == size]
                scores.append(
                    dict(model=model, case=case, repeat=rep, size_um=size, **metric(sf))
                )
    scores = pd.DataFrame(scores)
    scores.to_csv(HERE / "metrics.csv", index=False)
    contrasts = []
    rng = np.random.default_rng(20261115)
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
            ix = [
                np.flatnonzero(wide.size_um.to_numpy() == s)
                for s in sorted(wide.size_um.unique())
            ]
            res = np.hstack(
                [rng.choice(v, size=(2000, len(v)), replace=True) for v in ix]
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
                    change = 100 * (1 - np.sqrt(a[res].mean(1) / b[res].mean(1)))
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
                            reduction_ci_low=float(np.quantile(change, 0.025)),
                            reduction_ci_high=float(np.quantile(change, 0.975)),
                            mean_squared_error_difference=float(a.mean() - b.mean()),
                        )
                    )
    pd.DataFrame(contrasts).to_csv(HERE / "paired_comparisons.csv", index=False)
    (HERE / "calibration_fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    pd.DataFrame(parity).to_csv(
        HERE / "current_calibration_reconstruction.csv", index=False
    )
    paths = [
        Path(__file__),
        HERE / "protocol_amendment.json",
        PARENT / "prediction_only_features.csv",
        BASE / "partitions.json",
    ]
    manifest = dict(
        status="complete",
        started_utc=start,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        input_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        n_predictions=len(pred),
        fallback_counts=pred.groupby("model").used_fallback.sum().astype(int).to_dict(),
        current_reconstruction_max_abs=max(v["max_absolute_error"] for v in parity),
        current_reconstruction_max_in_fitting_sd=max(
            v["max_error_in_fitting_sd"] for v in parity
        ),
        reconstruction_tolerance_in_fitting_sd=1e-10,
        closed_form_vs_sklearn_max_abs=max(
            v["max_absolute_difference_from_sklearn"] for v in fits
        ),
        standardization="mean and population SD on eligible stopping rows, ddof=0; constant feature scale=1",
        objective="sum squared residuals + 10 sum squared standardized slopes; intercept unpenalized",
        same_ensemble_exploratory=True,
        no_outer_label_fitting=True,
    )
    (HERE / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        scores[(scores["repeat"] == "both") & (scores.size_um == "pooled")]
        .pivot(index="model", columns="case", values="r2")
        .to_string()
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "input_hashes"}, indent=2)
    )


if __name__ == "__main__":
    main()
