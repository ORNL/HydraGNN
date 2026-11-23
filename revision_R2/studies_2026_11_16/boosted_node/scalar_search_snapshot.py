"""Complete-cohort tail-map experiments, with selection inside stopping data.

No outer label enters feature construction, coefficient fitting or selection.
The original neural checkpoints used the stopping set for checkpoint selection;
the inner search is therefore an internal predictive comparison, not a fresh
nested evaluation of the full historical neural-development procedure.
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "revision_R2/studies_2026_11_15"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
QUANTILES = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99]
SELECTORS = (
    ["full", "mean_sd_0.5", "mean_sd_1", "mean_sd_1.5", "mean_sd_2"]
    + [f"q{q}" for q in [0.5, 0.65, 0.75, 0.85, 0.9, 0.95]]
    + [f"top{k}" for k in [4, 8, 16, 32]]
)
C = np.sqrt(6) / np.pi


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=str, allow_nan=False) + "\n")


def finite_gpd_maximum(u, scale, shape, k):
    if abs(shape) < 1e-6:
        return u + scale * (digamma(k + 1) + np.euler_gamma)
    t = gammaln(k + 1) + gammaln(1 - shape) - gammaln(k + 1 - shape)
    return u + scale * np.expm1(t) / shape


def summarize_peaks(values):
    """Prediction-only features. Returned statistics are always finite."""
    x = np.sort(np.asarray(values, dtype=float))
    assert len(x) >= 4 and np.isfinite(x).all()
    n = len(x)
    sd = x.std(ddof=1)
    mean = x.mean()
    d = {
        "log_count": np.log(n),
        "log_count_squared": np.log(n) ** 2,
        "mean": mean,
        "sd": sd,
        "maximum": x[-1],
        "gumbel_full": mean + C * sd * np.log(n),
    }
    for q, v in zip(QUANTILES, np.quantile(x, QUANTILES)):
        d[f"quantile_{q}"] = v
    for k in [2, 4, 8, 16, 32]:
        d[f"top_mean_{k}"] = x[-min(k, n) :].mean()
    for k in [1, 2, 4, 8]:
        d[f"order_{k}"] = x[-min(k, n)]
    for selector in SELECTORS:
        if selector == "full":
            u = x[0]
            tail = x[x > u]
        elif selector.startswith("mean_sd_"):
            u = mean + float(selector.split("_")[-1]) * sd
            tail = x[x > u]
        elif selector.startswith("q"):
            u = float(np.quantile(x, float(selector[1:])))
            tail = x[x > u]
        else:
            k = min(int(selector[3:]), n - 1)
            u = x[-k - 1]
            tail = x[-k:]
        k = len(tail)
        valid = k >= 4 and tail.std(ddof=1) > 0
        # Sparse estimates use the predicted largest grain, without an observed label.
        if valid:
            mu = tail.mean()
            s = tail.std(ddof=1)
            excess = tail - u
            gumbel = mu + C * s * np.log(k)
            exponential = u + excess.mean() * (digamma(k + 1) + np.euler_gamma)
            # L-moment/PWM GPD, bounded shape to stabilize small samples and
            # guarantee the expected maximum exists. This is a sensitivity arm.
            b0 = excess.mean()
            b1 = np.mean(np.arange(k) / (k - 1) * excess)
            l2 = 2 * b1 - b0
            xi = float(np.clip(2 - b0 / l2, -0.5, 0.5)) if l2 > 0 else -0.5
            scale = b0 * (1 - xi)
            support_ok = bool(np.all(1 + xi * excess / scale > 0))
            gpd = finite_gpd_maximum(u, scale, xi, k) if support_ok else exponential
        else:
            mu = x[-1]
            s = 0.0
            xi = 0.0
            support_ok = False
            gumbel = exponential = gpd = x[-1]
        d.update(
            {
                f"{selector}__threshold": u,
                f"{selector}__count": float(k),
                f"{selector}__valid": float(valid),
                f"{selector}__mean": mu,
                f"{selector}__sd": s,
                f"{selector}__sd_log_count": s * np.log(max(k, 1)),
                f"{selector}__shape": xi,
                f"{selector}__gpd_support_ok": float(support_ok),
                f"{selector}__gumbel": gumbel,
                f"{selector}__exponential": exponential,
                f"{selector}__gpd_pwm": gpd,
            }
        )
    assert np.isfinite(list(d.values())).all()
    return d


def extract(data_dir, out):
    rows = []
    sources = {}
    parts = json.loads((BASE / "partitions.json").read_text())
    for part in parts:
        r, f = part["repeat"], part["fold"]
        for subset, prefix in [("stopping", "stopping_"), ("evaluation", "")]:
            path = data_dir / f"grain_predictions_{prefix}repeat{r}_fold{f}.npz"
            sources[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as archive:
                z = {key: archive[key] for key in archive.files}
                assert list(z["cases"]) == CASES
                assert set(z["dataset_idx"]) == set(part[subset])
                for j, idx in enumerate(z["dataset_idx"]):
                    a, b = z["offsets"][j : j + 2]
                    assert b - a == z["grain_count"][j]
                    for c, case in enumerate(CASES):
                        assert np.isclose(
                            z["true_peak"][a:b, c].max(), z["y_true"][j, c], rtol=2e-6
                        )
                        d = summarize_peaks(z["pred_peak"][a:b, c])
                        d.update(
                            repeat=r,
                            fold=f,
                            subset=subset,
                            dataset_idx=int(idx),
                            case=case,
                            size_um=int(z["size_um"][j]),
                            truth=float(z["y_true"][j, c]),
                        )
                        if "direct_prediction" in z:
                            d["direct_head"] = float(z["direct_prediction"][j, c])
                        rows.append(d)
        print("extracted", r, f, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "prediction_features.csv.gz", index=False)
    dump(
        out / "feature_manifest.json",
        {
            "input_hashes": sources,
            "rows": len(frame),
            "feature_code_sha256": sha(__file__),
        },
    )
    return frame


def specs(columns):
    result = []
    for selector in SELECTORS:
        for estimator in ["gumbel", "exponential", "gpd_pwm"]:
            name = f"{selector}__{estimator}"
            result.append(
                {
                    "name": "raw/" + name,
                    "features": [name],
                    "mode": "identity",
                    "alpha": 0,
                    "family": "raw",
                }
            )
            for mode in ["affine", "count", "within_size"]:
                result.append(
                    {
                        "name": mode + "/" + name,
                        "features": [name],
                        "mode": mode,
                        "alpha": 10,
                        "family": "scalar_" + mode,
                    }
                )
    sets = {
        "tail_moments": [
            "mean_sd_1__mean",
            "mean_sd_1__sd",
            "mean_sd_1__sd_log_count",
            "log_count",
        ],
        "quantiles": ["mean", "sd", "maximum"]
        + [f"quantile_{q}" for q in QUANTILES]
        + ["log_count"],
        "top_order": ["mean", "sd", "maximum"]
        + [f"top_mean_{k}" for k in [2, 4, 8, 16, 32]]
        + ["log_count"],
        "combined": ["mean", "sd", "maximum"]
        + [f"quantile_{q}" for q in QUANTILES]
        + [f"top_mean_{k}" for k in [2, 4, 8, 16, 32]]
        + ["log_count"],
        "multithreshold": [
            f"{s}__{e}"
            for s in ["full", "mean_sd_1", "q0.75", "q0.9", "top8"]
            for e in ["gumbel", "exponential", "gpd_pwm"]
        ]
        + ["log_count"],
    }
    if "direct_head" in columns:
        sets["combined_direct"] = sets["combined"] + ["direct_head"]
    for label, fields in sets.items():
        for mode in ["affine", "count", "within_size"]:
            for alpha in [1.0, 10.0, 100.0]:
                result.append(
                    {
                        "name": f"ridge/{label}/{mode}/a{alpha:g}",
                        "features": fields,
                        "mode": mode,
                        "alpha": alpha,
                        "family": "summary",
                    }
                )
    return result


def fit_predict(spec, train, test, y, bt, bv, means, scales):
    fields = spec["features"]
    a = train[fields].to_numpy()
    b = test[fields].to_numpy()
    mode = spec["mode"]
    target = y.copy()
    if mode == "identity":
        return b[:, 0], {}
    if mode == "count":
        ln = train.log_count.to_numpy()
        lv = test.log_count.to_numpy()
        # Interactions allow the scalar bias/slope to change smoothly with grain count.
        a = np.column_stack([a, ln, ln ** 2, a[:, 0] * ln])
        b = np.column_stack([b, lv, lv ** 2, b[:, 0] * lv])
    centers = None
    if mode == "within_size":
        centers = np.stack([a[bt == i].mean(0) for i in range(5)])
        a = a - centers[bt]
        b = b - centers[bv]
        target = (target - means[bt]) / scales[bt]
    xm = a.mean(0)
    xs = a.std(0)
    xs[xs == 0] = 1
    a = (a - xm) / xs
    b = (b - xm) / xs
    ym = target.mean()
    coef = np.linalg.solve(
        a.T @ a + spec["alpha"] * np.eye(a.shape[1]), a.T @ (target - ym)
    )
    pred = ym + b @ coef
    if mode == "within_size":
        pred = means[bv] + scales[bv] * pred
    # FIP maxima in this dataset are nonnegative; this deterministic constraint
    # uses no evaluated target and applies uniformly to fitted alternatives.
    pred = np.maximum(pred, 0)
    assert np.isfinite(pred).all()
    return pred, {
        "x_mean": xm.tolist(),
        "x_scale": xs.tolist(),
        "coefficients": coef.tolist(),
        "intercept": float(ym),
        "size_feature_centers": None if centers is None else centers.tolist(),
    }


def score(y, p):
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return dict(
        r2=float(1 - (e * e).sum() / sst),
        rmse=float(np.sqrt(np.mean(e * e))),
        mae=float(abs(e).mean()),
        bias=float(e.mean()),
        truth_sd=float(y.std(ddof=1)),
        pred_sd=float(p.std(ddof=1)),
        pearson=float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 else 0.0,
        spearman=float(spearmanr(y, p).statistic) if p.std() > 0 else 0.0,
        n=len(y),
    )


def run(data_dir, out):
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    featurefile = out / "prediction_features.csv.gz"
    if featurefile.exists():
        manifest = json.loads((out / "feature_manifest.json").read_text())
        assert manifest["feature_code_sha256"] == sha(
            __file__
        ), "Feature cache comes from different code; use a new output directory."
        features = pd.read_csv(featurefile)
    else:
        features = extract(data_dir, out)
    candidates = specs(features.columns)
    protocol = {
        "date": "2026-11-16",
        "models": candidates,
        "selectors": SELECTORS,
        "GPD": "PWM/L-moment shape bounded [-0.5,0.5]; expected finite-k maximum; support-violating fits use exponential excesses; sparse tails use predicted maximum",
        "selection": "3-fold size-stratified CV within original stopping identities, normalized loss using fitting-only size SD; no outer-label use",
        "all_models_reported": True,
        "author_review_pending": True,
        "manuscript_modified": False,
        "neural_checkpoints_frozen": True,
        "stopping_previously_used_for_checkpoint_selection": True,
        "raw_labels_not_used_in_predictive_features": True,
        "script_sha256": sha(__file__),
    }
    dump(out / "protocol.json", protocol)
    parts = json.loads((BASE / "partitions.json").read_text())
    with np.load(BASE / "regressors/raw_features940.npz") as z:
        all_y = z["raw_truth"]
        buckets = z["size_bucket"]
    rows = []
    fitrecords = []
    selection = []
    cvrecords = []
    for part in parts:
        r, f = part["repeat"], part["fold"]
        fitids = np.array(part["fitting"])
        means = np.stack(
            [all_y[fitids[buckets[fitids] == j]].mean(0) for j in range(5)]
        )
        scales = np.stack(
            [all_y[fitids[buckets[fitids] == j]].std(0, ddof=1) for j in range(5)]
        )
        assert not set(part["evaluation"]) & (set(fitids) | set(part["stopping"]))
        for c, case in enumerate(CASES):
            sub = features[
                (features["repeat"] == r)
                & (features.fold == f)
                & (features.case == case)
            ]
            tr = (
                sub[sub.subset == "stopping"]
                .sort_values("dataset_idx")
                .reset_index(drop=True)
            )
            ev = (
                sub[sub.subset == "evaluation"]
                .sort_values("dataset_idx")
                .reset_index(drop=True)
            )
            assert set(tr.dataset_idx) == set(part["stopping"]) and set(
                ev.dataset_idx
            ) == set(part["evaluation"])
            bt = np.searchsorted(SIZES, tr.size_um.to_numpy())
            bv = np.searchsorted(SIZES, ev.size_um.to_numpy())
            y = tr.truth.to_numpy()
            truth = ev.truth.to_numpy()
            cv = list(
                StratifiedKFold(
                    3, shuffle=True, random_state=20261116 + r * 10 + f
                ).split(y, bt)
            )
            predictions = {}
            loss = {}
            for spec in candidates:
                pred, record = fit_predict(
                    spec, tr, ev, y, bt, bv, means[:, c], scales[:, c]
                )
                predictions[spec["name"]] = pred
                fitrecords.append(
                    {
                        "repeat": r,
                        "fold": f,
                        "case": case,
                        "model": spec["name"],
                        **record,
                    }
                )
                if spec["mode"] != "identity":
                    oof = np.zeros_like(y)
                    for a, b in cv:
                        pp, _ = fit_predict(
                            spec,
                            tr.iloc[a],
                            tr.iloc[b],
                            y[a],
                            bt[a],
                            bt[b],
                            means[:, c],
                            scales[:, c],
                        )
                        oof[b] = pp
                    loss[spec["name"]] = float(
                        np.mean(((oof - y) / scales[bt, c]) ** 2)
                    )
                    cvrecords.append(
                        {
                            "repeat": r,
                            "fold": f,
                            "case": case,
                            "model": spec["name"],
                            "normalized_cv_mse": loss[spec["name"]],
                        }
                    )
            families = {
                "selected_all": [
                    s["name"] for s in candidates if s["mode"] != "identity"
                ],
                "selected_scalar": [
                    s["name"] for s in candidates if s["family"].startswith("scalar")
                ],
                "selected_summary": [
                    s["name"] for s in candidates if s["family"] == "summary"
                ],
                "selected_within_size": [
                    s["name"] for s in candidates if s["mode"] == "within_size"
                ],
            }
            for label, names in families.items():
                winner = min(names, key=lambda n: loss[n])
                predictions[label] = predictions[winner]
                selection.append(
                    {
                        "repeat": r,
                        "fold": f,
                        "case": case,
                        "selector": label,
                        "chosen_model": winner,
                        "inner_cv_mse": loss[winner],
                        "stopping_ids": tr.dataset_idx.tolist(),
                        "evaluation_ids": ev.dataset_idx.tolist(),
                        "inner_folds": [
                            {
                                "training_ids": tr.iloc[a].dataset_idx.tolist(),
                                "validation_ids": tr.iloc[b].dataset_idx.tolist(),
                            }
                            for a, b in cv
                        ],
                    }
                )
            for name, pred in predictions.items():
                for j, idx in enumerate(ev.dataset_idx):
                    rows.append(
                        dict(
                            model=name,
                            repeat=r,
                            fold=f,
                            case=case,
                            dataset_idx=int(idx),
                            size_um=int(ev.size_um.iloc[j]),
                            truth=float(truth[j]),
                            prediction=float(pred[j]),
                        )
                    )
        print("fitted", r, f, "seconds", round(time.time() - started, 1), flush=True)
    pred = pd.DataFrame(rows)
    assert not pred.duplicated(["model", "repeat", "case", "dataset_idx"]).any()
    assert pred.groupby(["model", "repeat", "case"]).size().eq(940).all()
    assert np.isfinite(pred.prediction).all()
    pred.to_csv(out / "predictions.csv.gz", index=False)
    metrics = []
    for (model, case, r), g in pred.groupby(["model", "case", "repeat"]):
        for size in ["pooled", *SIZES.tolist()]:
            q = g if size == "pooled" else g[g.size_um == size]
            metrics.append(
                {
                    "model": model,
                    "case": case,
                    "repeat": r,
                    "size_um": str(size),
                    **score(q.truth.to_numpy(), q.prediction.to_numpy()),
                }
            )
    m = pd.DataFrame(metrics)
    m.to_csv(out / "metrics_by_repeat.csv", index=False)
    summary = m.groupby(["model", "case", "size_um"]).agg(
        {
            k: ["mean", "min", "max"]
            for k in [
                "r2",
                "rmse",
                "mae",
                "bias",
                "truth_sd",
                "pred_sd",
                "pearson",
                "spearman",
            ]
        }
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(cvrecords).to_csv(out / "inner_cv_scores.csv", index=False)
    dump(out / "selection_records.json", selection)
    dump(out / "fit_coefficients.json", fitrecords)
    dump(
        out / "execution_manifest.json",
        {
            "status": "complete",
            "seconds": time.time() - started,
            "models": pred.model.nunique(),
            "predictions": len(pred),
            "finite": True,
            "input_features_sha256": sha(featurefile),
            "partitions_sha256": sha(BASE / "partitions.json"),
            "script_sha256": sha(__file__),
            "no_outer_labels_in_fit_or_selection": True,
            "no_omitted_sves": True,
        },
    )
    print(
        summary.loc[
            [
                i
                for i in summary.index
                if i[0].startswith("selected_") and i[2] == "pooled"
            ],
            ["r2_mean", "rmse_mean"],
        ].to_string(),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=OLD / "data")
    p.add_argument("--output", type=Path, default=HERE / "gnn")
    args = p.parse_args()
    run(args.data_dir, args.output)
