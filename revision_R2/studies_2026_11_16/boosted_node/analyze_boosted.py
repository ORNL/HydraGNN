"""Complete, fair-label HGB node-to-maximum and direct-SVE comparisons."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import json, importlib.util, hashlib
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
KEYS = ["repeat", "fold", "dataset_idx", "size_um", "case"]
FAMILIES = ["hgb_mean", "hgb_q90"]


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def imported(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metric(g):
    y = g.truth.to_numpy()
    p = g.prediction.to_numpy()
    e = p - y
    return dict(
        n=len(y),
        r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(e * e))),
        mae=float(abs(e).mean()),
        bias=float(e.mean()),
        truth_sd=float(y.std(ddof=1)),
        pred_sd=float(p.std(ddof=1)),
        pearson=float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 else 0.0,
        spearman=float(spearmanr(y, p).statistic) if p.std() > 0 else 0.0,
    )


def main():
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    cal = imported("legacy_calibration", OLD / "baselines/run_moment_calibration.py")
    parts = json.loads((BASE / "partitions.json").read_text())
    raw = load(BASE / "regressors/raw_features940.npz")
    ally = raw["raw_truth"]
    bucket = raw["size_bucket"]
    rows = []
    fits = []
    grainrows = []
    sverows = []
    counts = []
    for family in FAMILIES:
        for part in parts:
            r, f = part["repeat"], part["fold"]
            fi = np.array(part["fitting"])
            means = np.stack([ally[fi[bucket[fi] == b]].mean(0) for b in range(5)])
            stop = load(
                HERE
                / family
                / "data"
                / f"grain_predictions_stopping_repeat{r}_fold{f}.npz"
            )
            test = load(
                HERE / family / "data" / f"grain_predictions_repeat{r}_fold{f}.npz"
            )
            assert set(stop["dataset_idx"]) == set(part["stopping"]) and set(
                test["dataset_idx"]
            ) == set(part["evaluation"])
            assert not set(stop["dataset_idx"]) & set(test["dataset_idx"])
            xs, ms, ks = cal.summaries(stop)
            xt, mt, kt = cal.summaries(test)

            def compact(z, c):
                fs = []
                for j in range(len(z["dataset_idx"])):
                    x = np.sort(
                        z["pred_peak"][z["offsets"][j] : z["offsets"][j + 1], c]
                    )
                    q = np.quantile(x, [0.9, 0.95])
                    fs.append(
                        [
                            x[-1],
                            q[0],
                            q[1],
                            x[-min(4, len(x)) :].mean(),
                            x.std(ddof=1),
                            np.log(len(x)),
                        ]
                    )
                return np.asarray(fs)

            for c, case in enumerate(CASES):
                bridge_s = xs[:, c, 0] + cal.FACTOR * xs[:, c, 2]
                bridge_t = xt[:, c, 0] + cal.FACTOR * xt[:, c, 2]
                sparse = ~np.isfinite(bridge_t)
                base = means[test["size_bucket"], c]
                rawbridge = np.where(sparse, base, bridge_t)
                predictions = {"maximum": mt[:, c], "bridge_raw_complete": rawbridge}
                for route, a, b in [
                    ("maximum_ridge10", ms[:, c, None], mt[:, c, None]),
                    ("bridge_ridge10", bridge_s[:, None], bridge_t[:, None]),
                    ("order_statistics_ridge10", compact(stop, c), compact(test, c)),
                ]:
                    oks = np.isfinite(a).all(1)
                    okt = np.isfinite(b).all(1)
                    sc = StandardScaler().fit(a[oks])
                    model = Ridge(alpha=10).fit(
                        sc.transform(a[oks]), stop["y_true"][oks, c]
                    )
                    pp = base.copy()
                    pp[okt] = model.predict(sc.transform(b[okt]))
                    assert np.isfinite(pp).all()
                    predictions[route] = pp
                    fits.append(
                        dict(
                            family=family,
                            route=route,
                            repeat=r,
                            fold=f,
                            case=case,
                            stopping_ids=stop["dataset_idx"][oks].tolist(),
                            mean=sc.mean_.tolist(),
                            scale=sc.scale_.tolist(),
                            coef=model.coef_.tolist(),
                            intercept=float(model.intercept_),
                            alpha=10.0,
                            fallback_count=int((~okt).sum()),
                        )
                    )
                for name, pred in predictions.items():
                    for j, idx in enumerate(test["dataset_idx"]):
                        rows.append(
                            dict(
                                model=family + "/" + name,
                                repeat=r,
                                fold=f,
                                dataset_idx=int(idx),
                                size_um=int(test["size_um"][j]),
                                case=case,
                                truth=float(test["y_true"][j, c]),
                                prediction=float(pred[j]),
                            )
                        )
                counts.append(
                    dict(
                        family=family,
                        repeat=r,
                        fold=f,
                        case=case,
                        n_sves=len(test["dataset_idx"]),
                        selected_fewer_four=int((kt[:, c] < 4).sum()),
                        bridge_fallback=int(sparse.sum()),
                    )
                )
                for j, idx in enumerate(test["dataset_idx"]):
                    a, b = test["offsets"][j : j + 2]
                    y = test["true_peak"][a:b, c]
                    p = test["pred_peak"][a:b, c]
                    sverows.append(
                        dict(
                            family=family,
                            repeat=r,
                            fold=f,
                            dataset_idx=int(idx),
                            size_um=int(test["size_um"][j]),
                            case=case,
                            n_grains=len(y),
                            truth_mean=float(y.mean()),
                            pred_mean=float(p.mean()),
                            truth_sd=float(y.std(ddof=1)),
                            pred_sd=float(p.std(ddof=1)),
                            true_max=float(y.max()),
                            pred_max=float(p.max()),
                            grain_rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                            true_max_grain_prediction=float(p[np.argmax(y)]),
                        )
                    )
        for r in range(2):
            zs = [
                load(
                    HERE / family / "data" / f"grain_predictions_repeat{r}_fold{f}.npz"
                )
                for f in range(5)
            ]
            size = np.concatenate([z["grain_size_um"] for z in zs])
            ids = np.concatenate([z["grain_dataset_idx"] for z in zs])
            ww = np.concatenate(
                [np.repeat(1 / z["grain_count"], z["grain_count"]) for z in zs]
            )
            for c, case in enumerate(CASES):
                y = np.concatenate([z["true_peak"][:, c] for z in zs])
                p = np.concatenate([z["pred_peak"][:, c] for z in zs])
                for sz in ["pooled", *SIZES]:
                    keep = np.ones(len(y), bool) if sz == "pooled" else size == sz
                    yy = y[keep]
                    pp = p[keep]
                    e = pp - yy
                    for weighting in ["grain", "SVE balanced"]:
                        w = np.ones(len(yy)) if weighting == "grain" else ww[keep]
                        w = w / w.sum()
                        mu = w @ yy
                        mp = w @ pp
                        v = w @ ((yy - mu) ** 2)
                        grainrows.append(
                            dict(
                                family=family,
                                repeat=r,
                                case=case,
                                size_um=sz,
                                weighting=weighting,
                                n_grains=len(yy),
                                r2=float(1 - (w @ (e * e)) / v),
                                rmse=float(np.sqrt(w @ (e * e))),
                                mae=float(w @ abs(e)),
                                bias=float(w @ e),
                                pearson=float(
                                    (w @ ((yy - mu) * (pp - mp)))
                                    / np.sqrt(v * (w @ ((pp - mp) ** 2)))
                                ),
                            )
                        )
    direct = pd.concat(
        [
            pd.read_csv(HERE / f"repeat{r}_fold{f}/direct_predictions.csv")
            for r in range(2)
            for f in range(5)
        ],
        ignore_index=True,
    )
    direct = direct[direct.subset == "evaluation"].drop(columns="subset")
    prior = pd.read_csv(OLD / "label_matched_trees/predictions_long.csv")
    prior = prior[prior.model == "Outer-training Histogram boosting"]
    baserows = []
    for part in parts:
        tr = np.r_[part["fitting"], part["stopping"]]
        ev = np.array(part["evaluation"])
        mu = np.stack([ally[tr[bucket[tr] == b]].mean(0) for b in range(5)])
        for idx in ev:
            for c, case in enumerate(CASES):
                baserows.append(
                    dict(
                        model="Outer-training size mean",
                        repeat=part["repeat"],
                        fold=part["fold"],
                        dataset_idx=int(idx),
                        size_um=int(SIZES[bucket[idx]]),
                        case=case,
                        truth=float(ally[idx, c]),
                        prediction=float(mu[bucket[idx], c]),
                    )
                )
    base = pd.DataFrame(baserows)
    graph = pd.read_csv(OLD / "baselines/graph_calibration_predictions.csv")
    graph = graph[graph.model == "Graph Affine bridge calibration"]
    preds = pd.concat(
        [pd.DataFrame(rows), direct, prior, base, graph], ignore_index=True
    )[["model", *KEYS, "truth", "prediction"]]
    assert not preds.duplicated(["model", *KEYS]).any()
    assert preds.groupby(["model", "repeat", "case"]).size().eq(940).all()
    assert np.isfinite(preds.prediction).all()
    preds.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    metrics = []
    for (name, r, c), g in preds.groupby(["model", "repeat", "case"]):
        for sz in ["pooled", *SIZES]:
            q = g if sz == "pooled" else g[g.size_um == sz]
            metrics.append(
                dict(model=name, repeat=r, case=c, size_um=str(sz), **metric(q))
            )
    met = pd.DataFrame(metrics)
    met.to_csv(out / "metrics_by_repeat.csv", index=False)
    summary = met.groupby(["model", "case", "size_um"]).agg(
        {
            k: ["mean", "min", "max"]
            for k in ["r2", "rmse", "mae", "bias", "pearson", "spearman"]
        }
    )
    summary.columns = ["_".join(x) for x in summary.columns]
    summary.reset_index().to_csv(out / "metrics_summary.csv", index=False)
    rng = np.random.default_rng(20261116)
    paired = []
    for case in CASES:
        for size in ["pooled", *SIZES]:
            sub = preds[preds.case == case].copy()
            if size != "pooled":
                sub = sub[sub.size_um == size]
            sub["loss"] = (sub.truth - sub.prediction) ** 2
            wide = (
                sub.groupby(["dataset_idx", "size_um", "model"])
                .loss.mean()
                .unstack("model")
                .reset_index()
            )
            groups = [
                np.flatnonzero(wide.size_um.to_numpy() == v)
                for v in sorted(wide.size_um.unique())
            ]
            res = np.hstack(
                [rng.choice(g, size=(2000, len(g)), replace=True) for g in groups]
            )
            for family in FAMILIES:
                for route in [
                    "maximum",
                    "bridge_raw_complete",
                    "maximum_ridge10",
                    "bridge_ridge10",
                    "order_statistics_ridge10",
                ]:
                    name = family + "/" + route
                    refs = (
                        ["hgb direct mean fitting"]
                        if route in ["maximum", "bridge_raw_complete"]
                        else ["hgb direct mean outer_training"]
                    )
                    refs += [
                        "Outer-training size mean",
                        "Outer-training Histogram boosting",
                        "Graph Affine bridge calibration",
                    ]
                    for ref in refs:
                        aa = wide[name].to_numpy()
                        bb = wide[ref].to_numpy()
                        delta = 100 * (1 - np.sqrt(aa[res].mean(1) / bb[res].mean(1)))
                        paired.append(
                            dict(
                                model=name,
                                reference=ref,
                                case=case,
                                size_um=size,
                                n_sves=len(wide),
                                rmse_reduction_percent=float(
                                    100 * (1 - np.sqrt(aa.mean() / bb.mean()))
                                ),
                                ci_low=float(np.quantile(delta, 0.025)),
                                ci_high=float(np.quantile(delta, 0.975)),
                            )
                        )
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(grainrows).to_csv(out / "grain_metrics_by_repeat.csv", index=False)
    pd.DataFrame(sverows).to_csv(out / "grain_to_maximum_diagnostics.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "fallback_counts.csv", index=False)
    (out / "calibration_fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    (out / "completion.json").write_text(
        json.dumps(
            dict(
                status="complete",
                models=preds.model.nunique(),
                prediction_rows=len(preds),
                all940_each_case_repeat=True,
                paired_bootstraps=2000,
                repeats_clustered_by_sve=True,
                no_outer_labels_used_for_fit=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        summary.loc[
            [i for i in summary.index if i[2] == "pooled"], ["r2_mean", "rmse_mean"]
        ].to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
