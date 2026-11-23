"""Fixed mechanical and maximum comparisons from saved checkpoint predictions."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
OLD = ROOT / "revision_R2/studies_2026_11_15"
TREE = P.parent / "mechanical_tree_baselines"
sp = importlib.util.spec_from_file_location(
    "existing_calibration", OLD / "baselines/run_moment_calibration.py"
)
cal = importlib.util.module_from_spec(sp)
sp.loader.exec_module(cal)
FIELDS = [
    "yield_stress",
    "hardening_modulus",
    "Wp_lcf",
    "Wp_hcf",
    "sigma_amp_lcf",
    "sigma_amp_hcf",
]
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
ROUTES = ["direct", "uncalibrated_bridge", "calibrated_bridge", "grain_max"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ah = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load(p):
    with np.load(p) as z:
        return {k: z[k].copy() for k in z.files}


def dump(name, value):
    (P / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv(name, frame):
    frame.to_csv(P / name, index=False, float_format="%.17g")


def scores(y, p):
    e = p - y
    sy = y.std(ddof=1)
    sd = p.std(ddof=1)
    return dict(
        r2=float(1 - (e * e).sum() / ((y - y.mean()) ** 2).sum()),
        rmse=float(np.sqrt((e * e).mean())),
        mae=float(abs(e).mean()),
        bias=float(e.mean()),
        truth_sd=float(sy),
        prediction_sd=float(sd),
        prediction_to_truth_sd=float(sd / sy),
        pearson=float(np.corrcoef(y, p)[0, 1]) if sd > 0 else 0.0,
    )


def export(prefix, pred, truth, models, fields, sizes, folds):
    assert pred.shape == (len(models), 2, 940, len(fields)) and np.isfinite(pred).all()
    rows = []
    metrics = []
    for mi, model in enumerate(models):
        for r in range(2):
            for ci, field in enumerate(fields):
                unit = "dimensionless" if field.endswith("_fs") else "MPa"
                for i in range(940):
                    rows.append(
                        dict(
                            model=model,
                            repeat=r,
                            fold=int(folds[r, i]),
                            dataset_idx=i,
                            size_um=int(sizes[i]),
                            field=field,
                            truth=float(truth[i, ci]),
                            prediction=float(pred[mi, r, i, ci]),
                            units=unit,
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
    csv(prefix + "_predictions_long.csv", pd.DataFrame(rows))
    m = pd.DataFrame(metrics)
    csv(prefix + "_metrics_by_repeat.csv", m)
    summary = (
        m.groupby(["model", "field", "size_um"], sort=False)[
            list(scores(truth[:, 0], pred[0, 0, :, 0]))
        ]
        .mean()
        .add_suffix("_mean")
        .reset_index()
    )
    csv(prefix + "_metrics_summary.csv", summary)
    np.savez_compressed(
        P / (prefix + "_arrays.npz"),
        prediction=pred,
        truth=truth,
        models=np.array(models),
        fields=np.array(fields),
        size_um=sizes,
        fold=folds,
        dataset_idx=np.arange(940),
    )
    return summary


def pairs(prefix, pred, truth, models, fields, sizes, contrasts):
    losses = ((pred - truth[None, None, :, :]) ** 2).mean(1)
    out = []
    hashes = {}
    for size in ["pooled", *map(str, SIZES)]:
        ix = np.arange(940) if size == "pooled" else np.flatnonzero(sizes == int(size))
        n = len(ix)
        rng = np.random.default_rng(20261117)
        w = np.zeros((4000, n), np.int32)
        for s in np.unique(sizes[ix]):
            col = np.flatnonzero(sizes[ix] == s)
            draw = rng.integers(0, len(col), size=(4000, len(col)))
            w[:, col] = np.array([np.bincount(d, minlength=len(col)) for d in draw])
        hashes[size] = ah(w)
        cells = losses[:, ix, :].transpose(1, 0, 2).reshape(n, -1)
        mse = cells.mean(0).reshape(len(models), len(fields))
        bmse = (w @ cells / n).reshape(4000, len(models), len(fields))
        y = truth[ix]
        var = y.var(0, ddof=0)
        bvar = w @ (y * y) / n - (w @ y / n) ** 2
        for candidate, reference in contrasts:
            a, b = models.index(candidate), models.index(reference)
            dr2 = (mse[b] - mse[a]) / var
            drmse = np.sqrt(mse[a]) - np.sqrt(mse[b])
            qr2 = np.quantile((bmse[:, b] - bmse[:, a]) / bvar, [0.025, 0.975], axis=0)
            qrmse = np.quantile(
                np.sqrt(bmse[:, a]) - np.sqrt(bmse[:, b]), [0.025, 0.975], axis=0
            )
            for c, field in enumerate(fields):
                out.append(
                    dict(
                        model=candidate,
                        reference=reference,
                        field=field,
                        size_um=size,
                        n_sve=n,
                        model_r2=float(1 - mse[a, c] / var[c]),
                        reference_r2=float(1 - mse[b, c] / var[c]),
                        delta_r2=float(dr2[c]),
                        delta_r2_ci_low=float(qr2[0, c]),
                        delta_r2_ci_high=float(qr2[1, c]),
                        model_rmse=float(np.sqrt(mse[a, c])),
                        reference_rmse=float(np.sqrt(mse[b, c])),
                        delta_rmse=float(drmse[c]),
                        delta_rmse_ci_low=float(qrmse[0, c]),
                        delta_rmse_ci_high=float(qrmse[1, c]),
                        units="dimensionless" if field.endswith("_fs") else "MPa",
                    )
                )
    csv(prefix + "_paired_comparisons.csv", pd.DataFrame(out))
    return hashes


def main():
    tic = time.monotonic()
    assert not (P / "analysis_completion.json").exists()
    inference = json.loads((P / "inference_completion.json").read_text())
    assert inference["status"] == "complete"
    selection = json.loads((P / "selection.json").read_text())
    choices = selection["decisions"]
    for path, digest in selection["hashes"].items():
        assert sha(path) == digest
    priorpath = OLD / "baselines/graph_calibration_predictions.csv"
    publishedpath = OLD / "bridge/predictions_long.csv"
    tailpath = OLD / "baselines/graph_tail_summaries.csv"
    ledgerpath = OLD / "baselines/graph_calibration_ledger.csv"
    paths = [
        Path(__file__),
        P / "BRIDGE_PARITY_ADDENDUM.txt",
        Path(cal.__file__),
        priorpath,
        publishedpath,
        tailpath,
        ledgerpath,
        P / "selection.json",
        P / "inference_completion.json",
    ]
    paths += [Path(c["output_path"]) for c in inference["checks"]]
    hashes = {str(p): sha(p) for p in paths}
    dump(
        "analysis_manifest.json",
        dict(
            before_new_scores=True,
            created_utc=datetime.now(timezone.utc).isoformat(),
            hashes=hashes,
        ),
    )
    source = load(TREE / "prediction_arrays.npz")
    truth = source["truth"]
    sizes = source["size_um"]
    folds = source["fold"]
    models = list(source["models"].astype(str)) + [
        "best_replay",
        "last_fixed",
        "joint_selected",
    ]
    mech = np.concatenate([source["prediction"], np.full((3, 2, 940, 6), np.nan)])
    maxmodels = [
        f"{c}_{route}" for c in ["best", "last", "joint_selected"] for route in ROUTES
    ] + ["published_calibrated_bridge"]
    maximum = np.full((len(maxmodels), 2, 940, 4), np.nan)
    maxtruth = np.full((940, 4), np.nan)
    calibration = []
    tailrows = []
    parityrows = []
    scales = np.full((2, 940, 4), np.nan)
    fallbackrows = []
    priortails = pd.read_csv(tailpath, float_precision="round_trip")
    oldledger = pd.read_csv(ledgerpath, float_precision="round_trip")
    oldledger = oldledger[oldledger.model == "Graph Affine bridge calibration"]
    for choice in choices:
        r, f = choice["repeat"], choice["fold"]
        key = f"repeat{r}_fold{f}"
        stat = load(BASE / "gnn" / key / "fitting_statistics.npz")
        for label in ["best", "last"]:
            stop = load(choice["cached_stopping"][label])
            test = load(P / "outer" / f"{key}_{label}.npz")
            ids = test["dataset_idx"]
            assert ids.tolist() == choice["evaluation_ids"]
            assert stop["dataset_idx"].tolist() == choice["stopping_ids"] and not set(
                ids
            ) & set(stop["dataset_idx"])
            assert np.array_equal(test["graph_target_raw"][:, 8:14], truth[ids])
            assert list(test["graph_mean_names"][8:14]) == FIELDS
            assert np.array_equal(test["size_um"], sizes[ids]) and np.all(
                folds[r, ids] == f
            )
            if np.isfinite(maxtruth[ids]).all():
                assert np.array_equal(maxtruth[ids], test["y_true"])
            maxtruth[ids] = test["y_true"]
            scales[r, ids] = stat["sd_by_size"][test["size_bucket"]]
            mech[
                models.index("best_replay" if label == "best" else "last_fixed"), r, ids
            ] = test["graph_mean_raw"][:, 8:14]
            if label == choice["selected"]:
                mech[models.index("joint_selected"), r, ids] = test["graph_mean_raw"][
                    :, 8:14
                ]
            xs, ms, ks = cal.summaries(stop)
            xt, mt, kt = cal.summaries(test)
            for subset, z, x, k in [
                ("stopping", stop, xs, ks),
                ("evaluation", test, xt, kt),
            ]:
                assert np.isfinite(z["pred_peak"]).all()
                for j in range(len(z["dataset_idx"])):
                    n = int(z["grain_count"][j])
                    assert z["offsets"][j + 1] - z["offsets"][j] == n
                    assert np.array_equal(
                        n * k[j] / n, k[j]
                    ), "Effective count differs from selected count"
                    for c, case in enumerate(CASES):
                        tailrows.append(
                            dict(
                                checkpoint=label,
                                subset=subset,
                                repeat=r,
                                fold=f,
                                dataset_idx=int(z["dataset_idx"][j]),
                                case=case,
                                size_um=int(z["size_um"][j]),
                                selected_count=int(k[j, c]),
                                eligible=bool(np.isfinite(x[j, c]).all()),
                                tail_mean=float(x[j, c, 0]),
                                tail_sd=float(x[j, c, 1]),
                                sd_log_count=float(x[j, c, 2]),
                            )
                        )
                if label == "best":
                    oldtail = priortails[
                        (priortails.repeat == r)
                        & (priortails.fold == f)
                        & (priortails.subset == subset)
                    ].set_index(["dataset_idx", "case"])
                    for j, i in enumerate(z["dataset_idx"]):
                        for c, case in enumerate(CASES):
                            row = oldtail.loc[(i, case)]
                            oldok = bool(
                                np.isfinite(
                                    [row.tail_mean, row.tail_sd, row.sd_log_count]
                                ).all()
                            )
                            parityrows.append(
                                dict(
                                    subset=subset,
                                    repeat=r,
                                    fold=f,
                                    dataset_idx=int(i),
                                    case=case,
                                    old_count=int(row.selected_count),
                                    new_count=int(k[j, c]),
                                    old_eligible=oldok,
                                    new_eligible=bool(np.isfinite(x[j, c]).all()),
                                    count_changed=int(row.selected_count)
                                    != int(k[j, c]),
                                    eligibility_changed=oldok
                                    != bool(np.isfinite(x[j, c]).all()),
                                )
                            )
            calibrated = stat["mean_by_size"][test["size_bucket"]].copy()
            for c, case in enumerate(CASES):
                a = (xs[:, c, 0] + np.sqrt(6) / np.pi * xs[:, c, 2])[:, None]
                b = (xt[:, c, 0] + np.sqrt(6) / np.pi * xt[:, c, 2])[:, None]
                ok = np.isfinite(a).all(1)
                good = np.isfinite(b).all(1)
                assert ok.sum() >= 10
                scaler = StandardScaler().fit(a[ok])
                fit = Ridge(alpha=10).fit(
                    scaler.transform(a[ok]), stop["y_true"][ok, c]
                )
                calibrated[good, c] = fit.predict(scaler.transform(b[good]))
                calibration.append(
                    dict(
                        checkpoint=label,
                        repeat=r,
                        fold=f,
                        case=case,
                        alpha=10,
                        calibration_ids=stop["dataset_idx"][ok].tolist(),
                        n_calibration=int(ok.sum()),
                        x_mean=scaler.mean_.tolist(),
                        x_scale=scaler.scale_.tolist(),
                        coefficient=fit.coef_.tolist(),
                        intercept=float(fit.intercept_),
                        selected_for_all_outputs=label == choice["selected"],
                    )
                )
                for j, i in enumerate(ids):
                    fallbackrows.append(
                        dict(
                            checkpoint=label,
                            repeat=r,
                            fold=f,
                            dataset_idx=int(i),
                            case=case,
                            bridge_fallback=bool(not good[j]),
                            uncalibrated_fallback=bool(test["bridge_fallback"][j, c]),
                        )
                    )
            values = {
                "direct": test["direct_prediction"],
                "uncalibrated_bridge": test["bridge_prediction"],
                "calibrated_bridge": calibrated,
                "grain_max": test["maximum_predicted_grain"],
            }
            for route, v in values.items():
                assert np.isfinite(v).all()
                maximum[maxmodels.index(label + "_" + route), r, ids] = v
                if label == choice["selected"]:
                    maximum[maxmodels.index("joint_selected_" + route), r, ids] = v
    csv("tail_summaries.csv", pd.DataFrame(tailrows))
    csv("best_tail_eligibility_parity.csv", pd.DataFrame(parityrows))
    csv("bridge_fallback_ledger.csv", pd.DataFrame(fallbackrows))
    dump("calibration_fits.json", calibration)
    oldpred = pd.read_csv(priorpath, float_precision="round_trip")
    oldpred = oldpred[oldpred.model == "Graph Affine bridge calibration"].copy()
    assert len(oldpred) == 7520
    pub = pd.read_csv(
        publishedpath,
        usecols=[
            "model",
            "repeat",
            "fold",
            "dataset_idx",
            "size_um",
            "case",
            "truth",
            "prediction",
        ],
        float_precision="round_trip",
        low_memory=False,
    )
    pub = pub[pub.model == "Graph Affine bridge calibration"]
    assert len(pub) == 7520
    keys = ["repeat", "fold", "dataset_idx", "size_um", "case"]
    pub = oldpred.merge(
        pub, on=keys, validate="one_to_one", suffixes=("_fit", "_published")
    )
    assert np.max(abs(pub.prediction_fit - pub.prediction_published)) < 1e-14
    differences = []
    for row in oldpred.itertuples():
        c = CASES.index(row.case)
        i = row.dataset_idx
        r = row.repeat
        assert abs(row.truth - maxtruth[i, c]) < 1e-14 and folds[r, i] == row.fold
        maximum[-1, r, i, c] = row.prediction
        new = maximum[maxmodels.index("best_calibrated_bridge"), r, i, c]
        delta = new - row.prediction
        oldfallback = bool(
            oldledger[
                (oldledger.repeat == r)
                & (oldledger.fold == row.fold)
                & (oldledger.dataset_idx == i)
                & (oldledger.case == row.case)
            ]
            .iloc[0]
            .fallback
        )
        newfallback = next(
            x["bridge_fallback"]
            for x in fallbackrows
            if x["checkpoint"] == "best"
            and x["repeat"] == r
            and x["fold"] == row.fold
            and x["dataset_idx"] == i
            and x["case"] == row.case
        )
        differences.append(
            dict(
                repeat=r,
                fold=row.fold,
                dataset_idx=i,
                size_um=row.size_um,
                case=row.case,
                old_prediction=row.prediction,
                new_prediction=new,
                difference=delta,
                difference_in_fitting_size_sd=delta / scales[r, i, c],
                old_fallback=oldfallback,
                new_fallback=newfallback,
                fallback_changed=oldfallback != newfallback,
            )
        )
    diff = pd.DataFrame(differences)
    csv("best_calibrated_bridge_parity.csv", diff)
    parityreport = dict(
        rows=7520,
        max_abs_error=float(abs(diff.difference).max()),
        max_error_in_fitting_size_sd=float(
            abs(diff.difference_in_fitting_size_sd).max()
        ),
        count_changed_rows=int(pd.DataFrame(parityrows).count_changed.sum()),
        eligibility_changed_rows=int(
            pd.DataFrame(parityrows).eligibility_changed.sum()
        ),
        fallback_changed_rows=int(diff.fallback_changed.sum()),
        effective_count_equals_selected_count=True,
        published_prediction_table_matches_saved_calibrator=True,
        tolerance_in_fitting_size_sd=1e-4,
    )
    dump("bridge_parity.json", parityreport)
    assert parityreport["max_error_in_fitting_size_sd"] < 1e-4, parityreport
    assert np.isfinite(maxtruth).all()
    m = export("mechanical", mech, truth, models, FIELDS, sizes, folds)
    mx = export("maximum", maximum, maxtruth, maxmodels, CASES, sizes, folds)
    refs = [
        "original",
        "RF_fitting",
        "HGB_fitting",
        "SizeMean_fitting",
        "RF_outer_training",
        "HGB_outer_training",
        "SizeMean_outer_training",
    ]
    h = pairs(
        "mechanical",
        mech,
        truth,
        models,
        FIELDS,
        sizes,
        [(a, b) for a in ["joint_selected", "last_fixed"] for b in refs],
    )
    h2 = pairs(
        "maximum",
        maximum,
        maxtruth,
        maxmodels,
        CASES,
        sizes,
        [
            (a + "_" + route, "best_" + route)
            for a in ["joint_selected", "last"]
            for route in ROUTES
        ],
    )
    assert h == h2
    assert all(sha(p) == digest for p, digest in hashes.items())
    dump(
        "analysis_completion.json",
        dict(
            status="complete",
            created_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic() - tic,
            mechanical_models=models,
            maximum_models=maxmodels,
            bridge_parity=parityreport,
            bootstrap_weight_sha256=h,
            input_hashes_unchanged=True,
            outputs={p.name: sha(p) for p in P.glob("*.csv")},
        ),
    )
    print(
        "MECHANICAL POOLED R2\n"
        + m[m.size_um == "pooled"]
        .pivot(index="field", columns="model", values="r2_mean")
        .to_string(),
        flush=True,
    )
    print(
        "MAXIMUM POOLED R2\n"
        + mx[mx.size_um == "pooled"]
        .pivot(index="field", columns="model", values="r2_mean")
        .to_string(),
        flush=True,
    )
    print("BRIDGE PARITY " + json.dumps(parityreport), flush=True)


if __name__ == "__main__":
    main()
