"""Honest SVE-grouped cross-fitted grain-to-maximum stacking, fixed settings."""
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import sys, json, time, argparse, importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "base_boosted", HERE.parent / "run_boosted.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = m.CONFIG.copy()
NAMES = [
    "grain_count",
    "log_grain_count",
    "mean",
    "sd",
    "max",
    "q90",
    "q95",
    "q99",
    "top4mean",
    "top16mean",
    "tail_moment",
] + [
    n + "_times_logN"
    for n in [
        "mean",
        "sd",
        "max",
        "q90",
        "q95",
        "q99",
        "top4mean",
        "top16mean",
        "tail_moment",
    ]
]


def summaries(pred, counts):
    offsets = np.r_[0, np.cumsum(counts)]
    result = np.empty((len(counts), 4, len(NAMES)))
    for i, n in enumerate(counts):
        for c in range(4):
            x = np.sort(pred[offsets[i] : offsets[i + 1], c])
            mean = x.mean()
            sd = x.std(ddof=1)
            u = mean + sd
            tail = x[x > u]
            moment = (
                tail.mean() + np.sqrt(6) / np.pi * tail.std(ddof=1) * np.log(len(tail))
                if len(tail) >= 4 and tail.std(ddof=1) > 0
                else x[-1]
            )
            base = np.r_[
                n,
                np.log(n),
                mean,
                sd,
                x[-1],
                np.quantile(x, [0.9, 0.95, 0.99]),
                x[-min(4, n) :].mean(),
                x[-min(16, n) :].mean(),
                moment,
            ]
            result[i, c] = np.r_[base, base[2:] * np.log(n)]
    assert np.isfinite(result).all()
    return result


def fit_predict(a, tr, ev, folder, seed, threads):
    folder.mkdir(exist_ok=True)
    gb = a["grain_size_bucket"]
    gid = a["grain_dataset_idx"]
    fit = np.isin(gid, tr)
    ii = np.concatenate([np.arange(a["offsets"][i], a["offsets"][i + 1]) for i in ev])
    assert not set(tr) & set(ev)
    xc = np.stack([a["raw32"][fit & (gb == k)].mean(0, dtype=float) for k in range(5)])
    mu = np.stack([a["grain_truth"][fit & (gb == k)].mean(0) for k in range(5)])
    sd = np.stack([a["grain_truth"][fit & (gb == k)].std(0, ddof=1) for k in range(5)])
    X = np.column_stack([a["raw32"][fit] - xc[gb[fit]], np.eye(5)[gb[fit]]]).astype(
        np.float32
    )
    Z = np.column_stack([a["raw32"][ii] - xc[gb[ii]], np.eye(5)[gb[ii]]]).astype(
        np.float32
    )
    yy = (a["grain_truth"][fit] - mu[gb[fit]]) / sd[gb[fit]]
    w = 1 / a["grain_count"][gid[fit]].astype(float)
    w /= w.mean()
    total = np.bincount(gid[fit], weights=w, minlength=940)[tr]
    assert np.allclose(total, total[0])
    pred = np.empty((len(ii), 4))
    start = time.time()
    for c, case in enumerate(m.CASES):
        saved = folder / (case + "_predictions.npy")
        mp = folder / (case + ".joblib")
        if saved.exists() and mp.exists():
            pred[:, c] = np.load(saved)
            continue
        model = HistGradientBoostingRegressor(
            **CONFIG, loss="squared_error", random_state=seed
        )
        with threadpool_limits(limits=threads):
            model.fit(X, yy[:, c], sample_weight=w)
            pred[:, c] = model.predict(Z) * sd[gb[ii], c] + mu[gb[ii], c]
        joblib.dump(model, mp, compress=3)
        np.save(saved, pred[:, c])
    m.dump(
        folder / "preprocessing.json",
        dict(
            fit_ids=tr.tolist(),
            predicted_ids=ev.tolist(),
            feature_mean=xc.tolist(),
            grain_mean=mu.tolist(),
            grain_sd=sd.tolist(),
            sample_weights="1/n_grains per SVE normalized mean1",
            seed=seed,
            seconds=time.time() - start,
        ),
    )
    return pred


def run(r, f, threads):
    assert json.loads((HERE / "protocol.json").read_text())["fixed_settings"] == CONFIG
    a = m.inputs()
    b = a["size_bucket"]
    p = json.loads((m.BASE / f"partition_r{r}_f{f}.json").read_text())
    union = np.sort(np.r_[p["fitting"], p["stopping"]])
    ev = np.array(p["evaluation"])
    seed = p["initialization_seed"]
    start = time.time()
    folder = HERE / f"repeat{r}_fold{f}"
    folder.mkdir(exist_ok=True)
    if (folder / "completion.json").exists():
        print("EXISTS", r, f, flush=True)
        return
    assert not set(union) & set(ev) and set(np.r_[union, ev]) == set(range(940))
    cv = StratifiedKFold(3, shuffle=True, random_state=20261116 + r * 10 + f)
    oof = np.empty((len(union), 4, len(NAMES)))
    seen = []
    records = []
    for k, (tr, va) in enumerate(cv.split(union, b[union])):
        preds = fit_predict(
            a, union[tr], union[va], folder / f"inner{k}", seed + k, threads
        )
        oof[va] = summaries(preds, a["grain_count"][union[va]])
        seen.extend(union[va].tolist())
        records.append(
            dict(
                inner_fold=k,
                fitting_ids=union[tr].tolist(),
                heldout_ids=union[va].tolist(),
            )
        )
        print("INNER COMPLETE", r, f, k, time.time() - start, flush=True)
    assert sorted(seen) == union.tolist() and len(set(seen)) == len(union)
    pred = fit_predict(a, union, ev, folder / "union_refit", seed, threads)
    features = summaries(pred, a["grain_count"][ev])
    target = a["sve_truth"][union]
    truth = a["sve_truth"][ev]
    means = np.stack([target[b[union] == k].mean(0) for k in range(5)])
    sd = np.stack([target[b[union] == k].std(0, ddof=1) for k in range(5)])
    rows = []
    fits = []
    for c, case in enumerate(m.CASES):
        X = np.column_stack([oof[:, c], np.eye(5)[b[union]]])
        Z = np.column_stack([features[:, c], np.eye(5)[b[ev]]])
        sc = StandardScaler().fit(X)
        xx = sc.transform(X)
        zz = sc.transform(Z)
        predictions = {
            "raw_maximum": features[:, c, NAMES.index("max")],
            "raw_tail_moment": features[:, c, NAMES.index("tail_moment")],
        }
        for route in ["ridge10_raw", "ridge10_size_residual"]:
            yy = (
                target[:, c]
                if route == "ridge10_raw"
                else (target[:, c] - means[b[union], c]) / sd[b[union], c]
            )
            model = Ridge(alpha=10).fit(xx, yy)
            pp = model.predict(zz)
            if route == "ridge10_size_residual":
                pp = means[b[ev], c] + sd[b[ev], c] * pp
            predictions[route] = pp
            joblib.dump(
                dict(
                    scaler=sc, model=model, target_mean=means[:, c], target_sd=sd[:, c]
                ),
                folder / f"{route}_{case}.joblib",
                compress=3,
            )
            fits.append(
                dict(
                    case=case,
                    route=route,
                    fit_ids=union.tolist(),
                    features=NAMES + [f"size_{s}" for s in m.SIZES],
                    alpha=10.0,
                    scaler_mean=sc.mean_.tolist(),
                    scaler_sd=sc.scale_.tolist(),
                    coef=model.coef_.tolist(),
                    intercept=float(model.intercept_),
                )
            )
        for name, pp in predictions.items():
            assert np.isfinite(pp).all()
            for j, idx in enumerate(ev):
                rows.append(
                    dict(
                        model="crossfit_hgb_mean/" + name,
                        repeat=r,
                        fold=f,
                        dataset_idx=int(idx),
                        size_um=int(m.SIZES[b[idx]]),
                        case=case,
                        truth=float(truth[j, c]),
                        prediction=float(pp[j]),
                    )
                )
    # Exact fixed-setting same-union direct baseline from the initial HGB run.
    direct = pd.read_csv(HERE.parent / f"repeat{r}_fold{f}/direct_predictions.csv")
    direct = direct[
        (direct.subset == "evaluation")
        & (direct.model == "hgb direct mean outer_training")
    ].drop(columns="subset")
    assert set(direct.dataset_idx) == set(ev)
    df = pd.concat([pd.DataFrame(rows), direct], ignore_index=True)
    df.to_csv(folder / "predictions.csv", index=False, float_format="%.17g")
    np.savez_compressed(
        folder / "summary_features.npz",
        oof_features=oof,
        evaluation_features=features,
        outer_training_ids=union,
        evaluation_ids=ev,
        feature_names=np.array(NAMES),
        cases=np.array(m.CASES),
        oof_targets=target,
        evaluation_targets=truth,
    )
    np.savez_compressed(
        folder / "evaluation_grain_predictions.npz",
        dataset_idx=ev,
        offsets=np.r_[0, np.cumsum(a["grain_count"][ev])],
        pred_peak=pred,
        cases=np.array(m.CASES),
    )
    m.dump(folder / "inner_partitions.json", records)
    m.dump(folder / "stacker_fits.json", fits)
    m.dump(
        folder / "completion.json",
        dict(
            status="complete",
            all_union_oof_ids_once=True,
            all_splits_disjoint=True,
            outer_labels_used_only_for_scoring=True,
            repeat=r,
            fold=f,
            seconds=time.time() - start,
            script_sha256=m.sha(__file__),
            fixed_settings=CONFIG,
        ),
    )
    print("CROSSFIT FOLD COMPLETE", r, f, time.time() - start, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    run(args.repeat, args.fold, args.threads)
