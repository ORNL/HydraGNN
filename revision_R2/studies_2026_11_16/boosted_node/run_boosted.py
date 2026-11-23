"""Fixed mean and upper-quantile HGB node predictors on frozen SVE splits."""
import os

os.environ.update(
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
from pathlib import Path
import argparse, hashlib, json, time, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = HERE.parent / "rf_matched/inputs"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
CONFIG = dict(
    max_iter=100,
    learning_rate=0.07,
    max_depth=4,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
)
FAMILIES = ["hgb_mean", "hgb_q90"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def inputs():
    assert json.loads((INPUT / "validation.json").read_text())["status"] == "pass"
    return {p.stem: np.load(p, mmap_mode="r") for p in INPUT.glob("*.npy")}


def center_direct(X, b, centers, size_centers):
    xx = X.copy()
    for k in range(5):
        mask = b == k
        for first in [0, 64, 105, 137, 169]:
            xx[mask, first : first + 32] -= centers[k]
        xx[mask, 96:105] -= size_centers[k]
    return xx


def make_protocol():
    p = HERE / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text())["fixed_settings"] == CONFIG
        return
    dump(
        p,
        dict(
            date="2026-11-16",
            created_utc=datetime.now(timezone.utc).isoformat(),
            status="Declared before any boosted-node fit or outer scoring",
            families={
                "hgb_mean": "squared_error",
                "hgb_q90": "quantile pinball loss, quantile0.90",
            },
            fixed_settings=CONFIG,
            grain_inputs="Audited32 input-only grain descriptors with fitting-only size-specific centering plus5physical-size indicators.",
            targets="Raw grain peaks standardized per size and case on fitting grains only; four cases fitted separately.",
            weights="Inverse grain count of each fitting SVE, normalized to mean1; equal total weight per fitting SVE.",
            stopping="Never used for fitting/early stopping of HGB. Used only for scalar readout calibration/selection after forests frozen.",
            direct="Same HGB settings, squared-error loss,249 input-only SVE descriptors; fit separate fitting-only and fitting+stopping budgets.",
            direct_q90="Also a direct SVE quantile0.90 comparator at both budgets; distinct quantile endpoint, not a conditional mean.",
            selection="No outer-label tuning or selection; all fixed arms and routes reported.",
            endpoint="One numerical maximum per evaluated SVE; conditional node quantiles are intermediate predictions and may overestimate maxima without calibration.",
            splits="Original two five-fold SVE splits; all grains of an SVE remain in its split.",
            normalization="Fitting-only feature centers and target location/scales; direct preprocessing uses only its stated label budget.",
            docs="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
            runtime_sklearn=sklearn.__version__,
            input_validation_sha256=sha(INPUT / "validation.json"),
            partitions_sha256=sha(BASE / "partitions.json"),
        ),
    )


def export(a, ids, pred, direct, r, f, family, subset):
    counts = a["grain_count"][ids]
    off = np.r_[0, np.cumsum(counts)]
    ix = np.concatenate([np.arange(a["offsets"][i], a["offsets"][i + 1]) for i in ids])
    y = a["grain_truth"][ix]
    b = a["size_bucket"][ids]
    size = SIZES[b]
    assert pred.shape == y.shape and np.isfinite(pred).all()
    for j, idx in enumerate(ids):
        assert np.array_equal(y[off[j] : off[j + 1]].max(0), a["sve_truth"][idx])
    path = (
        HERE
        / family
        / "data"
        / f'grain_predictions_{"stopping_" if subset=="stopping" else ""}repeat{r}_fold{f}.npz'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cases=np.array(CASES),
        repeat=r,
        fold=f,
        subset=subset,
        dataset_idx=ids,
        size_um=size,
        size_bucket=b,
        grain_count=counts,
        offsets=off,
        grain_dataset_idx=np.repeat(ids, counts),
        grain_id=np.concatenate([np.arange(n, dtype=np.int32) for n in counts]),
        grain_size_um=np.repeat(size, counts),
        pred_peak=pred,
        true_peak=y,
        y_true=a["sve_truth"][ids],
        direct_prediction=direct,
    )
    dump(
        path.with_suffix(".json"),
        dict(
            status="pass",
            n_sves=len(ids),
            n_grains=len(y),
            finite=True,
            true_grain_maximum_equals_sve_target=True,
            sha256=sha(path),
            direct_definition="Fitting-only squared-error direct HGB; safe held-out stopping predictions.",
        ),
    )


def run_fold(r, f, threads):
    make_protocol()
    a = inputs()
    folder = HERE / f"repeat{r}_fold{f}"
    folder.mkdir(exist_ok=True)
    if (folder / "completion.json").exists():
        print("COMPLETE EXISTS", r, f, flush=True)
        return
    part = json.loads((BASE / f"partition_r{r}_f{f}.json").read_text())
    seed = part["initialization_seed"]
    fit = np.array(part["fitting"])
    stop = np.array(part["stopping"])
    ev = np.array(part["evaluation"])
    union = np.sort(np.r_[fit, stop])
    infer = np.r_[stop, ev]
    assert (
        not set(fit) & set(stop)
        and not set(union) & set(ev)
        and set(np.r_[union, ev]) == set(range(940))
    )
    b = a["size_bucket"]
    gb = a["grain_size_bucket"]
    gid = a["grain_dataset_idx"]
    gfit = np.isin(gid, fit)
    gi = np.concatenate(
        [np.arange(a["offsets"][i], a["offsets"][i + 1]) for i in infer]
    )
    centers = np.stack(
        [
            a["grain_sum32"][fit[b[fit] == k]].sum(0)
            / a["grain_count"][fit[b[fit] == k]].sum()
            for k in range(5)
        ]
    )
    means = np.stack([a["grain_truth"][gfit & (gb == k)].mean(0) for k in range(5)])
    sd = np.stack([a["grain_truth"][gfit & (gb == k)].std(0, ddof=1) for k in range(5)])
    assert (sd > 0).all()
    X = np.column_stack(
        [a["raw32"][gfit] - centers[gb[gfit]], np.eye(5)[gb[gfit]]]
    ).astype(np.float32)
    Z = np.column_stack([a["raw32"][gi] - centers[gb[gi]], np.eye(5)[gb[gi]]]).astype(
        np.float32
    )
    target = (a["grain_truth"][gfit] - means[gb[gfit]]) / sd[gb[gfit]]
    w = 1 / a["grain_count"][gid[gfit]].astype(float)
    w /= w.mean()
    totals = np.bincount(gid[gfit], weights=w, minlength=940)[fit]
    assert np.allclose(totals, totals[0], rtol=1e-12)
    clock = time.perf_counter()
    times = []
    preds = {}
    for family in FAMILIES:
        pred = np.empty((len(gi), 4))
        for c, case in enumerate(CASES):
            ppath = folder / f"{family}_{case}_predictions.npy"
            mpath = folder / f"{family}_{case}.joblib"
            if ppath.exists() and mpath.exists():
                pred[:, c] = np.load(ppath)
                continue
            kwargs = (
                {"loss": "squared_error"}
                if family == "hgb_mean"
                else {"loss": "quantile", "quantile": 0.9}
            )
            model = HistGradientBoostingRegressor(**CONFIG, **kwargs, random_state=seed)
            t = time.perf_counter()
            with threadpool_limits(limits=threads):
                model.fit(X, target[:, c], sample_weight=w)
                pp = model.predict(Z) * sd[gb[gi], c] + means[gb[gi], c]
            assert np.isfinite(pp).all()
            pred[:, c] = pp
            np.save(ppath, pp)
            joblib.dump(model, mpath, compress=3)
            seconds = time.perf_counter() - t
            times.append(dict(family=family, case=case, seconds=seconds))
            dump(folder / "timings.json", times)
            print("NODE", r, f, family, case, round(seconds, 2), flush=True)
        preds[family] = pred
    del X, Z, target
    direct = {}
    rows = []
    records = []
    for budget, tr in [("fitting", fit), ("outer_training", union)]:
        mu = np.stack([a["sve_truth"][tr[b[tr] == k]].mean(0) for k in range(5)])
        var = np.stack(
            [a["sve_truth"][tr[b[tr] == k]].var(0, ddof=1) for k in range(5)]
        )
        ss = np.sqrt(
            np.maximum(np.maximum(var, 1e-6 * a["sve_truth"][tr].var(0, ddof=1)), 1e-30)
        )
        xc = np.stack(
            [
                a["grain_sum32"][tr[b[tr] == k]].sum(0)
                / a["grain_count"][tr[b[tr] == k]].sum()
                for k in range(5)
            ]
        )
        sc = np.stack([a["size9"][tr[b[tr] == k]].mean(0) for k in range(5)])
        xx = center_direct(a["raw_extreme249"], b, xc, sc)
        for endpoint in ["mean", "q90"]:
            d = np.empty((len(infer), 4))
            for c, case in enumerate(CASES):
                kwargs = (
                    {"loss": "squared_error"}
                    if endpoint == "mean"
                    else {"loss": "quantile", "quantile": 0.9}
                )
                m = HistGradientBoostingRegressor(**CONFIG, **kwargs, random_state=seed)
                with threadpool_limits(limits=threads):
                    m.fit(xx[tr], (a["sve_truth"][tr, c] - mu[b[tr], c]) / ss[b[tr], c])
                    d[:, c] = m.predict(xx[infer]) * ss[b[infer], c] + mu[b[infer], c]
                joblib.dump(
                    m, folder / f"direct_{endpoint}_{budget}_{case}.joblib", compress=3
                )
            assert np.isfinite(d).all()
            direct[(endpoint, budget)] = d
            for j, idx in enumerate(infer):
                for c, case in enumerate(CASES):
                    rows.append(
                        dict(
                            model=f"hgb direct {endpoint} {budget}",
                            repeat=r,
                            fold=f,
                            dataset_idx=int(idx),
                            size_um=int(SIZES[b[idx]]),
                            case=case,
                            truth=float(a["sve_truth"][idx, c]),
                            prediction=float(d[j, c]),
                            subset="stopping" if j < len(stop) else "evaluation",
                        )
                    )
        records.append(
            dict(
                budget=budget,
                ids=tr.tolist(),
                target_mean=mu.tolist(),
                target_sd=ss.tolist(),
                grain_feature_centers=xc.tolist(),
                size_centers=sc.tolist(),
            )
        )
    ns = int(a["grain_count"][stop].sum())
    for family, pred in preds.items():
        export(
            a,
            stop,
            pred[:ns],
            direct[("mean", "fitting")][: len(stop)],
            r,
            f,
            family,
            "stopping",
        )
        export(
            a,
            ev,
            pred[ns:],
            direct[("mean", "fitting")][len(stop) :],
            r,
            f,
            family,
            "evaluation",
        )
    pd.DataFrame(rows).to_csv(
        folder / "direct_predictions.csv", index=False, float_format="%.17g"
    )
    dump(
        folder / "fitted_preprocessing.json",
        dict(
            grain_mean=means.tolist(),
            grain_sd=sd.tolist(),
            grain_feature_center=centers.tolist(),
            weights="1/grain_count, mean1",
            equal_total_weight_per_sve=True,
            direct=records,
        ),
    )
    dump(
        folder / "completion.json",
        dict(
            status="complete",
            repeat=r,
            fold=f,
            seed=seed,
            elapsed_seconds=time.perf_counter() - clock,
            fit_ids=fit.tolist(),
            stop_ids=stop.tolist(),
            evaluation_ids=ev.tolist(),
            n_grains_fit=int(gfit.sum()),
            all_predictions_finite=True,
            model_settings=CONFIG,
            script_sha256=sha(__file__),
            protocol_sha256=sha(HERE / "protocol.json"),
            partition_sha256=sha(BASE / f"partition_r{r}_f{f}.json"),
            runtime_sklearn=sklearn.__version__,
        ),
    )
    print(
        "FOLD COMPLETE",
        r,
        f,
        "seconds",
        round(time.perf_counter() - clock, 1),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    run_fold(args.repeat, args.fold, args.threads)
