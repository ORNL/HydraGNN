"""Frozen scalar calibrators trained on inner stopping predictions only.

No evaluation labels enter the fitted models. The stopping set also selected
the original checkpoint, so it is not an independently validated calibration set.
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
STUDY = OUT.parent
ROOT = STUDY.parents[1]
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
FACTOR = np.sqrt(6.0) / np.pi


def summaries(data):
    n = len(data["dataset_idx"])
    feature = np.full((n, 4, 3), np.nan)
    maximum = np.full((n, 4), np.nan)
    selected = np.zeros((n, 4), dtype=int)
    for i in range(n):
        grain = data["pred_peak"][data["offsets"][i] : data["offsets"][i + 1]]
        for c in range(4):
            x = grain[:, c]
            x = x[np.isfinite(x)]
            if len(x) < 2:
                continue
            maximum[i, c] = x.max()
            q = np.clip(1 - np.mean(x > x.mean() + x.std(ddof=1)), 0, 0.99)
            tail = x[x > np.quantile(x, q)]
            selected[i, c] = len(tail)
            neff = data["grain_count"][i] * len(tail) / len(x)
            if len(tail) < 4 or neff < 2:
                continue
            sd = tail.std(ddof=1)
            if not sd > 0:
                continue
            feature[i, c] = [tail.mean(), sd, sd * np.log(neff)]
    return feature, maximum, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=STUDY / "data")
    parser.add_argument("--label", default="Graph")
    args = parser.parse_args()
    parts = json.loads((BASE / "partitions.json").read_text())
    raw = np.load(BASE / "regressors/raw_features940.npz", allow_pickle=False)
    all_y, all_b = raw["raw_truth"], raw["size_bucket"]
    rows, fitted, ledger, summary_rows = [], [], [], []
    for part in parts:
        r, f = part["repeat"], part["fold"]
        with np.load(
            args.data_dir / f"grain_predictions_stopping_repeat{r}_fold{f}.npz",
            allow_pickle=False,
        ) as archive:
            stop = {k: archive[k] for k in archive.files}
        with np.load(
            args.data_dir / f"grain_predictions_repeat{r}_fold{f}.npz",
            allow_pickle=False,
        ) as archive:
            test = {k: archive[k] for k in archive.files}
        assert np.array_equal(stop["cases"], CASES) and np.array_equal(
            test["cases"], CASES
        )
        assert set(stop["dataset_idx"]) == set(part["stopping"])
        assert set(test["dataset_idx"]) == set(part["evaluation"])
        assert not set(stop["dataset_idx"]) & set(test["dataset_idx"])
        train = np.array(part["fitting"])
        means = np.stack([all_y[train[all_b[train] == b]].mean(0) for b in range(5)])
        scales = np.stack(
            [all_y[train[all_b[train] == b]].std(0, ddof=1) for b in range(5)]
        )
        xs, ms, ks = summaries(stop)
        xt, mt, kt = summaries(test)
        bs, bt = stop["size_bucket"], test["size_bucket"]
        for c, case in enumerate(CASES):
            for subset, data, features, maxpred, kcount in [
                ("stopping", stop, xs, ms, ks),
                ("evaluation", test, xt, mt, kt),
            ]:
                for i, idx in enumerate(data["dataset_idx"]):
                    summary_rows.append(
                        dict(
                            model=args.label,
                            subset=subset,
                            repeat=r,
                            fold=f,
                            dataset_idx=int(idx),
                            case=case,
                            size_um=int(data["size_um"][i]),
                            selected_count=int(kcount[i, c]),
                            tail_mean=features[i, c, 0],
                            tail_sd=features[i, c, 1],
                            sd_log_count=features[i, c, 2],
                            max_predicted_grain=maxpred[i, c],
                        )
                    )
            for method in [
                "Affine direct calibration",
                "Affine maximum calibration",
                "Affine bridge calibration",
                "Tail-moment ridge",
                "Size-residual tail ridge",
            ]:
                if method == "Affine direct calibration":
                    a, b = (
                        stop["direct_prediction"][:, c, None],
                        test["direct_prediction"][:, c, None],
                    )
                elif method == "Affine maximum calibration":
                    a, b = ms[:, c, None], mt[:, c, None]
                elif method == "Affine bridge calibration":
                    a = (xs[:, c, 0] + FACTOR * xs[:, c, 2])[:, None]
                    b = (xt[:, c, 0] + FACTOR * xt[:, c, 2])[:, None]
                else:
                    a, b = xs[:, c].copy(), xt[:, c].copy()
                okfit = np.isfinite(a).all(1)
                oktest = np.isfinite(b).all(1)
                target = stop["y_true"][:, c].copy()
                centers = None
                residual = method == "Size-residual tail ridge"
                if residual:
                    centers = np.stack(
                        [
                            a[okfit & (bs == j)].mean(0)
                            if np.any(okfit & (bs == j))
                            else a[okfit].mean(0)
                            for j in range(5)
                        ]
                    )
                    a -= centers[bs]
                    b -= centers[bt]
                    target = (target - means[bs, c]) / scales[bs, c]
                assert okfit.sum() >= max(10, a.shape[1] + 2)
                scaler = StandardScaler().fit(a[okfit])
                estimator = (
                    Ridge(alpha=10.0) if "ridge" in method else LinearRegression()
                )
                estimator.fit(scaler.transform(a[okfit]), target[okfit])
                pred = means[bt, c].copy()
                pp = estimator.predict(scaler.transform(b[oktest]))
                if residual:
                    pp = means[bt[oktest], c] + scales[bt[oktest], c] * pp
                pred[oktest] = pp
                assert np.isfinite(pred).all()
                name = args.label + " " + method
                fitted.append(
                    dict(
                        model=name,
                        repeat=r,
                        fold=f,
                        case=case,
                        calibration_ids=stop["dataset_idx"][okfit].tolist(),
                        n_calibration=int(okfit.sum()),
                        x_mean=scaler.mean_.tolist(),
                        x_scale=scaler.scale_.tolist(),
                        coefficients=estimator.coef_.tolist(),
                        intercept=float(estimator.intercept_),
                        alpha=10.0 if "ridge" in method else None,
                        size_centers=None if centers is None else centers.tolist(),
                        target_mean_by_size=means[:, c].tolist(),
                        target_sd_by_size=scales[:, c].tolist(),
                    )
                )
                for i, idx in enumerate(test["dataset_idx"]):
                    rows.append(
                        dict(
                            model=name,
                            repeat=r,
                            fold=f,
                            dataset_idx=int(idx),
                            size_um=int(test["size_um"][i]),
                            case=case,
                            truth=float(test["y_true"][i, c]),
                            prediction=float(pred[i]),
                            target="SVE maximum",
                            units="dimensionless" if case.endswith("fs") else "MPa",
                        )
                    )
                    ledger.append(
                        dict(
                            model=name,
                            repeat=r,
                            fold=f,
                            dataset_idx=int(idx),
                            case=case,
                            fallback=bool(not oktest[i]),
                            n_calibration=int(okfit.sum()),
                        )
                    )
    slug = args.label.lower().replace(" ", "_")
    frame = pd.DataFrame(rows)
    assert len(frame) == 940 * 2 * 4 * 5
    assert not frame.duplicated(["model", "repeat", "dataset_idx", "case"]).any()
    frame.to_csv(OUT / f"{slug}_calibration_predictions.csv", index=False)
    pd.DataFrame(ledger).to_csv(OUT / f"{slug}_calibration_ledger.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / f"{slug}_tail_summaries.csv", index=False)
    (OUT / f"{slug}_calibration_fits.json").write_text(
        json.dumps(fitted, indent=2) + "\n"
    )
    report = dict(
        status="complete",
        label=args.label,
        n_rows=len(frame),
        n_models=5,
        no_evaluation_labels_used_for_fitting=True,
        stopping_reused_for_checkpoint_and_calibration=True,
        no_hyperparameter_selection=True,
        fallback_counts=pd.DataFrame(ledger).groupby("model").fallback.sum().to_dict(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / f"{slug}_calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
