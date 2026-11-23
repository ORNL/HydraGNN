"""Predeclared size/count controls on the frozen 940-SVE partitions."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
OUT = Path(__file__).resolve().parent
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-stopping", action="store_true")
    args = parser.parse_args()
    prefix = "Outer-training " if args.include_stopping else ""
    data = np.load(BASE / "regressors/raw_features940.npz", allow_pickle=False)
    y = data["raw_truth"]
    bucket = data["size_bucket"]
    logn = np.log(data["node_counts"].astype(float))
    parts = json.loads((BASE / "partitions.json").read_text())
    assert y.shape == (940, 4) and np.array_equal(data["dataset_idx"], np.arange(940))
    rows, fitted = [], []
    for part in parts:
        tr = np.array(
            part["fitting"] + (part["stopping"] if args.include_stopping else [])
        )
        te = np.array(part["evaluation"])
        assert not set(tr) & set(te)
        predictions = {}
        if args.include_stopping:
            means = np.stack([y[tr[bucket[tr] == b]].mean(0) for b in range(5)])
            predictions["Outer-training size mean"] = means[bucket[te]]
        for degree in (1, 2):
            x = np.column_stack([logn ** k for k in range(1, degree + 1)])
            scaler = StandardScaler().fit(x[tr])
            model = LinearRegression().fit(scaler.transform(x[tr]), y[tr])
            name = prefix + (
                "Log count linear" if degree == 1 else "Log count quadratic"
            )
            predictions[name] = model.predict(scaler.transform(x[te]))
            fitted.append(
                dict(
                    model=name,
                    repeat=part["repeat"],
                    fold=part["fold"],
                    x_mean=scaler.mean_.tolist(),
                    x_scale=scaler.scale_.tolist(),
                    coefficients=model.coef_.tolist(),
                    intercept=model.intercept_.tolist(),
                )
            )
        ridge_name = prefix + "Within-size count ridge"
        predictions[ridge_name] = np.empty((len(te), 4))
        for b in range(5):
            ti = tr[bucket[tr] == b]
            mask = bucket[te] == b
            scaler = StandardScaler().fit(logn[ti, None])
            model = Ridge(alpha=10.0).fit(scaler.transform(logn[ti, None]), y[ti])
            predictions[ridge_name][mask] = model.predict(
                scaler.transform(logn[te[mask], None])
            )
            fitted.append(
                dict(
                    model=ridge_name,
                    repeat=part["repeat"],
                    fold=part["fold"],
                    size_um=int(SIZES[b]),
                    n_fit=len(ti),
                    alpha=10.0,
                    x_mean=scaler.mean_.tolist(),
                    x_scale=scaler.scale_.tolist(),
                    coefficients=model.coef_.tolist(),
                    intercept=model.intercept_.tolist(),
                )
            )
        for name, pred in predictions.items():
            assert np.isfinite(pred).all()
            for i, idx in enumerate(te):
                for c, case in enumerate(CASES):
                    rows.append(
                        dict(
                            model=name,
                            repeat=part["repeat"],
                            fold=part["fold"],
                            dataset_idx=int(idx),
                            size_um=int(SIZES[bucket[idx]]),
                            case=case,
                            truth=float(y[idx, c]),
                            prediction=float(pred[i, c]),
                            target="SVE maximum",
                            units="dimensionless" if case.endswith("fs") else "MPa",
                        )
                    )
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["model", "repeat", "dataset_idx", "case"]).any()
    stem = "outer_training_size" if args.include_stopping else "size"
    frame.to_csv(OUT / (stem + "_baseline_predictions.csv"), index=False)
    (OUT / (stem + "_fitted_models.json")).write_text(
        json.dumps(fitted, indent=2) + "\n"
    )
    report = dict(
        status="complete",
        rows=len(frame),
        n_models=int(frame.model.nunique()),
        n_unique_sves=940,
        n_repeats=2,
        fitting_ids_only=not args.include_stopping,
        stopping_ids_included=args.include_stopping,
        outer_training_ids_only=True,
        no_model_selection=True,
        partitions_sha256=hashlib.sha256(
            (BASE / "partitions.json").read_bytes()
        ).hexdigest(),
    )
    (OUT / (stem + "_baseline_report.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
