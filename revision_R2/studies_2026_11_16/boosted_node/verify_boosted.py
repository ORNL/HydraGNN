"""Reload models and independently verify exported predictions and identities."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import json, hashlib, joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = HERE.parent / "rf_matched/inputs"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    validation = json.loads((INPUT / "validation.json").read_text())
    checks = []
    for name, digest in validation["artifacts"].items():
        assert sha(INPUT / name) == digest, name
    a = {p.stem: np.load(p, mmap_mode="r") for p in INPUT.glob("*.npy")}
    allkeys = []
    maxdelta = 0.0
    models = 0
    for r in range(2):
        for f in range(5):
            folder = HERE / f"repeat{r}_fold{f}"
            completion = json.loads((folder / "completion.json").read_text())
            assert completion["status"] == "complete"
            part = json.loads((BASE / f"partition_r{r}_f{f}.json").read_text())
            stat = json.loads((folder / "fitted_preprocessing.json").read_text())
            for key, oldkey in [
                ("fit_ids", "fitting"),
                ("stop_ids", "stopping"),
                ("evaluation_ids", "evaluation"),
            ]:
                assert completion[key] == part[oldkey]
            assert not set(part["evaluation"]) & (
                set(part["fitting"]) | set(part["stopping"])
            )
            fi = np.array(part["fitting"])
            gb = a["grain_size_bucket"]
            gfit = np.isin(a["grain_dataset_idx"], fi)
            mu = np.stack(
                [a["grain_truth"][gfit & (gb == b)].mean(0) for b in range(5)]
            )
            sd = np.stack(
                [a["grain_truth"][gfit & (gb == b)].std(0, ddof=1) for b in range(5)]
            )
            assert np.array_equal(mu, stat["grain_mean"]) and np.array_equal(
                sd, stat["grain_sd"]
            )
            counts = a["grain_count"]
            w = 1 / counts[a["grain_dataset_idx"][gfit]].astype(float)
            w /= w.mean()
            totals = np.bincount(
                a["grain_dataset_idx"][gfit], weights=w, minlength=940
            )[fi]
            assert np.allclose(totals, totals[0], rtol=1e-12)
            for family in ["hgb_mean", "hgb_q90"]:
                for subset, prefix in [("stopping", "stopping_"), ("evaluation", "")]:
                    path = (
                        HERE
                        / family
                        / "data"
                        / f"grain_predictions_{prefix}repeat{r}_fold{f}.npz"
                    )
                    with np.load(path) as z:
                        zz = {k: z[k] for k in z.files}
                    ids = zz["dataset_idx"]
                    assert set(ids) == set(part[subset])
                    assert list(zz["cases"]) == CASES
                    assert (
                        np.isfinite(zz["pred_peak"]).all()
                        and np.isfinite(zz["direct_prediction"]).all()
                    )
                    for j, idx in enumerate(ids):
                        aa, bb = zz["offsets"][j : j + 2]
                        original = slice(a["offsets"][idx], a["offsets"][idx + 1])
                        assert np.array_equal(
                            zz["true_peak"][aa:bb], a["grain_truth"][original]
                        )
                        assert np.array_equal(zz["grain_id"][aa:bb], np.arange(bb - aa))
                        assert np.array_equal(
                            zz["true_peak"][aa:bb].max(0), zz["y_true"][j]
                        )
                        assert np.array_equal(zz["y_true"][j], a["sve_truth"][idx])
                        if subset == "evaluation":
                            allkeys.append((family, r, int(idx)))
                    # Independently reconstruct100input rows from raw arrays.
                    loc = np.linspace(0, len(zz["pred_peak"]) - 1, 100, dtype=int)
                    sve = zz["grain_dataset_idx"][loc]
                    original_rows = a["offsets"][sve] + zz["grain_id"][loc]
                    b = a["size_bucket"][sve]
                    x = np.column_stack(
                        [
                            a["raw32"][original_rows]
                            - np.array(stat["grain_feature_center"])[b],
                            np.eye(5)[b],
                        ]
                    ).astype(np.float32)
                    for c, case in enumerate(CASES):
                        model = joblib.load(folder / f"{family}_{case}.joblib")
                        assert model.max_iter == 100 and model.early_stopping is False
                        assert model.loss == (
                            "squared_error" if family == "hgb_mean" else "quantile"
                        )
                        if family == "hgb_q90":
                            assert model.quantile == 0.9
                        with threadpool_limits(1):
                            p = model.predict(x) * sd[b, c] + mu[b, c]
                        delta = float(abs(p - zz["pred_peak"][loc, c]).max())
                        maxdelta = max(maxdelta, delta)
                        assert np.allclose(
                            p, zz["pred_peak"][loc, c], rtol=1e-12, atol=1e-12
                        )
                        models += 1
                    checks.append(
                        dict(
                            family=family,
                            repeat=r,
                            fold=f,
                            subset=subset,
                            n_sves=len(ids),
                            n_grains=len(zz["pred_peak"]),
                            export_sha256=sha(path),
                        )
                    )
    assert len(allkeys) == 3760 and len(set(allkeys)) == 3760
    result = dict(
        status="pass",
        input_arrays_unchanged=True,
        all_key_occurrences=3760,
        raw_truth_identity_exact=True,
        fitting_only_preprocessing_exact=True,
        equal_sve_weights_verified=True,
        model_reload_predictions_checked=models * 100,
        max_reload_prediction_difference=maxdelta,
        archive_checks=checks,
    )
    (HERE / "independent_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "archive_checks"}, indent=2)
    )


if __name__ == "__main__":
    main()
