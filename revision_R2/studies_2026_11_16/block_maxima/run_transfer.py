"""Prediction-only transfer of the audited aluminum block-maxima MLE recipe."""
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import argparse, concurrent.futures, hashlib, importlib.util, json, time, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import gumbel_r

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEW = HERE.parent
OLD = ROOT / "revision_R2/studies_2026_11_15"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
SOURCE = (
    ROOT
    / "revision_final/working/aluminum_160_audit_2026_11_02/block_maxima/test_uncensored_mle.py"
)
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
BLOCKS = [8, 16, 32]
ENCODERS = {
    "original_gnn": OLD / "data",
    "rf": NEW / "rf_matched/rf/data",
    "raw_sve": NEW / "neural_tail/raw_sve/data",
    "hgb_mean": NEW / "boosted_node/hgb_mean/data",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, x):
    Path(path).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def partition_maps(values, idx, size, nparts=5):
    x = np.sort(np.asarray(values, float))
    N = len(x)
    assert N > 0 and np.isfinite(x).all()
    rows = []
    for p in range(nparts):
        order = np.random.default_rng(
            np.random.SeedSequence([101, int(size), int(idx), p])
        ).permutation(N)
        shuffled = x[order]
        for B in BLOCKS:
            m = N // B
            loc = scale = np.nan
            fallback = ""
            pred = float(x[-1])
            nwarn = 0
            if m < 1:
                fallback = "fewer_grains_than_blocks"
            else:
                maxima = np.sort(shuffled[: B * m].reshape(B, m).max(1))
                shift = float(maxima.mean())
                unit = float(maxima.std(ddof=1))
                if not np.isfinite(unit) or unit <= 0:
                    fallback = "constant_block_maxima"
                else:
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            a, b = gumbel_r.fit((maxima - shift) / unit)
                        nwarn = len(caught)
                        loc = float(shift + unit * a)
                        scale = float(unit * b)
                        candidate = float(
                            loc + scale * (np.log(N / m) + np.euler_gamma)
                        )
                        if np.isfinite(candidate) and np.isfinite(scale) and scale > 0:
                            pred = candidate
                        else:
                            fallback = "nonfinite_or_nonpositive_mle"
                    except Exception as exc:
                        fallback = type(exc).__name__
            rows.append(
                dict(
                    partition=p,
                    B=B,
                    N=N,
                    block_size=m,
                    remainder=N - B * m,
                    location=loc,
                    scale=scale,
                    prediction=pred,
                    fallback=fallback,
                    n_warnings=nwarn,
                    predicted_maximum=float(x[-1]),
                )
            )
    return rows


def extract_fold(job):
    encoder, r, f, nparts = job
    folder = HERE / "jobs" / encoder / f"r{r}_f{f}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "completion.json").exists():
        return json.loads((folder / "completion.json").read_text())
    rows = []
    ledger = []
    sources = {}
    started = time.perf_counter()
    for subset, prefix in [("stopping", "stopping_"), ("evaluation", "")]:
        path = ENCODERS[encoder] / f"grain_predictions_{prefix}repeat{r}_fold{f}.npz"
        sources[str(path)] = sha(path)
        with np.load(path) as archive:
            z = {k: archive[k] for k in archive.files}
        assert list(z["cases"]) == CASES
        for i, idx in enumerate(z["dataset_idx"]):
            lo, hi = z["offsets"][i : i + 2]
            N = int(z["grain_count"][i])
            assert hi - lo == N
            for c, case in enumerate(CASES):
                values = z["pred_peak"][lo:hi, c]
                truth = float(z["y_true"][i, c])
                assert np.isclose(
                    z["true_peak"][lo:hi, c].max(), truth, rtol=1e-12, atol=1e-14
                )
                key = dict(
                    encoder=encoder,
                    repeat=r,
                    fold=f,
                    subset=subset,
                    dataset_idx=int(idx),
                    size_um=int(z["size_um"][i]),
                    case=case,
                )
                calculations = partition_maps(
                    values, int(idx), int(z["size_um"][i]), nparts
                )
                for row in calculations:
                    ledger.append({**key, **row})
                feature = {
                    **key,
                    "truth": truth,
                    "log_count": np.log(N),
                    "maximum": float(values.max()),
                }
                for B in BLOCKS:
                    selected = [q for q in calculations if q["B"] == B]
                    feature[f"B{B}"] = float(
                        np.mean([q["prediction"] for q in selected])
                    )
                    feature[f"B{B}_partition_sd"] = float(
                        np.std([q["prediction"] for q in selected], ddof=1)
                    )
                    feature[f"B{B}_fallback_partitions"] = sum(
                        bool(q["fallback"]) for q in selected
                    )
                rows.append(feature)
    pd.DataFrame(rows).to_csv(
        folder / "features.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(ledger).to_csv(folder / "partition_fits.csv.gz", index=False)
    report = dict(
        status="complete",
        encoder=encoder,
        repeat=r,
        fold=f,
        partition_fits=len(ledger),
        feature_rows=len(rows),
        fallback_partitions=sum(bool(q["fallback"]) for q in ledger),
        seconds=time.perf_counter() - started,
        sources=sources,
    )
    dump(folder / "completion.json", report)
    return report


def benchmark():
    path = ENCODERS["original_gnn"] / "grain_predictions_repeat0_fold0.npz"
    with np.load(path) as archive:
        z = {k: archive[k] for k in archive.files}
    t = time.perf_counter()
    nfit = 0
    checks = []
    for i in range(8):
        for c in range(4):
            x = z["pred_peak"][z["offsets"][i] : z["offsets"][i + 1], c]
            results = partition_maps(
                x, int(z["dataset_idx"][i]), int(z["size_um"][i]), 5
            )
            nfit += len(results)
            # Independent literal transcription of aluminum block16 partition0.
            xx = np.sort(x)
            N = len(x)
            B = 16
            m = N // B
            if m:
                order = np.random.default_rng(
                    np.random.SeedSequence(
                        [101, int(z["size_um"][i]), int(z["dataset_idx"][i]), 0]
                    )
                ).permutation(N)
                v = np.sort(xx[order][: B * m].reshape(B, m).max(1))
                u = v.mean()
                sd = v.std(ddof=1)
                if sd > 0:
                    a, b = gumbel_r.fit((v - u) / sd)
                    expected = u + sd * a + sd * b * (np.log(N / m) + np.euler_gamma)
                    got = next(
                        q["prediction"]
                        for q in results
                        if q["B"] == 16 and q["partition"] == 0
                    )
                    assert np.isclose(got, expected, rtol=1e-13, atol=1e-15)
                    checks.append(abs(got - expected))
    elapsed = time.perf_counter() - t
    # At most4encoders *2subsets totaling~2630SVE observations *4cases *15fits.
    projected = 4 * 2630 * 4 * 15 * elapsed / nfit / 8
    nparts = 5 if projected < 130 else 3
    for x in [np.ones(10), np.ones(40), np.arange(4, dtype=float)]:
        assert all(
            np.isfinite(q["prediction"]) for q in partition_maps(x, 0, 125, nparts)
        )
    report = dict(
        seconds=elapsed,
        n_partition_fits=nfit,
        projected_fit_seconds_eight_workers=projected,
        selected_partition_count=nparts,
        timing_only_selection=True,
        aluminum_literal_parity_checks=len(checks),
        max_parity_error=max(checks),
        constant_sparse_fallback_pass=True,
    )
    dump(HERE / "benchmark.json", report)
    print(json.dumps(report), flush=True)
    return nparts


def analyze():
    s = module("shared_readouts", NEW / "bridge_search/run_search.py")
    parts = json.loads((BASE / "partitions.json").read_text())
    with np.load(BASE / "regressors/raw_features940.npz") as z:
        truth = z["raw_truth"]
        bucket = z["size_bucket"]
    rows = []
    fits = []
    fallbacks = []
    for encoder in ENCODERS:
        features = pd.concat(
            [
                pd.read_csv(
                    HERE
                    / "jobs"
                    / encoder
                    / f'r{p["repeat"]}_f{p["fold"]}'
                    / "features.csv"
                )
                for p in parts
            ],
            ignore_index=True,
        )
        features.to_csv(HERE / f"{encoder}_features.csv.gz", index=False)
        for part in parts:
            r, f = part["repeat"], part["fold"]
            ids = np.array(part["fitting"])
            means = np.stack([truth[ids[bucket[ids] == b]].mean(0) for b in range(5)])
            scales = np.stack(
                [truth[ids[bucket[ids] == b]].std(0, ddof=1) for b in range(5)]
            )
            for c, case in enumerate(CASES):
                cell = features[
                    (features["repeat"] == r)
                    & (features.fold == f)
                    & (features.case == case)
                ]
                tr = cell[cell.subset == "stopping"].sort_values("dataset_idx")
                ev = cell[cell.subset == "evaluation"].sort_values("dataset_idx")
                assert set(tr.dataset_idx) == set(part["stopping"]) and set(
                    ev.dataset_idx
                ) == set(part["evaluation"])
                bt = np.searchsorted(SIZES, tr.size_um)
                bv = np.searchsorted(SIZES, ev.size_um)
                for B in BLOCKS:
                    for mode in ["identity", "affine", "count", "within_size"]:
                        name = f"block{B}/" + (
                            "raw" if mode == "identity" else mode + "_ridge10"
                        )
                        spec = dict(features=[f"B{B}"], mode=mode, alpha=10)
                        prediction, record = s.fit_predict(
                            spec,
                            tr,
                            ev,
                            tr.truth.to_numpy(),
                            bt,
                            bv,
                            means[:, c],
                            scales[:, c],
                        )
                        fits.append(
                            dict(
                                encoder=encoder,
                                repeat=r,
                                fold=f,
                                case=case,
                                model=name,
                                calibration_ids=tr.dataset_idx.tolist()
                                if mode != "identity"
                                else [],
                                **record,
                            )
                        )
                        for j, idx in enumerate(ev.dataset_idx):
                            rows.append(
                                dict(
                                    encoder=encoder,
                                    model=name,
                                    repeat=r,
                                    fold=f,
                                    dataset_idx=int(idx),
                                    size_um=int(ev.size_um.iloc[j]),
                                    case=case,
                                    truth=float(ev.truth.iloc[j]),
                                    prediction=float(prediction[j]),
                                )
                            )
                for B in BLOCKS:
                    fallbacks.append(
                        dict(
                            encoder=encoder,
                            repeat=r,
                            fold=f,
                            case=case,
                            B=B,
                            evaluation_sves_with_any_fallback=int(
                                (ev[f"B{B}_fallback_partitions"] > 0).sum()
                            ),
                            evaluation_fallback_partitions=int(
                                ev[f"B{B}_fallback_partitions"].sum()
                            ),
                        )
                    )
    pred = pd.DataFrame(rows)
    assert np.isfinite(pred.prediction).all()
    assert not pred.duplicated(
        ["encoder", "model", "repeat", "dataset_idx", "case"]
    ).any()
    assert pred.groupby(["encoder", "model"]).size().eq(7520).all()
    pred.to_csv(HERE / "predictions.csv.gz", index=False)
    # Identical equal-maximum-label-budget references for every encoder.
    refs = pd.read_csv(OLD / "label_matched_trees/predictions_long.csv")
    size = pd.read_csv(OLD / "baselines/outer_training_size_baseline_predictions.csv")
    size = size[size.model == "Outer-training size mean"]
    newer = pd.read_csv(
        NEW / "rf_matched/rf/primary/predictions_long.csv", low_memory=False
    )
    newer = newer[newer.model == "rf direct outer_training"]
    refs = pd.concat([refs, size, newer], ignore_index=True)[
        [
            "model",
            "repeat",
            "fold",
            "dataset_idx",
            "size_um",
            "case",
            "truth",
            "prediction",
        ]
    ]
    refs.to_csv(HERE / "matched_direct_and_size_controls.csv.gz", index=False)
    allrows = [pred]
    for encoder in ENCODERS:
        rr = refs.copy()
        rr["encoder"] = encoder
        allrows.append(rr)
    comparison = pd.concat(allrows, ignore_index=True)
    metrics = []
    for (encoder, model, r, case), g in comparison.groupby(
        ["encoder", "model", "repeat", "case"]
    ):
        for size in ["pooled", *SIZES.tolist()]:
            q = g if size == "pooled" else g[g.size_um == size]
            metrics.append(
                dict(
                    encoder=encoder,
                    model=model,
                    repeat=r,
                    case=case,
                    size_um=str(size),
                    **s.score(q.truth.to_numpy(), q.prediction.to_numpy()),
                )
            )
    mm = pd.DataFrame(metrics)
    mm.to_csv(HERE / "metrics_by_repeat.csv", index=False)
    summary = mm.groupby(["encoder", "model", "case", "size_um"])[
        ["r2", "rmse", "mae", "bias", "pearson", "spearman"]
    ].agg(["mean", "min", "max"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(HERE / "metrics_summary.csv", index=False)
    pd.DataFrame(fallbacks).to_csv(HERE / "fallback_summary.csv", index=False)
    dump(HERE / "calibration_fits.json", fits)
    dump(
        HERE / "completion.json",
        dict(
            status="complete",
            encoders=list(ENCODERS),
            models_per_encoder=12,
            rows=len(pred),
            finite=True,
            exact_7520_keys_per_encoder_model=True,
            script_sha256=sha(__file__),
            common_readout_sha256=sha(NEW / "bridge_search/run_search.py"),
            no_outer_labels_in_scalar_fitting=True,
            per_sve_mle_uses_predicted_values_only=True,
            protocol_sha256=sha(HERE / "protocol.json"),
        ),
    )
    print(
        summary[
            (summary.size_um == "pooled") & (summary.model == "block16/affine_ridge10")
        ][["encoder", "case", "r2_mean", "rmse_mean"]].to_string(index=False),
        flush=True,
    )


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    nparts = benchmark()
    started = time.perf_counter()
    dump(
        HERE / "protocol.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source=str(SOURCE),
            source_sha256=sha(SOURCE),
            encoders={k: str(v) for k, v in ENCODERS.items()},
            blocks=BLOCKS,
            n_partitions=nparts,
            partition_seed="SeedSequence([101,size_um,canonical_dataset_idx,partition]); same across encoder/repeat/case after sorting predicted values",
            transfer="Equal random nonspatial blocks m=floor(N/B), discard random remainder, standardized uncensored Gumbel MLE, loc+scale*(log(N/m)+EulerGamma); average separate partition predictions, never pool correlated partitions.",
            changes_from_aluminum="Predicted rather than observed grain fields; different material, SVE sizes, grain counts and dependence. B8/16/32 replace historical16/32/64. No censoring or target-removal claim. Original aluminum accuracy does not transfer automatically.",
            fallback="Predicted maximum for insufficient grains, constant block maxima or failed/nonfinite MLE; retain all arms and failure ledger.",
            learned_corrections="Fixed Ridge(alpha10) scalar affine, count-interaction and within-size residual maps; coefficients/feature scaling fitted only on stopping SVEs, target means/SDs fitting-only. No model/threshold selection.",
            estimator_parameters="Per-SVE MLE parameters use its predicted grain field only, including for outer SVEs; no observed labels. This is feature construction, distinct from learned response calibration.",
            reference_budget="Identical fitting+stopping SVE-maximum-label-budget mean, existing RF/HGB direct models and new union-fit RF direct control for every encoder.",
            scope="Exploratory saved-prediction extension on reused940SVEs; random blocks do not establish spatial independence or Gumbel max attraction.",
        ),
    )
    parts = json.loads((BASE / "partitions.json").read_text())
    jobs = [(e, p["repeat"], p["fold"], nparts) for e in ENCODERS for p in parts]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for record in pool.map(extract_fold, jobs):
            print(
                "EXTRACTED",
                record["encoder"],
                record["repeat"],
                record["fold"],
                round(record["seconds"], 1),
                flush=True,
            )
    analyze()
    print(
        "BLOCK TRANSFER COMPLETE",
        round(time.perf_counter() - started, 2),
        "seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
