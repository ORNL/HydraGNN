"""Fixed direct SVE-maximum comparators, with raw940 preparation separate from fits."""
from fip_release_paths import release_root
import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import textwrap
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
R = BASE.parent.parent
ROOT = Path(str(release_root() / ""))
OLD = ROOT / "examples/fip_major_revision_2026_10_04"
STUDY = ROOT / "examples/fip_unified/isaac_v10_t011_multiseed"
DATA = ROOT / "examples/cpfem_fip_data/outputs/full_data_v5.pt"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
LABELS = ["Random forest", "Histogram boosting", "Fitting-size mean"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def prepare():
    assert not (
        OUT / "raw_features940.npz"
    ).exists(), "Do not silently replace a prepared raw feature artifact."
    t = time.perf_counter()
    sys.path[:0] = [str(STUDY / "code"), str(ROOT)]
    import torch
    import train
    from dataset import X_COL_LAYOUT

    torch.set_num_threads(1)
    cfg = json.loads((STUDY / "results/t011_split42_seed42/config.json").read_text())
    ds_cfg = cfg["Recipe"]
    seed = 42
    text = (STUDY / "code/train.py").read_text()
    start = text.index("    split_seed = ds_cfg.get", text.index("def train_one"))
    stop = text.index("    train_data, val_data, test_data = fipds.build()", start)
    scope = {"cfg": cfg, "ds_cfg": ds_cfg, "seed": seed, "FIPDataset": train.FIPDataset}
    exec(textwrap.dedent(text[start:stop]), scope)
    ds = scope["fipds"]
    ds.pt_path = str(DATA)
    ds._load()
    assert len(ds._raw) == 940 and ds.n_input_features == 32
    identity = pd.read_csv(OLD / "provenance/dataset_lineage/source_identity_map.csv")
    identity = identity[identity.canonical_filtered_dataset_idx.notna()].copy()
    identity["dataset_idx"] = identity.canonical_filtered_dataset_idx.astype(int)
    identity = identity.set_index("dataset_idx").sort_index()
    fingerprints = pd.read_csv(OLD / "provenance/r0/graph_fingerprints.csv").set_index(
        "dataset_idx"
    )
    columns = []
    for block in ds.use_x_blocks:
        sl = X_COL_LAYOUT[block]
        for j in range(sl.stop - sl.start):
            columns.append(
                {
                    "selected_index": len(columns),
                    "name": block + "_" + str(j),
                    "block": block,
                    "expanded_source_index": sl.start + j,
                }
            )
    prior_schema = json.loads(
        (OLD / "provenance/grouped_baselines/feature_schema.json").read_text()
    )
    assert columns == prior_schema["grain32"]
    rows = []
    sizes = []
    counts = []
    grain_sums = []
    truths = []
    hashes = []
    for i, d in enumerate(ds._raw):
        n = d.num_nodes
        b = int(d.size_bucket[0])
        g = ds._select_x(d.x).numpy().astype(np.float64)
        assert g.shape == (n, 32) and np.isfinite(g).all()
        assert ah(d.x.numpy()) == fingerprints.loc[i, "expanded_x_sha256"]
        assert n == int(identity.loc[i, "n_grains"]) and SIZES[b] == int(
            identity.loc[i, "size_um"]
        )
        y = np.column_stack(
            [getattr(d, c + "_fip_dist")[:, 0].numpy() for c in CASES]
        ).astype(np.float64)
        for c in CASES:
            assert (
                ah(getattr(d, c + "_fip_dist")[:, 0].numpy())
                == fingerprints.loc[i, c + "_raw_peak_sha256"]
            )
        assert np.isfinite(y).all()
        oh = ds._graph_attr_raw(d).astype(float)
        assert oh.shape == (5,) and oh[b] == 1 and (oh.sum() == 1)
        sv = np.r_[SIZES[b], np.log(SIZES[b]), n, np.log(n), oh]
        sm = g[:, 4:16].max(1)
        v = g[:, 16]
        surf = g[:, 17]
        deg = g[:, 19]
        assert v.mean() > 0 and surf.mean() > 0 and (deg.mean() > 0)
        vr = v / v.mean()
        dr = deg / deg.mean()
        sr = surf / surf.mean()
        values = np.column_stack([sm, vr, sm * vr, sm * np.cbrt(vr), sm * dr, sm * sr])
        base = np.r_[g.mean(0), g.std(0, ddof=0), g.max(0), sv]
        extremes = np.r_[
            np.quantile(g, 0.9, axis=0),
            np.quantile(g, 0.99, axis=0),
            np.sort(g, axis=0)[-min(3, n) :].mean(0),
        ]
        physical = np.r_[
            values.mean(0),
            values.std(0, ddof=0),
            values.max(0),
            np.quantile(values, [0.5, 0.9, 0.95, 0.99], axis=0).ravel(),
            np.sort(values, axis=0)[-min(3, n) :].mean(0),
        ]
        rows.append(np.r_[base, extremes, physical])
        sizes.append(sv)
        counts.append(n)
        grain_sums.append(g.sum(0))
        truths.append(y.max(0))
        hashes.append(ah(g))
    X = np.stack(rows)
    y = np.stack(truths)
    sv = np.stack(sizes)
    sums = np.stack(grain_sums)
    counts = np.array(counts)
    assert X.shape == (940, 249) and y.shape == (940, 4) and (sv.shape == (940, 9))
    assert np.isfinite(X).all() and np.isfinite(y).all()
    raw = np.load(
        OLD / "results/grouped_baselines/raw_features_targets.npz", allow_pickle=False
    )
    assert np.array_equal(y, raw["maximum"]) and np.array_equal(sv, raw["size"])
    assert np.array_equal(X[:, :105], raw["pooled"])
    prior = np.load(
        OLD / "results/within_size_training/tabular/features_predictions.npz",
        allow_pickle=False,
    )
    fold = json.loads((OLD / "configs/evaluation_groups/main_fold_0.json").read_text())
    tr = np.array(fold["train"])
    buckets = sv[:, 4:].argmax(1)
    centers = np.stack(
        [
            sums[tr[buckets[tr] == b]].sum(0) / counts[tr[buckets[tr] == b]].sum()
            for b in range(5)
        ]
    )
    szcenters = np.stack([sv[tr[buckets[tr] == b]].mean(0) for b in range(5)])
    reconstructed = center_features(X, sv, buckets, centers, szcenters)
    old_idx = prior["dataset_idx"]
    absdiff = np.max(np.abs(reconstructed[old_idx] - prior["extreme249"]))
    column_scale = np.maximum(np.abs(prior["extreme249"]).max(0), 1.0)
    column_relative_error = float(
        (np.abs(reconstructed[old_idx] - prior["extreme249"]) / column_scale).max()
    )
    assert column_relative_error < 1e-12
    assert np.allclose(
        reconstructed[old_idx], prior["extreme249"], rtol=1e-12, atol=1e-07
    )
    assert np.array_equal(y[old_idx], prior["raw_truth"])
    np.savez_compressed(
        OUT / "raw_features940.npz",
        dataset_idx=np.arange(940),
        size_bucket=buckets,
        size9=sv,
        node_counts=counts,
        grain_sum32=sums,
        raw_extreme249=X,
        raw_truth=y,
        selected32_sha256=np.array(hashes),
    )
    np.savez_compressed(
        OUT / "raw_truth940.npz",
        dataset_idx=np.arange(940),
        size_um=np.array(SIZES)[buckets],
        cases=np.array(CASES),
        raw_truth=y,
    )
    td = identity[
        ["source_geometry_id", "size_um", "original_v5_dataset_idx", "n_grains"]
    ].copy()
    for j, c in enumerate(CASES):
        td[c] = y[:, j]
    td.reset_index().to_csv(OUT / "raw_truth940.csv", index=False)
    config = {
        "status": "prepared; fits wait for frozen selection and independent gate",
        "model_labels": LABELS,
        "target": "raw SVE maximum",
        "cases": CASES,
        "units": {"FS": "dimensionless", "WP": "MPa"},
        "RF": {
            "n_estimators": 128,
            "max_depth": 8,
            "min_samples_leaf": 8,
            "max_features": 0.8,
            "n_jobs": 1,
        },
        "HGB": {
            "max_iter": 80,
            "max_leaf_nodes": 7,
            "max_depth": 3,
            "min_samples_leaf": 30,
            "learning_rate": 0.05,
            "l2_regularization": 1,
            "early_stopping": False,
        },
        "random_state": "Use the predeclared initialization_seed from each partition manifest for both families; paired comparison schedule replaces historical single seed42.",
        "source_recipe": str(
            OLD
            / "provenance/historical_maximum_reconciliation/within_size_pilot/spec.json"
        ),
        "source_recipe_sha256": sha(
            OLD
            / "provenance/historical_maximum_reconciliation/within_size_pilot/spec.json"
        ),
        "feature_definition": "Established249: mean/std0/max of raw32 selected grain inputs plus size9; q90/q99/top3 means of raw32;48 summaries of six physical input interactions. No CPFEM responses used as predictors. Degree is graph-derived.",
        "feature_preprocessing": "For each final partition, calculate grain-weighted raw32 means per size and size9 SVE means only over fitting IDs; subtract means in mean/max/q90/q99/top3 blocks and size9. All raw descriptors/materialization are deterministic before these fitted centers.",
        "target_preprocessing": "Per-size four-case fitting mean and sample variance, floor=max(size-case variance,1e-6 pooled fitting-case variance,1e-30); fit residual/SD and invert using fitting statistics for every outer SVE.",
        "fitting": "Same inner fitting IDs as graph arms; stopping SVEs not included in regressor fitting. HGB has early_stopping=False and fixed80 iterations; no tuning. RF fits four standardized targets jointly; HGB fits each case separately.",
        "partitions_sha256": sha(BASE / "partitions.json"),
        "raw_feature_artifact_sha256": sha(OUT / "raw_features940.npz"),
        "raw_truth_artifact_sha256": sha(OUT / "raw_truth940.npz"),
        "script_sha256": sha(Path(__file__)),
        "outputs": "predictions_long.csv: model,repeat,fold,dataset_idx,size_um,case,truth,prediction,target,units",
        "outer_inference_gate": "Run stage requires root protocol selected_variant_frozen flag, or a frozen_selection.json with status frozen. No fits from prepare stage.",
    }
    dump("config.json", config)
    dump(
        "preparation_report.json",
        {
            "status": "pass_prepared_without_fits",
            "n_sve": 940,
            "raw_input_dimension": 249,
            "n_case_targets": 4,
            "raw_graph_feature_fingerprints_verified": 940,
            "raw_case_peak_fingerprints_verified": 3760,
            "all940_raw_truth_exact_existing_audit": True,
            "all940_raw105_exact_existing_audit": True,
            "legacy752_raw_truth_exact": True,
            "legacy752_feature_parity_max_abs_error": float(absdiff),
            "legacy752_feature_parity_tolerance": {"rtol": 1e-12, "atol": 1e-07},
            "legacy752_feature_parity_max_column_scaled_error": column_relative_error,
            "legacy752_feature_roundoff_reason": "Raw pool then subtract center versus historical subtract grain centers before pooling; all raw105 descriptors and source fingerprints are exact.",
            "no_fitted_input_preprocessor_loaded_for_final_features": True,
            "no_models_fitted_or_scored": True,
            "seconds": time.perf_counter() - t,
            "raw_data_path": str(DATA),
            "raw_data_sha256": sha(DATA),
            "source_identity_sha256": sha(
                OLD / "provenance/dataset_lineage/source_identity_map.csv"
            ),
            "selected32_schema": columns,
            "feature_array_sha256": ah(X),
            "raw_truth_array_sha256": ah(y),
            "config_sha256": sha(OUT / "config.json"),
        },
    )
    print(
        "PREPARATION PASS: raw940×249 features and exact940×4 maxima; no fits",
        flush=True,
    )


def center_features(X, sv, buckets, centers, szcenters):
    xx = X.copy()
    for b in range(5):
        mask = buckets == b
        for first in [0, 64, 105, 137, 169]:
            xx[mask, first : first + 32] -= centers[b]
        xx[mask, 96:105] -= szcenters[b]
    return xx


def run():
    config = json.loads((OUT / "config.json").read_text())
    assert config["partitions_sha256"] == sha(BASE / "partitions.json")
    assert config["raw_feature_artifact_sha256"] == sha(OUT / "raw_features940.npz")
    protocol = json.loads((BASE / "protocol.json").read_text())
    frozen = bool(protocol.get("selected_variant_frozen", False))
    if (BASE / "frozen_selection.json").exists():
        frozen = (
            frozen
            or json.loads((BASE / "frozen_selection.json").read_text()).get("status")
            == "frozen"
        )
    assert (
        frozen
    ), "Wait for root frozen selection; outer regressor fits are not launched before variant choice."
    assert not (
        OUT / "predictions_long.csv"
    ).exists(), "Do not silently refit completed final predictions."
    z = np.load(OUT / "raw_features940.npz", allow_pickle=False)
    X = z["raw_extreme249"]
    y = z["raw_truth"]
    sv = z["size9"]
    buckets = z["size_bucket"]
    sums = z["grain_sum32"]
    counts = z["node_counts"]
    predrows = []
    fitrecords = []
    allstats = []
    started = time.perf_counter()
    for p in sorted(BASE.glob("partition_r*_f*.json")):
        partition = json.loads(p.read_text())
        repeat = partition["repeat"]
        fold = partition["fold"]
        seed = partition["initialization_seed"]
        tr = np.array(partition["fitting"], int)
        ev = np.array(partition["evaluation"], int)
        stopping = np.array(partition["stopping"], int)
        assert (
            not set(tr) & set(ev)
            and (not set(tr) & set(stopping))
            and (not set(stopping) & set(ev))
        )
        assert set(np.r_[tr, ev, stopping]) == set(range(940))
        centers = np.stack(
            [
                sums[tr[buckets[tr] == b]].sum(0) / counts[tr[buckets[tr] == b]].sum()
                for b in range(5)
            ]
        )
        szcenters = np.stack([sv[tr[buckets[tr] == b]].mean(0) for b in range(5)])
        xx = center_features(X, sv, buckets, centers, szcenters)
        means = np.stack([y[tr[buckets[tr] == b]].mean(0) for b in range(5)])
        variances = np.stack([y[tr[buckets[tr] == b]].var(0, ddof=1) for b in range(5)])
        pooled = y[tr].var(0, ddof=1)
        denominators = np.maximum(np.maximum(variances, 1e-06 * pooled), 1e-30)
        sd = np.sqrt(denominators)
        targets = (y[tr] - means[buckets[tr]]) / sd[buckets[tr]]
        allstats.append(
            {
                "repeat": repeat,
                "fold": fold,
                "partition_sha256": sha(p),
                "fitting_ids": tr.tolist(),
                "stopping_not_used": stopping.tolist(),
                "outer_ids": ev.tolist(),
                "grain32_center_by_size": centers.tolist(),
                "size9_center_by_size": szcenters.tolist(),
                "target_mean_by_size": means.tolist(),
                "target_sample_variance_by_size": variances.tolist(),
                "target_variance_floor_by_size": denominators.tolist(),
                "initialization_seed": seed,
            }
        )
        models = [
            (LABELS[0], RandomForestRegressor(**config["RF"], random_state=seed)),
            (
                LABELS[1],
                [
                    HistGradientBoostingRegressor(**config["HGB"], random_state=seed)
                    for c in CASES
                ],
            ),
        ]
        predictions = {LABELS[2]: means[buckets[ev]]}
        for label, model in models:
            t = time.perf_counter()
            with threadpool_limits(1):
                if isinstance(model, list):
                    for j, m in enumerate(model):
                        m.fit(xx[tr], targets[:, j])
                    raw = np.column_stack([m.predict(xx[ev]) for m in model])
                else:
                    model.fit(xx[tr], targets)
                    raw = model.predict(xx[ev])
            predictions[label] = means[buckets[ev]] + sd[buckets[ev]] * raw
            assert np.isfinite(predictions[label]).all()
            modelpath = (
                OUT
                / f"model_r{repeat}_f{fold}_{('rf' if label == LABELS[0] else 'hgb')}.pkl"
            )
            with modelpath.open("wb") as f:
                pickle.dump(model, f)
            fitrecords.append(
                {
                    "model": label,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": seed,
                    "n_fitting": len(tr),
                    "n_outer": len(ev),
                    "optimized_estimators": 4 if isinstance(model, list) else 1,
                    "seconds": time.perf_counter() - t,
                    "model_sha256": sha(modelpath),
                }
            )
            print(
                "FIT",
                label,
                "repeat",
                repeat,
                "fold",
                fold,
                "seconds",
                round(time.perf_counter() - t, 2),
                flush=True,
            )
        for label, prediction in predictions.items():
            for i, idx in enumerate(ev):
                for j, c in enumerate(CASES):
                    predrows.append(
                        {
                            "model": label,
                            "repeat": repeat,
                            "fold": fold,
                            "dataset_idx": int(idx),
                            "size_um": SIZES[int(buckets[idx])],
                            "case": c,
                            "truth": float(y[idx, j]),
                            "prediction": float(prediction[i, j]),
                            "target": "SVE maximum",
                            "units": "MPa" if c.endswith("wp") else "dimensionless",
                        }
                    )
        pd.DataFrame(predrows).to_csv(OUT / "predictions_long.partial.csv", index=False)
        dump("fitted_preprocessing.json", allstats)
        pd.DataFrame(fitrecords).to_csv(OUT / "fit_records.csv", index=False)
    df = pd.DataFrame(predrows)
    assert len(df) == 3 * 2 * 940 * 4 and (
        not df.duplicated(["model", "repeat", "dataset_idx", "case"]).any()
    )
    for (m, r), g in df.groupby(["model", "repeat"]):
        assert g.dataset_idx.nunique() == 940
    df.to_csv(OUT / "predictions_long.csv", index=False)
    dump(
        "report.json",
        {
            "status": "complete",
            "target": "raw SVE maximum",
            "n_models": 3,
            "n_repeats": 2,
            "n_outer_folds_per_repeat": 5,
            "unique_sves_per_model_repeat": 940,
            "optimized_estimator_fits": 50,
            "no_stopping_data_in_regressor_fits": True,
            "all_transforms_fitting_only": True,
            "no_hyperparameter_selection": True,
            "all_raw_truths_exact_saved940": True,
            "total_seconds": time.perf_counter() - started,
            "config_sha256": sha(OUT / "config.json"),
            "partitions_sha256": sha(BASE / "partitions.json"),
            "prediction_csv_sha256": sha(OUT / "predictions_long.csv"),
            "interpretation": "Fixed post-selection repeated evaluation on reused SVEs; two predictions of one SVE are dependent. Root combined summary supplies physical within-size metrics.",
        },
    )
    print("COMPLETE", len(df), "raw outer prediction rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        run()
