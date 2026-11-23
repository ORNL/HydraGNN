"""Matched analysis of the twenty frozen-encoder residual grain heads."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import time
import numpy as np
import pandas as pd
from fip_release_paths import matches_published_hash

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "frozen_residual"
PRIOR = HERE.parent.parent / "joint_branch_analysis"
VERSION = os.environ.get("FIP_RESIDUAL_VERSION", "55f907da468ecf20")
ARMS = ("bulk_control", "critical_treatment")
FAMILIES = ("cached_original", *ARMS)
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
SCOPES = ["pooled", *map(str, SIZES)]
BOOT = 4000
DIAGNOSTICS = [
    "top1_hit",
    "top4_hit",
    "true_value_capture",
    "top4_true_value_capture",
    "winner_shortfall",
]
STAT_NAMES = [
    "n",
    "sum_true",
    "sum_pred",
    "sum_true2",
    "sum_pred2",
    "sum_product",
    "sse",
    "sae",
]
SUFFIXES = [
    "grain_maximum/raw",
    "grain_maximum/ridge10",
    "bridge/raw",
    "bridge/ridge10",
    "summary/ridge100",
]
_spec = importlib.util.spec_from_file_location(
    "verified_joint_analysis_helpers", PRIOR / "analyze.py"
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
stats, score_stats, clean, sha = (ref.stats, ref.score_stats, ref.clean, ref.sha)


def dump(path, value):
    Path(path).write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load(path):
    return json.loads(Path(path).read_text())


def prepare_references():
    """Only already completed references and code; never a new outer archive."""
    paths = [
        PRIOR / n
        for n in (
            "analyze.py",
            "maximum_route_arrays.npz",
            "physical_sve_bootstrap.npz",
            "verification.json",
            "independent_verification.json",
            "independent_interval_verification.json",
            "reference_parity.json",
            "model_manifest.json",
        )
    ]
    assert all(
        (
            load(PRIOR / n)["status"] == "pass"
            for n in (
                "verification.json",
                "independent_verification.json",
                "independent_interval_verification.json",
            )
        )
    )
    pinned = dict(
        reference_files={str(p): sha(p) for p in paths},
        reference_arrays=str(PRIOR / "maximum_route_arrays.npz"),
        bootstrap_arrays=str(PRIOR / "physical_sve_bootstrap.npz"),
        reference_routes=34,
        duplicate_reference_arrays_written=False,
        new_outer_archives_opened=False,
    )
    path = HERE / "reference_sources.json"
    if path.exists():
        assert load(path) == pinned, "Previously pinned references changed"
    else:
        dump(path, pinned)
    return pinned


def verify_references():
    pinned = load(HERE / "reference_sources.json")
    for path, digest in pinned["reference_files"].items():
        assert sha(path) == digest, (path, "pinned reference changed")
    return pinned


def gate():
    """Verify the complete matched study before reading its prediction arrays."""
    records = []
    for part in load(ref.BASE / "partitions.json"):
        pair = f"repeat{part['repeat']}_fold{part['fold']}"
        prepdir = TRAIN / "preparation" / VERSION / pair
        prep = load(prepdir / "preparation.json")
        assert prep["partition"] == part
        for source_path, expected in prep["source_manifest"].items():
            assert matches_published_hash(source_path, expected), source_path
        for name in ("initial_head.pt", "fitting_statistics.npz"):
            assert sha(prepdir / name) == prep["files"][name]
        paired_cache = []
        for arm in ARMS:
            folder = TRAIN / "runs" / VERSION / arm / pair
            cp = load(folder / "completion.json")
            protocol = load(folder / "protocol.json")
            config = protocol["release"]
            assert protocol["arm"] == arm and protocol["partition"] == part
            assert config["source_manifest"] == prep["source_manifest"]
            assert config["partitions_sha256"] == sha(ref.BASE / "partitions.json")
            assert config["arms"] == list(ARMS) and config["target_fits"] == 20
            assert (
                config["head_training_device"]
                == protocol["head_training_device"]
                == "cuda:0"
            )
            assert matches_published_hash(
                prepdir / "preparation.json", config["preparations"][pair]
            )
            assert protocol["initial_head_sha256"] == prep["paired_initial_head_sha256"]
            assert protocol["original_checkpoint"] == prep["original_checkpoint"]
            assert protocol["optimizer_parameter_count"] == 16708
            assert (
                cp["status"] == "complete"
                and cp["arm"] == arm
                and (cp["partition"] == part)
            )
            assert not (folder / "failure.json").exists()
            assert cp["original_outputs_preserved"] and cp["optimizer_head_only"]
            assert cp["paired_outer_inputs_and_legacy_arrays_exact"]
            assert sha(folder / "best.pt") == cp["head_checkpoint_sha256"]
            seal = load(folder / "selected_head.json")
            assert (
                seal["strict_reloaded"]
                and seal["before_outer_target_extraction"]
                and seal["before_outer_transform_cache"]
            )
            assert (
                seal["partition"] == part
                and seal["selected_head_sha256"] == cp["head_checkpoint_sha256"]
            )
            assert cp["selected_identity"] == (cp["selected_epoch_zero_based"] == -1)
            assert cp["selected_stopping_score"] <= cp["identity_stopping_score"]
            assert [x["subset"] for x in cp["exports"]] == [
                "fitting",
                "stopping",
                "evaluation",
            ]
            for entry in cp["exports"]:
                assert (
                    sha(folder / f"{entry['subset']}_joint_predictions.npz")
                    == entry["archive_sha256"]
                )
            paired_cache.append(cp["shared_outer_cache_sha256"])
            records.append(
                dict(
                    arm=arm,
                    repeat=part["repeat"],
                    fold=part["fold"],
                    selected_epoch=cp["selected_epoch_zero_based"],
                    selected_identity=cp["selected_identity"],
                    actual_update_epochs=cp["actual_update_epochs"],
                    stop_reason=cp["stop_reason"],
                    identity_stopping_score=cp["identity_stopping_score"],
                    selected_stopping_score=cp["selected_stopping_score"],
                    checkpoint_sha256=cp["head_checkpoint_sha256"],
                    completion_sha256=sha(folder / "completion.json"),
                    preparation_sha256=sha(prepdir / "preparation.json"),
                    shared_outer_cache_sha256=cp["shared_outer_cache_sha256"],
                    exports=cp["exports"],
                )
            )
        assert paired_cache[0] == paired_cache[1]
        assert (
            sha(TRAIN / "outer_cache" / VERSION / pair / "cache.pt") == paired_cache[0]
        )
    assert len(records) == 20
    return records


def context():
    ctx = ref.context()
    global _BUCKETS
    _BUCKETS = ctx["buckets"]
    gv = np.full((2, 940, 4), np.nan)
    for p in ctx["parts"]:
        path = (
            TRAIN
            / "preparation"
            / VERSION
            / f"repeat{p['repeat']}_fold{p['fold']}"
            / "fitting_statistics.npz"
        )
        with np.load(path) as z:
            assert np.array_equal(
                z["maximum_variance"][ctx["buckets"][p["evaluation"]]],
                ctx["variance"][p["repeat"], p["evaluation"]],
            )
            gv[p["repeat"], p["evaluation"]] = z["grain_variance"][
                ctx["buckets"][p["evaluation"]]
            ]
    assert np.isfinite(gv).all() and (gv > 0).all()
    ctx["grain_variance"] = gv
    return ctx


def bootstrap():
    with np.load(PRIOR / "physical_sve_bootstrap.npz") as z:
        return [z[f"size_{size}"] for size in SIZES]


def load_all_predictions():
    result = {}
    for path in (PRIOR / "maximum_route_arrays.npz", HERE / "maximum_route_arrays.npz"):
        with np.load(path) as z:
            for name, arr in zip(z["models"].tolist(), z["predictions"]):
                assert name not in result
                result[name] = arr
    return result


def read_export(p, arm, subset):
    path = (
        TRAIN
        / "runs"
        / VERSION
        / arm
        / f"repeat{p['repeat']}_fold{p['fold']}"
        / f"{subset}_joint_predictions.npz"
    )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def validate_export(z, p, subset, ctx):
    ids = np.array(p[subset])
    assert z["dataset_idx"].tolist() == ids.tolist() and z["cases"].tolist() == CASES
    assert np.array_equal(z["y_true"], ctx["truth"][ids])
    assert np.array_equal(z["size_bucket"], ctx["buckets"][ids])
    assert np.array_equal(z["grain_count"], ctx["counts"][ids])
    assert np.array_equal(z["offsets"], np.r_[0, ctx["counts"][ids].cumsum()])
    assert z["new_head_has_no_uncertainty"].item() is True
    assert np.array_equal(z["original_prediction"], z["legacy_pred_peak"])
    assert np.array_equal(z["original_prediction"], z["legacy_node_mean_raw"][:, :4])
    assert np.array_equal(
        np.isnan(z["node_target_raw"][:, 1:]), z["node_aux_source_nan_mask"]
    )
    assert np.array_equal(z["node_target_raw"][:, :4], z["true_peak"])
    for key in (
        "new_raw_grain_prediction",
        "new_raw_grain_maximum_prediction",
        "true_peak",
        "original_prediction",
        "legacy_node_mean_raw",
        "legacy_graph_mean_raw",
        "legacy_node_transformed",
        "legacy_graph_transformed",
    ):
        assert np.isfinite(z[key]).all(), key
    for j, idx in enumerate(ids):
        lo, hi = z["offsets"][j : j + 2]
        assert np.array_equal(z["grain_id"][lo:hi], np.arange(hi - lo))
        assert np.all(z["grain_dataset_idx"][lo:hi] == idx)
        assert len(np.unique(z["vtk_grain_id"][lo:hi])) == hi - lo
        assert np.array_equal(z["true_peak"][lo:hi].max(0), z["y_true"][j])
        assert np.array_equal(
            z["new_raw_grain_prediction"][lo:hi].max(0),
            z["new_raw_grain_maximum_prediction"][j],
        )


def route_inputs(z, family):
    return dict(
        dataset_idx=z["dataset_idx"],
        y_true=z["y_true"],
        size_bucket=z["size_bucket"],
        grain_count=z["grain_count"],
        offsets=z["offsets"],
        pred_peak=z["original_prediction"]
        if family == "cached_original"
        else z["new_raw_grain_prediction"],
    )


def routes(stop, ev, p, ctx, family):
    si, ei = (stop["dataset_idx"], ev["dataset_idx"])
    st, sm, sn = ctx["bridge"](stop)
    et, em, en = ctx["bridge"](ev)
    sb, eb = (
        st[:, :, 0] + ref.FACTOR * st[:, :, 2],
        et[:, :, 0] + ref.FACTOR * et[:, :, 2],
    )
    ss = ctx["summaries"](stop["pred_peak"], stop["grain_count"])
    es = ctx["summaries"](ev["pred_peak"], ev["grain_count"])
    values = {family + "/" + suffix: np.empty((len(ei), 4)) for suffix in SUFFIXES}
    values[family + "/grain_maximum/raw"] = em
    fits, eligibility = ([], [])
    for c, case in enumerate(CASES):
        fallback = ctx["fit_means"][p["repeat"], p["fold"]][ev["size_bucket"], c]
        values[family + "/bridge/raw"][:, c] = np.where(
            np.isfinite(eb[:, c]), eb[:, c], fallback
        )
        for kind, x, v in (
            ("grain_maximum", sm[:, c, None], em[:, c, None]),
            ("bridge", sb[:, c, None], eb[:, c, None]),
            (
                "summary",
                np.c_[ss[:, c], np.eye(5)[stop["size_bucket"]]],
                np.c_[es[:, c], np.eye(5)[ev["size_bucket"]]],
            ),
        ):
            alpha = 100.0 if kind == "summary" else 10.0
            name = (
                family + "/" + kind + ("/ridge100" if kind == "summary" else "/ridge10")
            )
            values[name][:, c], fit = ref.calibrate(
                x,
                v,
                stop["y_true"][:, c],
                si,
                ei,
                name,
                p,
                case,
                alpha=alpha,
                eligibility=kind == "bridge",
                fallback=fallback if kind == "bridge" else None,
            )
            if kind == "summary":
                fit["feature_names"] = ctx["summary_names"] + [
                    f"size_{s}" for s in SIZES
                ]
            fits.append(fit)
        for subset, ids, count, scalar in (
            ("stopping", si, sn[:, c], sb[:, c]),
            ("evaluation", ei, en[:, c], eb[:, c]),
        ):
            eligibility.extend(
                (
                    dict(
                        model=family,
                        repeat=p["repeat"],
                        fold=p["fold"],
                        subset=subset,
                        case=case,
                        dataset_idx=int(i),
                        selected_count=int(n),
                        eligible=bool(np.isfinite(v)),
                    )
                    for i, n, v in zip(ids, count, scalar)
                )
            )
    features = dict(
        stopping_grain_maximum=sm,
        evaluation_grain_maximum=em,
        stopping_bridge=sb,
        evaluation_bridge=eb,
        stopping_summary=ss,
        evaluation_summary=es,
        stopping_ids=si,
        evaluation_ids=ei,
    )
    return (values, fits, eligibility, features)


def weighted_stats(y, p, w):
    w = w / w.sum(0)
    return np.stack(
        [
            np.ones(y.shape[1]),
            (w * y).sum(0),
            (w * p).sum(0),
            (w * y * y).sum(0),
            (w * p * p).sum(0),
            (w * y * p).sum(0),
            (w * (y - p) ** 2).sum(0),
            (w * abs(y - p)).sum(0),
        ],
        -1,
    )


def critical(y, p):
    """All calculations are diagnostics; canonical row tie-breaking is explicit."""
    n, k = (len(y), min(4, len(y)))
    maximum = y.max(0)
    tied = y == maximum
    cutoff = np.partition(y, n - k, axis=0)[n - k]
    above, equal = (y > cutoff, y == cutoff)
    weight = above.astype(float) + equal * (k - above.sum(0)) / equal.sum(0)
    amplitude = np.stack(
        [weighted_stats(y, p, tied.astype(float)), weighted_stats(y, p, weight)]
    )
    ranked = np.argsort(-p, axis=0, kind="stable")[:k]
    columns = np.arange(4)
    winner = ranked[0]
    value = y[winner, columns]
    top_value = y[ranked, columns].max(0)
    zero = maximum == 0
    assert np.all(~zero | np.all(y == 0, axis=0))
    capture = np.divide(value, maximum, out=np.ones(4), where=~zero)
    capture4 = np.divide(top_value, maximum, out=np.ones(4), where=~zero)
    diag = np.stack(
        [value == maximum, top_value == maximum, capture, capture4, maximum - value], -1
    ).astype(float)
    m = tied.sum(0)
    chance4 = np.array(
        [
            1.0
            if n - int(t) < k
            else 1.0 - math.prod(((n - int(t) - j) / (n - j) for j in range(k)))
            for t in m
        ]
    )
    return (
        amplitude,
        diag,
        weight,
        dict(tie_count=m, random_top1=m / n, random_top4=chance4, zero=zero),
    )


def metric_frame(store, fields, field_key="case"):
    rows = []
    for model, arr in store.items():
        for r in range(2):
            for scope in SCOPES:
                ids = scope_ids(scope)
                for c, field in enumerate(fields):
                    rows.append(
                        dict(
                            model=model,
                            repeat=r,
                            size_um=scope,
                            **{field_key: field},
                            **score_stats(arr[r, ids, c].sum(0)),
                        )
                    )
    return pd.DataFrame(rows)


_BUCKETS = None


def scope_ids(scope):
    return (
        np.arange(940)
        if scope == "pooled"
        else np.flatnonzero(SIZES[_BUCKETS] == int(scope))
    )


def scope_draws(scope, draws):
    return (
        np.concatenate(draws, axis=1)
        if scope == "pooled"
        else draws[list(SIZES).index(int(scope))]
    )


def correlation_from_stats(values):
    n, sy, sp, sy2, sp2, syp = [values[..., i] for i in range(6)]
    vy, vp = (sy2 - sy * sy / n, sp2 - sp * sp / n)
    den = np.sqrt(np.maximum(vy, 0) * np.maximum(vp, 0))
    return np.divide(
        syp - sy * sp / n, den, out=np.full_like(den, np.nan), where=den > 0
    )


def paired_stats(store, comparisons, draws, channel=None):
    rows = []
    for scope in SCOPES:
        ids, sample = (scope_ids(scope), scope_draws(scope, draws))
        moments, bootmoments = ({}, {})
        for name in set(sum(([a, b] for a, b in comparisons), [])):
            arr = store[name]
            moments[name] = arr[:, ids].sum(1)
            bm = np.empty((2, BOOT, 4, 8))
            for r in range(2):
                for j in range(8):
                    bm[r, :, :, j] = arr[r, :, :, j][sample].sum(1)
            bootmoments[name] = bm
        for model, reference in comparisons:
            aa, bb = (moments[model], moments[reference])
            ba, br = (bootmoments[model], bootmoments[reference])
            for c, case in enumerate(CASES):
                am = aa[:, c, 6].sum() / aa[:, c, 0].sum()
                bm = bb[:, c, 6].sum() / bb[:, c, 0].sum()
                ratio = 100 * (1 - np.sqrt(am / bm))
                sampled = 100 * (
                    1
                    - np.sqrt(
                        ba[:, :, c, 6].sum(0)
                        / ba[:, :, c, 0].sum(0)
                        / (br[:, :, c, 6].sum(0) / br[:, :, c, 0].sum(0))
                    )
                )
                pdiff = (
                    correlation_from_stats(aa[:, c]).mean()
                    - correlation_from_stats(bb[:, c]).mean()
                )
                bd = correlation_from_stats(ba[:, :, c]).mean(
                    0
                ) - correlation_from_stats(br[:, :, c]).mean(0)
                valid = np.isfinite(bd)
                pci = (
                    np.percentile(bd[valid], [2.5, 97.5])
                    if valid.any()
                    else [np.nan, np.nan]
                )
                ci = np.percentile(sampled, [2.5, 97.5])
                row = dict(
                    model=model,
                    reference=reference,
                    case=case,
                    size_um=scope,
                    n_sve=len(ids),
                    rmse_reduction_percent=ratio,
                    rmse_ci_low=ci[0],
                    rmse_ci_high=ci[1],
                    model_rmse=np.sqrt(am),
                    reference_rmse=np.sqrt(bm),
                    pearson_difference=pdiff,
                    pearson_ci_low=pci[0],
                    pearson_ci_high=pci[1],
                    valid_pearson_bootstrap_draws=int(valid.sum()),
                )
                if channel is not None:
                    row["channel"] = channel
                rows.append(row)
    return pd.DataFrame(rows)


def maximum_analysis(store, ctx, draws):
    rows, moments = ([], {})
    for name, p in store.items():
        y = np.broadcast_to(ctx["truth"], p.shape)
        moments[name] = np.stack(
            [np.ones_like(y), y, p, y * y, p * p, y * p, (y - p) ** 2, abs(y - p)], -1
        )
        for r in range(2):
            for scope in SCOPES:
                ids = scope_ids(scope)
                for c, case in enumerate(CASES):
                    rows.append(
                        dict(
                            model=name,
                            repeat=r,
                            size_um=scope,
                            case=case,
                            normalized_mse=np.mean(
                                (p[r, ids, c] - y[r, ids, c]) ** 2
                                / ctx["variance"][r, ids, c]
                            ),
                            n_negative=int((p[r, ids, c] < 0).sum()),
                            minimum_prediction=float(p[r, ids, c].min()),
                            **score_stats(moments[name][r, ids, c].sum(0)),
                        )
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "metrics_by_repeat.csv", index=False)
    frame[
        ["model", "repeat", "case", "size_um", "n", "n_negative", "minimum_prediction"]
    ].to_csv(HERE / "negative_maximum_predictions.csv", index=False)
    frame.groupby(["model", "case", "size_um"])[
        [
            "r2",
            "rmse",
            "mae",
            "bias",
            "pearson",
            "truth_sd",
            "prediction_sd",
            "normalized_mse",
        ]
    ].mean().add_suffix("_mean").reset_index().to_csv(
        HERE / "metrics_summary.csv", index=False
    )
    comparisons = [(ARMS[1] + "/" + s, ARMS[0] + "/" + s) for s in SUFFIXES]
    for arm in ARMS:
        for suffix in SUFFIXES:
            comparisons += [
                (arm + "/" + suffix, "cached_original/" + suffix),
                (arm + "/" + suffix, "reference/size_mean_union"),
            ]
        for context_ref in [
            "original/legacy_direct/raw",
            "original/bridge/ridge10",
            "original/summary/ridge100",
            "reference/RF249_union",
            "reference/HGB249_union",
            "reference/HGB_crossfit_same_case_ridge100",
            "reference/HGB_crossfit_all_cases_ridge100",
            "attached/raw_branch/raw",
            "detached/raw_branch/raw",
        ]:
            comparisons.append((arm + "/grain_maximum/raw", context_ref))
    paired_stats(moments, comparisons, draws).to_csv(
        HERE / "paired_maximum_comparisons.csv", index=False
    )
    risks = []
    for name, p in store.items():
        loss = (p - ctx["truth"]) ** 2 / ctx["variance"]
        byrepeat = np.array(
            [
                np.mean([loss[r, scope_ids(str(s))].mean() for s in SIZES])
                for r in range(2)
            ]
        )
        risks.append(
            dict(
                model=name,
                risk_equal_20_cells=byrepeat.mean(),
                repeat0=byrepeat[0],
                repeat1=byrepeat[1],
            )
        )
    pd.DataFrame(risks).to_csv(HERE / "normalized_risk.csv", index=False)
    norm = {
        arm: (
            (store[arm + "/grain_maximum/raw"] - ctx["truth"]) ** 2 / ctx["variance"]
        ).mean(0)
        for arm in ARMS
    }
    risk = {
        arm: np.mean([x[scope_ids(str(s))].mean() for s in SIZES])
        for arm, x in norm.items()
    }
    br = {arm: sum((x[d].mean((1, 2)) for d in draws)) / 5 for arm, x in norm.items()}
    t, c = (risk[ARMS[1]], risk[ARMS[0]])
    bt, bc = (br[ARMS[1]], br[ARMS[0]])
    primary = dict(
        model=ARMS[1] + "/grain_maximum/raw",
        reference=ARMS[0] + "/grain_maximum/raw",
        treatment_risk=t,
        control_risk=c,
        absolute_risk_reduction=c - t,
        absolute_ci=np.percentile(bc - bt, [2.5, 97.5]),
        normalized_mse_reduction_percent=100 * (1 - t / c),
        reduction_percent_ci=np.percentile(100 * (1 - bt / bc), [2.5, 97.5]),
        bootstrap_draws=BOOT,
        unit="physical SVE",
        repeat_losses_averaged_before_resampling=True,
        prediction_ensemble=False,
        inference="pointwise conditional development-cohort comparison",
    )
    dump(HERE / "primary_comparison.json", primary)


def joint_analysis(
    grain, critical_stats, diagnostics, legacy_stats, objectives, ctx, draws
):
    frame = metric_frame(grain, CASES)
    frame.to_csv(HERE / "grain_metrics_by_repeat.csv", index=False)
    frame.groupby(["model", "case", "size_um"])[
        ["r2", "rmse", "mae", "bias", "pearson"]
    ].mean().add_suffix("_mean").reset_index().to_csv(
        HERE / "grain_metrics_summary.csv", index=False
    )
    critical_rows = []
    pairs = [
        (ARMS[1], ARMS[0]),
        (ARMS[0], "cached_original"),
        (ARMS[1], "cached_original"),
    ]
    paired = [paired_stats(grain, pairs, draws, "grain")]
    for j, channel in enumerate(("critical", "topfour")):
        ss = {name: arr[:, :, j] for name, arr in critical_stats.items()}
        f = metric_frame(ss, CASES)
        f["channel"] = channel
        critical_rows.append(f)
        paired.append(paired_stats(ss, pairs, draws, channel))
    pd.concat(critical_rows).to_csv(
        HERE / "critical_metrics_by_repeat.csv", index=False
    )
    pd.concat(paired).to_csv(HERE / "grain_paired_comparisons.csv", index=False)
    fields = (
        [c + "/grain_peak" for c in CASES]
        + [c + "/within_grain_mean" for c in CASES]
        + [c + "/physical_log_location" for c in CASES]
        + [c + "/physical_log_space_sd" for c in CASES]
        + [c + "/sve_maximum" for c in CASES]
        + [c + "/sve_mean" for c in CASES]
        + [
            "yield_stress",
            "hardening_modulus",
            "Wp_lcf",
            "Wp_hcf",
            "sigma_amp_lcf",
            "sigma_amp_hcf",
        ]
    )
    fip_units = ["dimensionless" if c.endswith("fs") else "MPa" for c in CASES]
    units = (
        fip_units * 2
        + ["log of FIP in stored numerical units"] * 4
        + ["dimensionless log-space SD"] * 4
        + fip_units * 2
        + ["MPa"] * 6
    )
    lf = metric_frame({"cached_original": legacy_stats}, fields, "field")
    lf["units"] = lf.field.map(dict(zip(fields, units)))
    lf.to_csv(HERE / "legacy_metrics_by_repeat.csv", index=False)
    lf.groupby(["model", "field", "units", "size_um"])[
        ["r2", "rmse", "mae", "bias", "pearson"]
    ].mean().add_suffix("_mean").reset_index().to_csv(
        HERE / "legacy_metrics_summary.csv", index=False
    )
    np.savez_compressed(
        HERE / "legacy_statistics.npz",
        statistics=legacy_stats,
        fields=np.array(fields),
        units=np.array(units),
        stat_names=np.array(STAT_NAMES),
    )
    objrows = []
    for model, values in objectives.items():
        for scope in SCOPES:
            avg = values[:, scope_ids(scope)].mean((0, 1))
            for c, case in enumerate(CASES):
                b, t, p = avg[c]
                objrows.append(
                    dict(
                        model=model,
                        case=case,
                        size_um=scope,
                        B=b,
                        T=t,
                        P=p,
                        composite=0.5 * b + 0.5 * p,
                    )
                )
        b, t, p = np.mean(
            [values[:, scope_ids(str(s))].mean((0, 1, 2)) for s in SIZES], axis=0
        )
        objrows.append(
            dict(
                model=model,
                case="all",
                size_um="equal_20_cells",
                B=b,
                T=t,
                P=p,
                composite=0.5 * b + 0.5 * p,
            )
        )
    pd.DataFrame(objrows).to_csv(HERE / "objective_summary.csv", index=False)
    capture = []
    contrasts = []
    for scope in SCOPES:
        ids, sample = (scope_ids(scope), scope_draws(scope, draws))
        mean = {name: arr.mean(0) for name, arr in diagnostics.items()}
        for name, arr in mean.items():
            for c, case in enumerate(CASES):
                capture.append(
                    dict(
                        model=name,
                        case=case,
                        size_um=scope,
                        **dict(zip(DIAGNOSTICS, arr[ids, c].mean(0))),
                    )
                )
        for name, reference in pairs:
            diff = mean[name] - mean[reference]
            for c, case in enumerate(CASES):
                samplemean = diff[sample, c].mean(1)
                for j, diag in enumerate(DIAGNOSTICS):
                    ci = np.percentile(samplemean[:, j], [2.5, 97.5])
                    contrasts.append(
                        dict(
                            model=name,
                            reference=reference,
                            case=case,
                            size_um=scope,
                            diagnostic=diag,
                            difference=diff[ids, c, j].mean(),
                            ci_low=ci[0],
                            ci_high=ci[1],
                        )
                    )
    pd.DataFrame(capture).to_csv(HERE / "capture_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(HERE / "capture_paired_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_references(), indent=2))
        return
    if not args.run:
        parser.error("Choose --prepare or --run")
    assert not (HERE / "verification.json").exists(), "Completed analysis preserved"
    started = time.monotonic()
    records = gate()
    verify_references()
    ctx = context()
    global _BUCKETS
    _BUCKETS = ctx["buckets"]
    store = {}
    fits = []
    elig = []
    identity = []
    negative = []
    grain = {name: np.full((2, 940, 4, 8), np.nan) for name in FAMILIES}
    cs = {name: np.full((2, 940, 2, 4, 8), np.nan) for name in FAMILIES}
    diag = {name: np.full((2, 940, 4, 5), np.nan) for name in FAMILIES}
    obj = {name: np.full((2, 940, 4, 3), np.nan) for name in FAMILIES}
    legacy_stats = np.full((2, 940, 30, 8), np.nan)
    metadata = {
        k: np.full((940, 4), np.nan)
        for k in ("true_max_tie_count", "random_top1", "random_top4", "zero_target")
    }
    import torch

    for p in ctx["parts"]:
        r, f = (p["repeat"], p["fold"])
        pair = f"repeat{r}_fold{f}"
        evs = {}
        stops = {}
        for arm in ARMS:
            stops[arm] = read_export(p, arm, "stopping")
            evs[arm] = read_export(p, arm, "evaluation")
            validate_export(stops[arm], p, "stopping", ctx)
            validate_export(evs[arm], p, "evaluation", ctx)
        common = torch.load(
            TRAIN / "outer_cache" / VERSION / pair / "cache.pt",
            weights_only=True,
            map_location="cpu",
        )
        for key, value in common.items():
            if key == "features":
                continue
            expected = value.numpy()
            for arm in ARMS:
                assert np.array_equal(evs[arm][key], expected, equal_nan=True), (
                    pair,
                    arm,
                    key,
                )
        del common
        for key in stops[ARMS[0]]:
            if key not in (
                "arm",
                "new_raw_grain_prediction",
                "new_raw_grain_maximum_prediction",
            ):
                left, right = (stops[ARMS[0]][key], stops[ARMS[1]][key])
                assert (
                    np.array_equal(left, right, equal_nan=True)
                    if left.dtype.kind in "fc"
                    else np.array_equal(left, right)
                ), key
        identity.append(
            dict(
                repeat=r,
                fold=f,
                all_cached_legacy_arrays_exact=True,
                all_paired_stopping_legacy_arrays_exact=True,
                all_raw_targets_masks_and_canonical_ids_verified=True,
                original_uncertainty_not_transferred=True,
            )
        )
        for family in FAMILIES:
            arm = ARMS[0] if family == "cached_original" else family
            stop, ev = (stops[arm], evs[arm])
            values, ff, ll, features = routes(
                route_inputs(stop, family), route_inputs(ev, family), p, ctx, family
            )
            for name, value in values.items():
                dest = store.setdefault(name, np.full((2, 940, 4), np.nan))
                assert np.isnan(dest[r, p["evaluation"]]).all()
                dest[r, p["evaluation"]] = value
            fits.extend(ff)
            elig.extend(ll)
            np.savez_compressed(
                HERE / f"{family}_{pair}_readout_features.npz", **features
            )
            prediction = (
                ev["original_prediction"]
                if family == "cached_original"
                else ev["new_raw_grain_prediction"]
            )
            for j, idx in enumerate(p["evaluation"]):
                lo, hi = ev["offsets"][j : j + 2]
                y = ev["true_peak"][lo:hi]
                pr = prediction[lo:hi]
                grain[family][r, idx] = stats(y, pr)
                amp, dd, weight, meta = critical(y, pr)
                cs[family][r, idx] = amp
                diag[family][r, idx] = dd
                b = ((pr - y) ** 2).mean(0) / ctx["grain_variance"][r, idx]
                t = (
                    (weight * (pr - y) ** 2).sum(0)
                    / weight.sum(0)
                    / ctx["grain_variance"][r, idx]
                )
                pp = (pr.max(0) - y.max(0)) ** 2 / ctx["variance"][r, idx]
                obj[family][r, idx] = np.stack([b, t, pp], -1)
                if family == "cached_original":
                    for dest, key in (
                        ("true_max_tie_count", "tie_count"),
                        ("random_top1", "random_top1"),
                        ("random_top4", "random_top4"),
                        ("zero_target", "zero"),
                    ):
                        if r == 0:
                            metadata[dest][idx] = meta[key]
                        else:
                            assert np.array_equal(metadata[dest][idx], meta[key])
                    legacy_stats[r, idx, :16] = stats(
                        ev["node_target_raw"][lo:hi], ev["legacy_node_mean_raw"][lo:hi]
                    )
                    legacy_stats[r, idx, 16:] = stats(
                        ev["graph_target_raw"][j : j + 1],
                        ev["legacy_graph_mean_raw"][j : j + 1],
                    )
                for c, case in enumerate(CASES):
                    negative.append(
                        dict(
                            model=family,
                            repeat=r,
                            dataset_idx=idx,
                            case=case,
                            size_um=int(SIZES[ctx["buckets"][idx]]),
                            n_grains=hi - lo,
                            n_negative=int((pr[:, c] < 0).sum()),
                            minimum_prediction=float(pr[:, c].min()),
                        )
                    )
        print("ANALYZED_PAIR", r, f, flush=True)
    for arrays in (store, grain, cs, diag, obj):
        assert all((np.isfinite(x).all() for x in arrays.values()))
    assert np.isfinite(legacy_stats).all()
    np.savez_compressed(
        HERE / "maximum_route_arrays.npz",
        models=np.array(list(store)),
        predictions=np.stack(list(store.values())),
        truth=ctx["truth"],
        size_bucket=ctx["buckets"],
        fitting_variance=ctx["variance"],
        cases=np.array(CASES),
    )
    np.savez_compressed(
        HERE / "grain_statistics.npz",
        models=np.array(FAMILIES),
        statistics=np.stack(list(grain.values())),
        stat_names=np.array(STAT_NAMES),
    )
    np.savez_compressed(
        HERE / "critical_statistics.npz",
        models=np.array(FAMILIES),
        statistics=np.stack(list(cs.values())),
        diagnostics=np.stack(list(diag.values())),
        diagnostic_names=np.array(DIAGNOSTICS),
        channels=np.array(["critical", "topfour"]),
        grain_count=ctx["counts"],
        **metadata,
    )
    np.savez_compressed(
        HERE / "objective_cells.npz",
        models=np.array(FAMILIES),
        B=np.stack([v[..., 0] for v in obj.values()]),
        T=np.stack([v[..., 1] for v in obj.values()]),
        P=np.stack([v[..., 2] for v in obj.values()]),
    )
    dump(HERE / "calibration_fits.json", fits)
    pd.DataFrame(elig).to_csv(HERE / "bridge_eligibility.csv", index=False)
    pd.DataFrame(negative).to_csv(HERE / "negative_grain_predictions.csv", index=False)
    dump(
        HERE / "legacy_preservation.json",
        dict(
            status="pass",
            pairs=identity,
            exact_paired_cached_arrays=True,
            source_replay_is_tolerant_not_bitwise=True,
        ),
    )
    pd.DataFrame(records).to_csv(HERE / "checkpoint_summary.csv", index=False)
    dump(HERE / "completed_fit_records.json", records)
    inherited = load(PRIOR / "model_manifest.json")
    additions = []
    for name in store:
        family, suffix = name.split("/", 1)
        additions.append(
            dict(
                model=name,
                original_anchor="current repeated 940-SVE baseline best.pt for this outer fold",
                head="frozen original p0"
                if family == "cached_original"
                else "16,708-parameter residual raw-grain point head on frozen h256",
                objective="none"
                if family == "cached_original"
                else "B"
                if family == ARMS[0]
                else "B + 0.5 T + 0.5 P",
                head_training_labels="none"
                if family == "cached_original"
                else "fitting grain targets only",
                head_training_device="none"
                if family == "cached_original"
                else "cuda:0",
                selection="original selected checkpoint"
                if family == "cached_original"
                else "75 stopping SVEs, equal20 mean of 0.5 B + 0.5 P, identity eligible",
                correction_labels="eligible stopping only"
                if suffix == "bridge/ridge10"
                else "all 75 stopping"
                if suffix.endswith(("ridge10", "ridge100"))
                else "none",
                no_new_uncertainty=True,
                original_model_functions_preserved=True,
                stopping_reused_for_selection_and_correction=family in ARMS
                and suffix.endswith(("ridge10", "ridge100")),
            )
        )
    dump(
        HERE / "model_manifest.json",
        dict(
            new_routes=additions,
            pinned_reference_manifest=str(PRIOR / "model_manifest.json"),
            reference_routes=inherited["routes"],
            partitions=inherited["partitions"],
            exact_preservation="shared cached arrays identical across arms; historical inference replay has recorded numerical tolerances",
            original_current_not_submitted=True,
            reference_hgb_readout_budget="751-753 union labels versus 75 stopping labels for these fixed corrections",
        ),
    )
    draws = bootstrap()
    maximum_analysis(load_all_predictions(), ctx, draws)
    joint_analysis(grain, cs, diag, legacy_stats, obj, ctx, draws)
    dump(
        HERE / "source_hashes.json",
        {
            **ref.SOURCES,
            **{str(p): sha(p) for p in [Path(__file__), HERE / "PRESPEC.txt"]},
        },
    )
    dump(
        HERE / "verification.json",
        dict(
            status="pass",
            completed_fits=20,
            new_routes=15,
            reference_routes=34,
            all_routes=49,
            predictions_per_route=7520,
            calibrators=len(fits),
            bootstrap_draws=BOOT,
            all_required_predictions_finite=True,
            no_dropped_sves=True,
            all_calibration_labels_stopping_only=True,
            stopping_reused_for_checkpoint_and_correction=True,
            all_twenty_completions_verified_before_analysis=True,
            original_legacy_arrays_exactly_preserved=True,
            no_uncertainty_transfer=True,
            reference_arrays_reused_without_duplication=True,
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    print(json.dumps(load(HERE / "verification.json"), indent=2))


if __name__ == "__main__":
    main()
