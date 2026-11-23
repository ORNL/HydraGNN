"""Fixed-input readout budget intervention; no encoder or tree fitting."""
from fip_release_paths import release_root
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parents[1]
W = HERE.parents[3]
BASE = W / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
H = Path(str(release_root() / "examples/fip_revision_restart_20261005"))
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
GROUPS = ["same_case", "all_cases"]
MODES = ["alpha100", "alpha_per_observation", "matched_df"]
DRAWS = 100
NBOOT = 4000
SEED = 2026111607
SOURCES = {}


def sha(p):
    p = Path(p)
    s = hashlib.sha256(p.read_bytes()).hexdigest()
    SOURCES[str(p)] = s
    return s


def load_npz(p):
    sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_json(p):
    sha(p)
    return json.loads(Path(p).read_text())


def frame(p):
    sha(p)
    return pd.read_csv(p)


def decomp(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    yc = y - y.mean()
    pc = p - p.mean()
    vy = np.mean(yc * yc)
    vp = np.mean(pc * pc)
    sy = np.sqrt(vy)
    sp = np.sqrt(vp)
    cov = np.mean(yc * pc)
    bias = float(np.mean(p - y))
    mse = float(np.mean((p - y) ** 2))
    spread = float((sp - sy) ** 2)
    corrterm = float(2 * (sp * sy - cov))
    rho = float(cov / (sy * sp)) if sy > 0 and sp > 0 else None
    residual = mse - (bias * bias + spread + corrterm)
    assert abs(residual) <= max(mse, vy, 1e-30) * 2e-12
    return dict(
        n=len(y),
        mse=mse,
        rmse=np.sqrt(mse),
        mae=float(np.mean(abs(p - y))),
        bias=bias,
        target_sd=float(sy),
        prediction_sd=float(sp),
        pearson=rho,
        r2=float(1 - mse / vy) if vy > 0 else None,
        bias_squared=bias * bias,
        spread_component=spread,
        correlation_component=corrterm,
        decomposition_residual=residual,
    )


def design(features, b, group, c):
    if group == "same_case":
        x = features[:, c]
    else:
        x = np.c_[features[:, 0, :2], features[:, :, 2:].reshape(len(features), -1)]
    return np.c_[x, np.eye(5)[b]]


def standardized_svd(X, Z, y):
    sc = StandardScaler().fit(X)
    xx = sc.transform(X)
    zz = sc.transform(Z)
    xm = xx.mean(0)
    xx = xx - xm
    zz = zz - xm
    ym = y.mean(0)
    yy = y - ym
    u, s, vt = np.linalg.svd(xx, full_matrices=False)
    tol = max(xx.shape) * np.finfo(float).eps * s[0] if len(s) else 0
    valid = s > tol

    def predict(alpha):
        if alpha == 0:
            weights = np.zeros_like(s)
            weights[valid] = 1 / s[valid]
        else:
            weights = s / (s * s + alpha)
        coef = vt.T * weights @ (u.T @ yy)
        return zz @ coef + ym

    def df(alpha):
        if alpha == 0:
            return float(valid.sum())
        return float(np.sum(s * s / (s * s + alpha)))

    return (predict, df, int(valid.sum()), sc, xx, zz)


def match_df(df, rank, target):
    if target >= rank - 1e-10:
        return (
            0.0,
            "unreachable_minimum_norm" if target > rank + 1e-08 else "rank_limit",
        )
    high = 100.0
    while df(high) > target:
        high *= 10
    return (
        float(brentq(lambda a: df(a) - target, 1e-12, high, xtol=1e-12, rtol=1e-13)),
        "matched",
    )


def budget():
    parts = read_json(BASE / "partitions.json")
    raw = load_npz(BASE / "regressors/raw_features940.npz")
    truth, b = (raw["raw_truth"], raw["size_bucket"])
    published = frame(STUDIES / "cross_case_readout/predictions.csv.gz")
    published_metrics = frame(STUDIES / "cross_case_readout/metrics_summary.csv")
    full = np.full((2, 2, 940, 4), np.nan)
    small = np.full((2, 2, 3, DRAWS, 940, 4), np.nan)
    fitrows = []
    checks = []
    feature_contract = {}
    subset_records = []
    baseline_max_scaled = 0.0
    manual_max_scaled = 0.0
    for part in parts:
        r, f = (part["repeat"], part["fold"])
        folder = STUDIES / f"boosted_node/stacking/repeat{r}_fold{f}"
        a = load_npz(folder / "summary_features.npz")
        tr, ev = (a["outer_training_ids"], a["evaluation_ids"])
        assert set(tr) == set(part["fitting"]) | set(part["stopping"])
        assert set(ev) == set(part["evaluation"]) and (not set(tr) & set(ev))
        assert list(a["cases"]) == CASES
        assert np.array_equal(a["oof_targets"], truth[tr])
        assert np.array_equal(a["evaluation_targets"], truth[ev])
        assert (
            np.isfinite(a["oof_features"]).all()
            and np.isfinite(a["evaluation_features"]).all()
        )
        assert np.all(a["oof_features"][:, :, :2] == a["oof_features"][:, 0:1, :2])
        seen = []
        for inner in read_json(folder / "inner_partitions.json"):
            ti, vi = (inner["fitting_ids"], inner["heldout_ids"])
            assert not set(ti) & set(vi) and set(ti) | set(vi) == set(tr)
            seen += vi
            pre = read_json(folder / f"inner{inner['inner_fold']}/preprocessing.json")
            assert pre["fit_ids"] == ti and pre["predicted_ids"] == vi
        assert sorted(seen) == tr.tolist() and len(set(seen)) == len(tr)
        pre = read_json(folder / "union_refit/preprocessing.json")
        assert pre["fit_ids"] == tr.tolist() and pre["predicted_ids"] == ev.tolist()
        ns = np.bincount(b[part["stopping"]], minlength=5)
        subset_ids = []
        subset_positions = []
        for d in range(DRAWS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, r, f, d]))
            ix = np.sort(
                np.concatenate(
                    [
                        rng.choice(
                            np.flatnonzero(b[tr] == k), int(ns[k]), replace=False
                        )
                        for k in range(5)
                    ]
                )
            )
            assert len(ix) == 75 and len(np.unique(ix)) == 75
            subset_positions.append(ix)
            subset_ids.append(tr[ix])
        subset_ids = np.stack(subset_ids)
        subset_records.append(
            dict(
                repeat=r,
                fold=f,
                size_counts=ns.tolist(),
                n_full=len(tr),
                subset_ids_sha256=hashlib.sha256(subset_ids.tobytes()).hexdigest(),
            )
        )
        foldfull = np.full((2, len(ev), 4), np.nan)
        foldsmall = np.full((2, 3, DRAWS, len(ev), 4), np.nan)
        for gi, group in enumerate(GROUPS):
            fc = [None] if group == "all_cases" else list(range(4))
            for c in fc:
                cc = slice(None) if c is None else [c]
                X = design(a["oof_features"], b[tr], group, c)
                Z = design(a["evaluation_features"], b[ev], group, c)
                yt = truth[tr][:, cc]
                names = (
                    list(a["feature_names"]) + [f"size_{s}" for s in SIZES]
                    if c is not None
                    else list(a["feature_names"][:2])
                    + [
                        case + ":" + name
                        for case in CASES
                        for name in a["feature_names"][2:]
                    ]
                    + [f"size_{s}" for s in SIZES]
                )
                assert X.shape[1] == (25 if group == "same_case" else 79)
                feature_contract[group] = names
                predict, df, rank, sc, xx, zz = standardized_svd(X, Z, yt)
                sk = Ridge(alpha=100).fit(xx, yt)
                pp = sk.predict(zz)
                manual = predict(100.0)
                scale = truth[tr][:, cc].std(0)
                manual_max_scaled = max(
                    manual_max_scaled, float(np.max(abs(pp - manual) / scale))
                )
                assert np.allclose(pp, manual, rtol=1e-10, atol=1e-12)
                target_df = df(100.0)
                for j, caseidx in enumerate(range(4) if c is None else [c]):
                    ref = (
                        published[
                            (published["model"] == group + "/ridge100_raw")
                            & (published["repeat"] == r)
                            & (published["fold"] == f)
                            & (published["case"] == CASES[caseidx])
                        ]
                        .set_index("dataset_idx")
                        .loc[ev]
                    )
                    assert np.array_equal(
                        ref["truth"].to_numpy(), truth[ev, caseidx]
                    ) or np.allclose(
                        ref["truth"], truth[ev, caseidx], rtol=1e-14, atol=1e-15
                    )
                    err = float(
                        np.max(abs(pp[:, j] - ref.prediction.to_numpy()))
                        / truth[tr, caseidx].std()
                    )
                    baseline_max_scaled = max(baseline_max_scaled, err)
                    assert err < 1e-09
                    foldfull[gi, :, caseidx] = pp[:, j]
                for d, ix in enumerate(subset_positions):
                    pr, dfsub, rk, _, _, _ = standardized_svd(X[ix], Z, yt[ix])
                    am, status = match_df(dfsub, rk, target_df)
                    alphas = [100.0, 100.0 * len(ix) / len(tr), am]
                    for mi, alpha in enumerate(alphas):
                        pred = pr(alpha)
                        assert np.isfinite(pred).all()
                        for j, caseidx in enumerate(range(4) if c is None else [c]):
                            foldsmall[gi, mi, d, :, caseidx] = pred[:, j]
                        fitrows.append(
                            dict(
                                repeat=r,
                                fold=f,
                                draw=d,
                                group=group,
                                case="shared_all_cases" if c is None else CASES[c],
                                mode=MODES[mi],
                                alpha=float(alpha),
                                effective_df=dfsub(alpha),
                                full_effective_df=target_df,
                                subset_rank=rk,
                                df_status=status if mi == 2 else "not_matched",
                            )
                        )
        assert np.isfinite(foldfull).all() and np.isfinite(foldsmall).all()
        full[r, :, ev, :] = np.moveaxis(foldfull, 1, 0)
        small[r][:, :, :, ev, :] = foldsmall
        np.savez_compressed(
            HERE / f"repeat{r}_fold{f}.npz",
            full_predictions=foldfull,
            subset_predictions=foldsmall,
            evaluation_ids=ev,
            outer_training_ids=tr,
            subset_ids=subset_ids,
            truth=truth[ev],
            groups=np.array(GROUPS),
            modes=np.array(MODES),
            cases=np.array(CASES),
        )
        checks.append(
            dict(
                repeat=r,
                fold=f,
                n_full=len(tr),
                n_subset=75,
                n_eval=len(ev),
                oof_disjoint=True,
                production_ids_exact=True,
            )
        )
        print("BUDGET FOLD COMPLETE", r, f, flush=True)
    assert np.isfinite(full).all() and np.isfinite(small).all()
    assert baseline_max_scaled < 1e-09 and manual_max_scaled < 1e-09
    metric_checks = []
    for gi, group in enumerate(GROUPS):
        for c, case in enumerate(CASES):
            for size in ["pooled", *SIZES.tolist()]:
                ii = (
                    np.arange(940)
                    if size == "pooled"
                    else np.flatnonzero(SIZES[b] == size)
                )
                observed = np.mean(
                    [decomp(truth[ii, c], full[r, gi, ii, c])["r2"] for r in range(2)]
                )
                ref = published_metrics[
                    (published_metrics.model == group + "/ridge100_raw")
                    & (published_metrics["case"] == case)
                    & (published_metrics.size_um.astype(str) == str(size))
                ]
                assert len(ref) == 1
                delta = abs(observed - float(ref.r2_mean.iloc[0]))
                assert delta < 1e-10
                metric_checks.append(delta)
    pd.DataFrame(fitrows).to_csv(HERE / "fit_records.csv.gz", index=False)
    (HERE / "subset_provenance.json").write_text(json.dumps(subset_records, indent=2))
    (HERE / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2))
    return (
        truth,
        b,
        full,
        small,
        dict(
            baseline_max_error_target_sd=baseline_max_scaled,
            svd_vs_sklearn_max_error_target_sd=manual_max_scaled,
            published_r2_max_abs_error=max(metric_checks),
            folds=checks,
            unreachable_df_draws=sum(
                (x["df_status"] == "unreachable_minimum_norm" for x in fitrows)
            ),
            matched_df_max_abs_error=max(
                (
                    abs(x["effective_df"] - x["full_effective_df"])
                    for x in fitrows
                    if x["mode"] == "matched_df" and x["df_status"] == "matched"
                )
            ),
        ),
    )


def budget_summary(truth, b, full, small):
    metrics = []
    drawmetrics = []
    comparisons = []
    loss_full = np.mean((full - truth[None, None, :, :]) ** 2, axis=0)
    loss_small = np.mean(
        (small - truth[None, None, None, None, :, :]) ** 2, axis=(0, 3)
    )
    mae_full = np.mean(abs(full - truth[None, None, :, :]), axis=0)
    mae_small = np.mean(abs(small - truth[None, None, None, None, :, :]), axis=(0, 3))
    np.savez_compressed(
        HERE / "physical_sve_expected_losses.npz",
        full_mse=loss_full,
        subset_mse=loss_small,
        full_mae=mae_full,
        subset_mae=mae_small,
        truth=truth,
        size_bucket=b,
        groups=np.array(GROUPS),
        modes=np.array(MODES),
    )
    for si, size in enumerate(["pooled", *SIZES.tolist()]):
        ii = np.arange(940) if size == "pooled" else np.flatnonzero(SIZES[b] == size)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 990, si]))
        boot = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(b[ii] == k),
                    (NBOOT, int(np.sum(b[ii] == k))),
                    replace=True,
                )
                for k in np.unique(b[ii])
            ],
            axis=1,
        )
        for gi, group in enumerate(GROUPS):
            for c, case in enumerate(CASES):
                fullstats = [decomp(truth[ii, c], full[r, gi, ii, c]) for r in range(2)]
                base = {
                    k: float(np.mean([q[k] for q in fullstats]))
                    for k in fullstats[0]
                    if k != "n"
                }
                metrics.append(
                    dict(
                        group=group,
                        mode="full_alpha100",
                        case=case,
                        size_um=size,
                        n_sve=len(ii),
                        expected_rmse=float(np.sqrt(loss_full[gi, ii, c].mean())),
                        expected_mae=float(mae_full[gi, ii, c].mean()),
                        **base,
                    )
                )
                for mi, mode in enumerate(MODES):
                    rows = []
                    for r in range(2):
                        for d in range(DRAWS):
                            q = decomp(truth[ii, c], small[r, gi, mi, d, ii, c])
                            rows.append(q)
                            drawmetrics.append(
                                dict(
                                    group=group,
                                    mode=mode,
                                    repeat=r,
                                    draw=d,
                                    case=case,
                                    size_um=size,
                                    **q,
                                )
                            )
                    avg = {
                        k: float(np.mean([q[k] for q in rows]))
                        for k in rows[0]
                        if k != "n"
                    }
                    pcts = np.percentile([q["r2"] for q in rows], [2.5, 97.5])
                    metrics.append(
                        dict(
                            group=group,
                            mode=mode,
                            case=case,
                            size_um=size,
                            n_sve=len(ii),
                            expected_rmse=float(
                                np.sqrt(loss_small[gi, mi, ii, c].mean())
                            ),
                            expected_mae=float(mae_small[gi, mi, ii, c].mean()),
                            r2_draw_q025=float(pcts[0]),
                            r2_draw_q975=float(pcts[1]),
                            **avg,
                        )
                    )
                    a = loss_small[gi, mi, ii, c]
                    v = loss_full[gi, ii, c]
                    effect = 100 * (1 - np.sqrt(v.mean() / a.mean()))
                    dist = 100 * (1 - np.sqrt(v[boot].mean(1) / a[boot].mean(1)))
                    lo, hi = np.percentile(dist, [2.5, 97.5])
                    comparisons.append(
                        dict(
                            group=group,
                            mode=mode,
                            case=case,
                            size_um=size,
                            n_sve=len(ii),
                            full_budget_rmse_reduction_percent=effect,
                            ci_low=lo,
                            ci_high=hi,
                            rmse_75=float(np.sqrt(a.mean())),
                            rmse_full=float(np.sqrt(v.mean())),
                        )
                    )
    pd.DataFrame(metrics).to_csv(HERE / "metrics_summary.csv", index=False)
    pd.DataFrame(drawmetrics).to_csv(HERE / "draw_metrics.csv.gz", index=False)
    pd.DataFrame(comparisons).to_csv(
        HERE / "paired_budget_comparisons.csv", index=False
    )
    return (pd.DataFrame(metrics), pd.DataFrame(comparisons))


def decompose_frames():
    plans = [
        (
            STUDIES / "readout_topology_check/predictions.csv.gz",
            [
                "size_mean_fitting",
                "size_mean_outer_training",
                "size_count_ridge100",
                "original/summary_ridge100",
                "original/direct",
                "original/direct_ridge10",
                "original/direct_ridge100",
            ],
        ),
        (
            STUDIES.parent / "studies_2026_11_15/bridge/predictions_long.csv",
            [
                "maximum_predicted_grain",
                "Graph Affine bridge calibration",
                "Graph Tail-moment ridge",
            ],
        ),
        (
            STUDIES / "cross_case_readout/predictions.csv.gz",
            ["same_case/ridge100_raw", "all_cases/ridge100_raw"],
        ),
    ]
    rows = []
    for p, models in plans:
        d = frame(p)
        assert set(models) <= set(d.model)
        d = d[d.model.isin(models)]
        assert not d.duplicated(["model", "repeat", "case", "dataset_idx"]).any()
        assert d.groupby(["model", "repeat", "case"]).size().eq(940).all()
        for (model, r, c), g in d.groupby(["model", "repeat", "case"]):
            for size in ["pooled", *SIZES.tolist()]:
                q = g if size == "pooled" else g[g.size_um.eq(size)]
                assert np.isfinite(q.prediction).all()
                rows.append(
                    dict(
                        cohort="current940",
                        model=model,
                        repeat=r,
                        case=c,
                        size_um=size,
                        source=str(p),
                        **decomp(q.truth, q.prediction),
                    )
                )
    current = pd.DataFrame(rows)
    current.to_csv(HERE / "current_error_decomposition.csv", index=False)
    historical = []
    contracts = []
    predictionrows = []
    parts = read_json(H / "maximum_loss_confirmation/partitions.json")
    report = read_json(H / "direct_branch_confirmation/study_report.json")
    replay = []
    for part in parts:
        pi = part["partition"]
        stats = load_npz(
            H / f"maximum_loss_confirmation/partition{pi}/fitting_statistics.npz"
        )
        var = stats["maximum_variance"]
        paths = {}
        for seed in [101, 202]:
            if pi == 0:
                paths[seed] = {
                    "branch": H
                    / f"extreme_readout/partition0/seed{seed}/matched_direct_branch/assessment_predictions.npz",
                    "lambda0": H
                    / f"maximum_loss_confirmation/partition0/seed{seed}/lambda0/assessment_predictions.npz",
                    "source_policy": H
                    / f"maximum_loss_confirmation/partition0/seed{seed}/source_policy/assessment_predictions.npz",
                }
            else:
                paths[seed] = {
                    "branch": H
                    / f"direct_branch_confirmation/partition1/seed{seed}/matched_direct_branch/assessment_predictions.npz",
                    "lambda0": H
                    / f"direct_branch_confirmation/references/partition1/seed{seed}/lambda0/assessment_predictions.npz",
                    "source_policy": H
                    / f"direct_branch_confirmation/references/partition1/seed{seed}/source_policy/assessment_predictions.npz",
                }
        case_nmse = {model: [] for model in ["branch", "lambda0", "source_policy"]}
        for seed in [101, 202]:
            reference = None
            for model, p in paths[seed].items():
                a = load_npz(p)
                ids = a["dataset_idx"]
                b = a["size_bucket"]
                y, pr = (a["y_true"], a["y_pred"])
                assert set(ids) == set(part["assessment"]) and (
                    not set(ids) & set(part["fitting"] + part["stopping"])
                )
                assert list(a["cases"]) == CASES and np.isfinite(pr).all()
                if reference is None:
                    reference = (ids.copy(), y.copy(), b.copy())
                assert all(
                    (np.array_equal(z, v) for z, v in zip(reference, (ids, y, b)))
                )
                nmse = np.empty((5, 4))
                for c, case in enumerate(CASES):
                    for size in ["pooled", *SIZES.tolist()]:
                        ii = (
                            np.arange(len(ids))
                            if size == "pooled"
                            else np.flatnonzero(SIZES[b] == size)
                        )
                        q = decomp(y[ii, c], pr[ii, c])
                        v = (
                            np.nan
                            if size == "pooled"
                            else var[np.where(SIZES == size)[0][0], c]
                        )
                        if size != "pooled":
                            nmse[np.where(SIZES == size)[0][0], c] = q["mse"] / v
                        historical.append(
                            dict(
                                cohort=f"historical_partition{pi}_169",
                                partition=pi,
                                seed=seed,
                                model=model,
                                case=case,
                                size_um=size,
                                fitting_variance=v,
                                source=str(p),
                                **q,
                            )
                        )
                    for j, idx in enumerate(ids):
                        predictionrows.append(
                            dict(
                                partition=pi,
                                seed=seed,
                                model=model,
                                dataset_idx=int(idx),
                                case=case,
                                size_um=int(SIZES[b[j]]),
                                truth=y[j, c],
                                prediction=pr[j, c],
                            )
                        )
                case_nmse[model].append(nmse.mean(0))
                contracts.append(
                    dict(
                        partition=pi,
                        seed=seed,
                        model=model,
                        n_assessment=len(ids),
                        source=str(p),
                        raw_layout_match=True,
                    )
                )
        obs = {m: np.mean(v, axis=0) for m, v in case_nmse.items()}
        for comparator in ["lambda0", "source_policy"]:
            expected = report["partition_comparisons"][pi]["comparisons"][comparator]
            assert np.allclose(
                obs["branch"], expected["candidate_case_nmse"], rtol=1e-12
            )
            assert np.allclose(
                obs[comparator], expected["reference_case_nmse"], rtol=1e-12
            )
            gain = 1 - obs["branch"][2:].mean() / obs[comparator][2:].mean()
            assert abs(gain - expected["hcf_gain_fraction"]) < 1e-12
            replay.append(
                dict(
                    partition=pi,
                    comparator=comparator,
                    replayed_hcf_nmse_gain=gain,
                    report_gain=expected["hcf_gain_fraction"],
                )
            )
    hd = pd.DataFrame(historical)
    hd.to_csv(HERE / "historical_error_decomposition.csv", index=False)
    pd.DataFrame(predictionrows).to_csv(
        HERE / "historical_assessment_predictions.csv.gz", index=False
    )
    (HERE / "historical_parity.json").write_text(
        json.dumps(dict(contracts=contracts, report_replay=replay), indent=2)
    )
    return (current, hd, replay)


def write_report(metrics, comp, current, historical, audit):
    lines = [
        "READOUT LABEL BUDGET AND ERROR DECOMPOSITION",
        "16 November2026",
        "",
        "Complete: no encoder/tree fits or new grain inference. Existing HGB grouped-OOF",
        "features and outer-production features were held fixed. All new calibrators used",
        "only outer-training labels.100fixed75-label subsets per fold, two feature groups,",
        "three shrinkage rules. Results are descriptive development diagnostics.",
        "",
        "FULL-REFERENCE REPLAY",
        json.dumps(audit, indent=2),
        "",
        "POOLED R2 (mean over repeats/draws; no100-model prediction ensemble)",
    ]
    lines.append(
        metrics[metrics.size_um.astype(str).eq("pooled")]
        .pivot(index=["group", "mode"], columns="case", values="r2")
        .to_string(float_format=lambda x: f"{x:.6f}")
    )
    lines += ["", "500 MICROMETRES R2"]
    lines.append(
        metrics[metrics.size_um.astype(str).eq("500")]
        .pivot(index=["group", "mode"], columns="case", values="r2")
        .to_string(float_format=lambda x: f"{x:.6f}")
    )
    lines += [
        "",
        "FULL-BUDGET RMSE REDUCTION RELATIVE TO75LABELS, PERCENT",
        "Positive means fullunion isbetter. Intervals bootstrap physicalSVEs after",
        "averagingdraws/repeats withinSVE; pointwise conditional, not freshvalidation.",
    ]
    lines.append(
        comp[comp.size_um.astype(str).isin(["pooled", "500"])][
            [
                "group",
                "mode",
                "case",
                "size_um",
                "full_budget_rmse_reduction_percent",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    lines += [
        "",
        "HISTORICAL BRANCH DECOMPOSITION",
        "Differences below arebranch-minuslambda0; normalized by the ORIGINAL fitting",
        "size/casevariance, then equalweighted acrossfivecells andtwoseeds. Negative",
        "componentdifference reducesMSE. Components aredescriptive, not causes.",
    ]
    ht = historical[historical.size_um.astype(str).ne("pooled")].copy()
    cols = [
        "mse",
        "bias_squared",
        "spread_component",
        "correlation_component",
        "pearson",
    ]
    for col in cols[:-1]:
        ht[col] = ht[col] / ht.fitting_variance
    ag = ht.groupby(["partition", "model", "case"])[cols].mean()
    diff = ag.xs("branch", level="model") - ag.xs("lambda0", level="model")
    lines.append(diff.to_string(float_format=lambda x: f"{x:.6f}"))
    lines += [
        "",
        "INTERPRETATION LIMITS",
        "This intervention identifies readout-budget/shrinkage sensitivity within the",
        "fixed HGB feature system. It does not prove that adding GNN calibration data",
        "will reproduce the effect, nor isolate GNNversusHGB encoderquality. Inner",
        "feature-producing models were trained on~500SVEs; productionfeatures come",
        "from~752-SVEfits, unchanged acrossallbudgets here. That stackingdistribution",
        "difference remains shared by the compared routes.",
        "The same-alpha, per-observation-penalty andmatched-df settings answer",
        "different questions and are allreported. No setting was selected byoutcomes.",
        "Historical592/85/169 results remain separatefromcurrent940folds. Their",
        "677-rowstatisticsfiles storefitting+stoppingrawtargets, butallscalequantities",
        "andtransformIDs independentlymatchthe592fittingSVEs exactly.",
        "An MSEcorrelationcomponent can shrinkbecausepredictionSD collapses evenwhen",
        "correlationdoesnotimprove. Inspect Pearson andspreadtogether; do not label",
        "componentdifferences causal or treat oracle recalibration asavailableprediction.",
        "No manuscript, baseline model or historical artifact was changed.",
        "",
        "Files: metrics_summary.csv,paired_budget_comparisons.csv,draw_metrics.csv.gz,",
        "physical_sve_expected_losses.npz,repeatR_foldF.npz,fit_records.csv.gz,",
        "current_error_decomposition.csv,historical_error_decomposition.csv,",
        "historical_assessment_predictions.csv.gz,historical_parity.json,verification.json.",
    ]
    (HERE / "REPORT.txt").write_text("\n".join(lines) + "\n")


def main():
    start = time.time()
    sha(HERE / "PRESPEC.txt")
    sha(Path(__file__))
    truth, b, full, small, audit = budget()
    metrics, comp = budget_summary(truth, b, full, small)
    current, historical, replay = decompose_frames()
    assert np.isfinite(full).all() and np.isfinite(small).all()
    audit.update(
        status="pass",
        n_draws=DRAWS,
        n_bootstrap=NBOOT,
        seed=SEED,
        new_encoder_fits=0,
        new_tree_fits=0,
        outer_labels_used_for_fitting=False,
        finite_subset_predictions=int(small.size),
        historical_report_replay=replay,
        max_decomposition_relative_residual=float(
            max(
                np.max(
                    abs(current.decomposition_residual) / np.maximum(current.mse, 1e-30)
                ),
                np.max(
                    abs(historical.decomposition_residual)
                    / np.maximum(historical.mse, 1e-30)
                ),
            )
        ),
        elapsed_seconds=time.time() - start,
    )
    (HERE / "source_hashes.json").write_text(json.dumps(SOURCES, indent=2))
    (HERE / "verification.json").write_text(json.dumps(audit, indent=2))
    write_report(metrics, comp, current, historical, audit)
    print("COMPLETE", json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
