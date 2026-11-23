"""Fixed K30 primary940 mixture analysis. No fitting, clipping or conditioning."""
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
from pathlib import Path
import json, pickle, hashlib, time
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, logsumexp

HERE = Path(__file__).resolve().parent
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
LEVELS = np.arange(1, 100) / 100
PROBS = np.unique(np.r_[(1 - LEVELS) / 2, 0.5, (1 + LEVELS) / 2])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cdf(z, mu, sigma):
    return ndtr((np.asarray(z)[None, ...] - mu) / sigma).mean(0)


def quantile(p, mu, sigma):
    points = mu + ndtri(p) * sigma
    lo, hi = points.min(0), points.max(0)
    for _ in range(40):
        mid = (lo + hi) * 0.5
        below = cdf(mid, mu, sigma) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    q = (lo + hi) * 0.5
    err = float(np.max(np.abs(cdf(q, mu, sigma) - p)))
    assert err < 1e-9, (p, err)
    return q, err


def bounds(mapping):
    st, sc = mapping["state"], mapping["scaler"]
    pt = st["power_transformer"]
    lam = float(pt.lambdas_[0])
    im = float(pt._scaler.mean_[0])
    sd = float(pt._scaler.scale_[0])
    rawlo = 1 / (2 - lam) if lam > 2 else -np.inf
    rawhi = -1 / lam if lam < 0 else np.inf
    mathlo = ((rawlo - im) / sd - sc.mean_[0]) / sc.scale_[0]
    mathhi = ((rawhi - im) / sd - sc.mean_[0]) / sc.scale_[0]
    safe = (np.array([st["z_min_safe"], st["z_max_safe"]]) - sc.mean_[0]) / sc.scale_[0]
    return mathlo, mathhi, safe[0], safe[1]


def inverse(q, mapping):
    st, sc = mapping["state"], mapping["scaler"]
    shape = q.shape
    with np.errstate(all="ignore"):
        y = (
            st["power_transformer"]
            .inverse_transform(sc.inverse_transform(q.reshape(-1, 1)))
            .reshape(shape)
        )
    lo, hi, _, _ = bounds(mapping)
    valid = (q > lo) & (q < hi) & np.isfinite(y)
    return y, valid


def main():
    start = time.time()
    all90 = []
    allcurve = []
    allchecks = []
    ids_by_repeat = {0: [], 1: []}
    maxerr = 0.0
    containchecks = 0
    endpoints = 0
    manifest = []
    for r in range(2):
        for f in range(5):
            key = f"repeat{r}_fold{f}"
            file = HERE / f"{key}_draws.npz"
            # Materialize once: NpzFile indexing decompresses an entire member on
            # every access, especially costly for per-SVE truth identity checks.
            with np.load(file) as archive:
                d = {name: archive[name] for name in archive.files}
            maps = pickle.load(open(HERE / f"{key}_fitted_maps.pkl", "rb"))
            report = json.loads((HERE / f"{key}_capture.json").read_text())
            assert report["draw_archive_sha256"] == sha(file)
            assert report["maps_archive_sha256"] == sha(HERE / f"{key}_fitted_maps.pkl")
            assert int(d["K"]) == 30 and list(d["cases"]) == CASES
            ids, counts, off = d["dataset_idx"], d["node_counts"], d["offsets"]
            assert np.array_equal(off, np.r_[0, np.cumsum(counts)])
            assert len(np.unique(ids)) == len(ids) == report["n_sves"]
            ids_by_repeat[r].extend(ids.tolist())
            sizes = SIZES[d["size_bucket"]]
            for i in range(len(ids)):
                assert np.array_equal(
                    d["node_truth"][off[i] : off[i + 1]].max(0), d["graph_truth"][i]
                )
            print(key, "identity checks passed", flush=True)
            saved = {}
            curve_saved = {}
            graph_quantiles = {}
            for level in ["node", "graph"]:
                n = len(d[level + "_truth"])
                ng = len(ids)
                for c, case in enumerate(CASES):
                    mapping = maps[level + "_" + case]
                    lo, hi, slo, shi = bounds(mapping)
                    y = d[level + "_truth"][:, c].astype(float)
                    zy = (
                        mapping["scaler"]
                        .transform(
                            mapping["state"]["power_transformer"].transform(y[:, None])
                        )
                        .ravel()
                    )
                    np.testing.assert_array_equal(
                        zy, d[level + "_target_mathematical"][:, c]
                    )
                    mu = d[level][:, :, c].astype(float)
                    ell = d[level][:, :, c + 4].astype(float)
                    sigma = np.exp(np.maximum(ell, -3))
                    assert np.isfinite(mu).all() and np.isfinite(sigma).all()
                    pit = cdf(zy, mu, sigma)
                    fl = cdf(np.full(n, lo), mu, sigma)
                    fu = cdf(np.full(n, hi), mu, sigma)
                    fsl = cdf(np.full(n, slo), mu, sigma)
                    fsu = cdf(np.full(n, shi), mu, sigma)
                    A = np.mean(sigma ** 2, axis=0)
                    E = np.var(mu, axis=0, ddof=0)
                    V = A + E
                    frac = E / V
                    nll = (
                        -logsumexp(
                            -0.5 * np.log(2 * np.pi)
                            - np.log(sigma)
                            - 0.5 * ((zy[None, :] - mu) / sigma) ** 2,
                            axis=0,
                        )
                        + np.log(30)
                    )
                    nll_supervised = (
                        -logsumexp(
                            -0.5 * np.log(2 * np.pi)
                            - np.log(sigma)
                            - 0.5
                            * (
                                (d[level + "_target_transformed"][None, :, c] - mu)
                                / sigma
                            )
                            ** 2,
                            axis=0,
                        )
                        + np.log(30)
                    )
                    qprobs = (
                        PROBS
                        if level == "graph"
                        else np.array([0.005, 0.05, 0.95, 0.995])
                    )
                    qs = np.empty((len(qprobs), n))
                    qraw = np.empty_like(qs)
                    qvalid = np.empty_like(qs, dtype=bool)
                    for j, p in enumerate(qprobs):
                        for a in range(0, n, 12000):
                            b = min(a + 12000, n)
                            qs[j, a:b], err = quantile(p, mu[:, a:b], sigma[:, a:b])
                            maxerr = max(maxerr, err)
                        qraw[j], qvalid[j] = inverse(qs[j], mapping)
                    endpoints += qraw.size
                    # Finite inverses at the outer .005/.995 bounds guarantee finite
                    # intermediate inverse values whenever their support checks pass.
                    lowidx = np.where(np.isclose(qprobs, 0.05, atol=1e-14))[0][0]
                    upidx = np.where(np.isclose(qprobs, 0.95, atol=1e-14))[0][0]
                    low, up = qraw[lowidx], qraw[upidx]
                    valid90 = qvalid[lowidx] & qvalid[upidx] & (up >= low)
                    hit90 = (pit >= 0.05) & (pit <= 0.95)
                    assert np.array_equal(
                        hit90[valid90], ((low <= y) & (y <= up))[valid90]
                    )
                    containchecks += int(valid90.sum())
                    width = up - low
                    score = (
                        width + 20 * np.maximum(low - y, 0) + 20 * np.maximum(y - up, 0)
                    )
                    location, locationvalid = inverse(mu.mean(0), mapping)
                    safe90 = (0.05 >= fsl) & (0.95 <= fsu)
                    # All entries are retained. Undefined raw intervals stay NaN and
                    # aggregate raw metrics become unavailable if any member is invalid.
                    width = np.where(valid90, width, np.nan)
                    score = np.where(valid90, score, np.nan)
                    for name, value in dict(
                        pit=pit,
                        nll_z=nll,
                        nll_supervised_z=nll_supervised,
                        conditional_variance=A,
                        location_variance=E,
                        total_variance=V,
                        dropout_fraction=frac,
                        lower90=low,
                        upper90=up,
                        valid90=valid90,
                        hit90=hit90,
                        width90=width,
                        score90=score,
                        location_raw=location,
                        location_valid=locationvalid,
                        math_tail_mass=fl + 1 - fu,
                        safe_exceed90=~safe90,
                        negative_lower90=low < 0,
                        quantile_probabilities=qprobs,
                        quantile_transformed=qs,
                        quantile_raw=qraw,
                        quantile_valid=qvalid,
                        floor_draw_fraction=np.mean(ell < -3, axis=0),
                    ).items():
                        saved[level + "_" + case + "_" + name] = value
                    if level == "graph":
                        graph_quantiles[case + "_z"] = qs
                        graph_quantiles[case + "_raw"] = qraw
                        graph_quantiles[case + "_valid"] = qvalid
                    hcounts = np.empty((99, ng), dtype=int)
                    badcounts = np.empty_like(hcounts)
                    guards = np.empty_like(hcounts)
                    for j, P in enumerate(LEVELS):
                        alpha = (1 - P) / 2
                        hit = (pit >= alpha) & (pit <= 1 - alpha)
                        good = (alpha > fl) & (1 - alpha < fu)
                        guard = (alpha >= fsl) & (1 - alpha <= fsu)
                        if level == "graph":
                            il = np.argmin(abs(qprobs - alpha))
                            iu = np.argmin(abs(qprobs - (1 - alpha)))
                            good &= qvalid[il] & qvalid[iu] & (qraw[iu] >= qraw[il])
                            rawhit = (qraw[il] <= y) & (y <= qraw[iu])
                            assert np.array_equal(rawhit[good], hit[good])
                            containchecks += int(good.sum())
                        else:
                            extremevalid = qvalid[0] & qvalid[-1]
                            # Check individual levels only where the extreme inverse
                            # failed; never classify a narrower finite interval invalid.
                            need = good & ~extremevalid
                            if need.any():
                                ql, e = quantile(alpha, mu[:, need], sigma[:, need])
                                qu, e2 = quantile(
                                    1 - alpha, mu[:, need], sigma[:, need]
                                )
                                maxerr = max(maxerr, e, e2)
                                yl, vl = inverse(ql, mapping)
                                yu, vu = inverse(qu, mapping)
                                good[need] &= vl & vu & (yu >= yl)
                        if level == "node":
                            hcounts[j] = np.add.reduceat(hit.astype(int), off[:-1])
                            badcounts[j] = np.add.reduceat(
                                (~good).astype(int), off[:-1]
                            )
                            guards[j] = np.add.reduceat((~guard).astype(int), off[:-1])
                        else:
                            hcounts[j] = hit.astype(int)
                            badcounts[j] = (~good).astype(int)
                            guards[j] = (~guard).astype(int)
                    den = counts if level == "node" else np.ones(ng, dtype=int)
                    curve_saved[level + "_" + case + "_hits"] = hcounts
                    curve_saved[level + "_" + case + "_invalid"] = badcounts
                    curve_saved[level + "_" + case + "_guard"] = guards
                    for i, idx in enumerate(ids):
                        sl = (
                            slice(off[i], off[i + 1])
                            if level == "node"
                            else slice(i, i + 1)
                        )
                        ok = bool(valid90[sl].all())
                        nn = int(den[i])
                        all90.append(
                            dict(
                                repeat=r,
                                fold=f,
                                dataset_idx=int(idx),
                                size_um=int(sizes[i]),
                                level=level,
                                case=case,
                                n=nn,
                                hits90=int(hit90[sl].sum()),
                                coverage90=float(hit90[sl].mean()) if ok else np.nan,
                                pit_coverage90=float(hit90[sl].mean()),
                                invalid90=int((~valid90[sl]).sum()),
                                guard_exceed90=int((~safe90[sl]).sum()),
                                negative_lower90=int((low[sl] < 0).sum()),
                                mean_width90=float(np.mean(width[sl]))
                                if ok
                                else np.nan,
                                mean_score90=float(np.mean(score[sl]))
                                if ok
                                else np.nan,
                                mean_nll_z=float(np.mean(nll[sl])),
                                mean_nll_supervised_z=float(
                                    np.mean(nll_supervised[sl])
                                ),
                                mean_conditional_variance=float(A[sl].mean()),
                                mean_location_variance=float(E[sl].mean()),
                                mean_total_variance=float(V[sl].mean()),
                                mean_dropout_fraction=float(frac[sl].mean()),
                                floor_draw_fraction=float(np.mean(ell[:, sl] < -3)),
                                mean_math_tail_mass=float((fl[sl] + 1 - fu[sl]).mean()),
                                max_math_tail_mass=float((fl[sl] + 1 - fu[sl]).max()),
                                y_true=float(y[i]) if level == "graph" else np.nan,
                                lower90=float(low[i]) if level == "graph" else np.nan,
                                upper90=float(up[i]) if level == "graph" else np.nan,
                                location_raw=float(location[i])
                                if level == "graph"
                                else np.nan,
                            )
                        )
                    # Both fold totals and fold/size totals, later combined by counts.
                    for label, mask in [("all", np.ones(ng, bool))] + [
                        (str(s), sizes == s) for s in SIZES
                    ]:
                        if not mask.any():
                            continue
                        for j, P in enumerate(LEVELS):
                            bad = int(badcounts[j, mask].sum())
                            nobs = int(den[mask].sum())
                            nhit = int(hcounts[j, mask].sum())
                            allcurve.append(
                                dict(
                                    repeat=r,
                                    fold=f,
                                    level=level,
                                    case=case,
                                    size=label,
                                    nominal=P,
                                    n_sves=int(mask.sum()),
                                    n=nobs,
                                    hits=nhit,
                                    invalid=bad,
                                    guard_exceed=int(guards[j, mask].sum()),
                                    pit_coverage=nhit / nobs,
                                    coverage=nhit / nobs if not bad else np.nan,
                                    sum_sve_coverage=float(
                                        np.sum(hcounts[j, mask] / den[mask])
                                    ),
                                    mean_sve_coverage=float(
                                        np.mean(hcounts[j, mask] / den[mask])
                                    )
                                    if not bad
                                    else np.nan,
                                )
                            )
                    allchecks.append(
                        dict(
                            repeat=r,
                            fold=f,
                            level=level,
                            case=case,
                            n=n,
                            math_lower_z=lo,
                            math_upper_z=hi,
                            invalid90=int((~valid90).sum()),
                            invalid99=int((~(qvalid[0] & qvalid[-1])).sum()),
                            guard_exceed90=int((~safe90).sum()),
                            negative_lower90=int((low < 0).sum()),
                            max_math_tail_mass=float((fl + 1 - fu).max()),
                            max_nll_effect_of_target_precision=float(
                                np.max(np.abs(nll - nll_supervised))
                            ),
                        )
                    )
                    print(
                        key,
                        level,
                        case,
                        "coverage90",
                        float(hit90.mean()),
                        "invalid90",
                        int((~valid90).sum()),
                        flush=True,
                    )
            np.savez_compressed(
                HERE / f"{key}_metrics.npz",
                **saved,
                dataset_idx=ids,
                node_counts=counts,
                offsets=off,
                size_um=sizes,
                cases=np.array(CASES),
            )
            np.savez_compressed(
                HERE / f"{key}_coverage_counts.npz",
                **curve_saved,
                dataset_idx=ids,
                node_counts=counts,
                size_um=sizes,
                nominal=LEVELS,
            )
            np.savez_compressed(
                HERE / f"{key}_graph_quantiles.npz",
                **graph_quantiles,
                dataset_idx=ids,
                size_um=sizes,
                probability=PROBS,
            )
            manifest.append(
                dict(
                    key=key,
                    draw_sha256=sha(file),
                    maps_sha256=sha(HERE / f"{key}_fitted_maps.pkl"),
                    metrics_sha256=sha(HERE / f"{key}_metrics.npz"),
                )
            )
    for r, values in ids_by_repeat.items():
        assert sorted(values) == list(range(940)), r
    per = pd.DataFrame(all90)
    per.to_csv(HERE / "per_sve90.csv", index=False, float_format="%.17g")
    curves = pd.DataFrame(allcurve)
    curves.to_csv(HERE / "coverage_by_fold.csv", index=False, float_format="%.17g")
    agg = []
    for keys, g in curves.groupby(
        ["repeat", "level", "case", "size", "nominal"], sort=False
    ):
        r, lev, case, size, P = keys
        n = int(g.n.sum())
        ns = int(g.n_sves.sum())
        bad = int(g.invalid.sum())
        hit = int(g.hits.sum())
        agg.append(
            dict(
                repeat=r,
                level=lev,
                case=case,
                size=size,
                nominal=P,
                n_sves=ns,
                n=n,
                hits=hit,
                invalid=bad,
                guard_exceed=int(g.guard_exceed.sum()),
                pit_coverage=hit / n,
                coverage=hit / n if not bad else np.nan,
                mean_sve_coverage=float(g.sum_sve_coverage.sum() / ns)
                if not bad
                else np.nan,
            )
        )
    pd.DataFrame(agg).to_csv(
        HERE / "coverage_by_repeat_size.csv", index=False, float_format="%.17g"
    )
    scores = []
    for r in range(2):
        for level in ["node", "graph"]:
            for case in CASES:
                base = per[
                    (per.repeat == r) & (per.level == level) & (per.case == case)
                ]
                for label, g in [("all", base)] + [
                    (str(s), base[base.size_um == s]) for s in SIZES
                ]:
                    weights = g.n.to_numpy()
                    bad = int(g.invalid90.sum())
                    n = int(weights.sum())
                    row = dict(
                        repeat=r,
                        level=level,
                        case=case,
                        size=label,
                        n_sves=len(g),
                        n=n,
                        hits90=int(g.hits90.sum()),
                        invalid90=bad,
                        guard_exceed90=int(g.guard_exceed90.sum()),
                        negative_lower90=int(g.negative_lower90.sum()),
                        coverage90=float(g.hits90.sum() / n) if not bad else np.nan,
                        mean_sve_coverage90=float(g.pit_coverage90.mean())
                        if not bad
                        else np.nan,
                        max_math_tail_mass=float(g.max_math_tail_mass.max()),
                    )
                    for col in [
                        "mean_width90",
                        "mean_score90",
                        "mean_nll_z",
                        "mean_nll_supervised_z",
                        "mean_conditional_variance",
                        "mean_location_variance",
                        "mean_total_variance",
                        "mean_dropout_fraction",
                        "floor_draw_fraction",
                        "mean_math_tail_mass",
                    ]:
                        row[col] = float(np.average(g[col], weights=weights))
                    row["ratio_of_mean_variances"] = (
                        row["mean_location_variance"] / row["mean_total_variance"]
                    )
                    scores.append(row)
    pd.DataFrame(scores).to_csv(
        HERE / "scores90_by_repeat_size.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(allchecks).to_csv(
        HERE / "support_and_precision_checks.csv", index=False, float_format="%.17g"
    )
    summary = dict(
        status="complete",
        K=30,
        unique_sves_each_repeat=940,
        repeats=2,
        independent_specimens=940,
        analysis_sha256=sha(__file__),
        capture_sha256=sha(HERE / "capture.py"),
        maximum_quantile_cdf_error=maxerr,
        raw_pit_containment_checks=containchecks,
        finite_inverse_endpoints_explicitly_computed=endpoints,
        exact_identity_truth_checks=True,
        scale_floor=-3,
        variance_ddof=0,
        analysis_seconds=time.time() - start,
        raw_intervals="Exact fitted inverse; no clipping or support conditioning; all observations retained.",
        target_precision="Mathematical float64 transformations for physical PIT/scores; original float32 supervised NLL archived separately.",
        grain_curve_finiteness="99% outer endpoint inversion plus exact mathematical-support CDF checks; narrower endpoints explicitly checked if extreme inverse fails.",
        scope="Conditional on already developed recipe; repeated predictions are not independent SVEs. Direct-head uncertainty does not describe bridge/readout/residual predictions.",
        manifests=manifest,
    )
    (HERE / "analysis_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
