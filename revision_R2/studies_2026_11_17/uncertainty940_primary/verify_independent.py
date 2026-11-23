"""Independent verification using scalar erfc CDF, Brent roots and explicit YJ.

Does not import analyze.py or its numerical functions. No inference or fitting.
"""
from pathlib import Path
import os

os.environ.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
import math, json, pickle, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scalar_cdf(x, m, s):
    return math.fsum(
        0.5 * math.erfc(-(x - float(a)) / (float(b) * math.sqrt(2)))
        for a, b in zip(m, s)
    ) / len(m)


def explicit_inverse(z, mp):
    pt = mp["state"]["power_transformer"]
    sc = mp["scaler"]
    lam = float(pt.lambdas_[0])
    w = (np.asarray(z) * sc.scale_[0] + sc.mean_[0]) * pt._scaler.scale_[
        0
    ] + pt._scaler.mean_[0]
    pos = w >= 0
    out = np.empty_like(w)
    with np.errstate(all="ignore"):
        out[pos] = (
            np.expm1(w[pos])
            if abs(lam) < 1e-15
            else np.expm1(np.log1p(lam * w[pos]) / lam)
        )
        out[~pos] = (
            -np.expm1(-w[~pos])
            if abs(lam - 2) < 1e-15
            else -np.expm1(np.log1p(-(2 - lam) * w[~pos]) / (2 - lam))
        )
    return out


def explicit_forward(y, mp):
    pt = mp["state"]["power_transformer"]
    sc = mp["scaler"]
    lam = float(pt.lambdas_[0])
    y = np.asarray(y)
    pos = y >= 0
    w = np.empty_like(y)
    w[pos] = (
        np.log1p(y[pos]) if abs(lam) < 1e-15 else np.expm1(lam * np.log1p(y[pos])) / lam
    )
    w[~pos] = (
        -np.log1p(-y[~pos])
        if abs(lam - 2) < 1e-15
        else -np.expm1((2 - lam) * np.log1p(-y[~pos])) / (2 - lam)
    )
    return ((w - pt._scaler.mean_[0]) / pt._scaler.scale_[0] - sc.mean_[0]) / sc.scale_[
        0
    ]


def main():
    per = pd.read_csv(HERE / "per_sve90.csv", float_precision="round_trip")
    aggregate = pd.read_csv(
        HERE / "scores90_by_repeat_size.csv", float_precision="round_trip"
    )
    points = []
    seen = {0: [], 1: []}
    truth_by_id = {}
    maxroot = 0.0
    maxcdf = 0.0
    maxinv = 0.0
    maxforward = 0.0
    nraw = 0
    nalgebra = 0
    ndomain = 0
    fingerprints = []
    for r in range(2):
        for f in range(5):
            key = f"repeat{r}_fold{f}"
            with np.load(HERE / f"{key}_draws.npz") as a:
                d = {name: a[name] for name in a.files}
            with np.load(HERE / f"{key}_metrics.npz") as a:
                v = {name: a[name] for name in a.files}
            cc = np.load(HERE / f"{key}_coverage_counts.npz")
            qg = np.load(HERE / f"{key}_graph_quantiles.npz")
            maps = pickle.load(open(HERE / f"{key}_fitted_maps.pkl", "rb"))
            rep = json.loads((HERE / f"{key}_capture.json").read_text())
            assert rep["draw_archive_sha256"] == digest(HERE / f"{key}_draws.npz")
            assert rep["maps_archive_sha256"] == digest(HERE / f"{key}_fitted_maps.pkl")
            ids = d["dataset_idx"]
            off = d["offsets"]
            counts = d["node_counts"]
            seen[r].extend(ids.tolist())
            assert np.array_equal(off, np.r_[0, np.cumsum(counts)])
            assert np.array_equal(
                d["node_grain_id"], np.concatenate([np.arange(n) for n in counts])
            )
            for i, idx in enumerate(ids):
                truth = d["node_truth"][off[i] : off[i + 1]]
                assert np.array_equal(truth.max(0), d["graph_truth"][i])
                if idx in truth_by_id:
                    assert np.array_equal(truth, truth_by_id[idx])
                else:
                    truth_by_id[idx] = truth.copy()
            assert (
                rep["K"] == 30
                and rep["batch_size"] == 16
                and rep["seed"] == 2026111700 + 100 * r + f
            )
            assert all(
                t["exact_supervised_reproduction"]
                for t in rep["transform_checks"].values()
            )
            for level in ["node", "graph"]:
                for c, case in enumerate(CASES):
                    prefix = level + "_" + case + "_"
                    mp = maps[level + "_" + case]
                    m = d[level][:, :, c].astype(float)
                    s = np.exp(np.maximum(d[level][:, :, c + 4].astype(float), -3))
                    y = d[level + "_truth"][:, c].astype(float)
                    z = d[level + "_target_mathematical"][:, c]
                    n = len(y)
                    indices = np.unique(np.linspace(0, n - 1, 9, dtype=int))
                    # Different full-array variance algebra from central deviations.
                    aa = np.einsum("ki,ki->i", s, s) / 30
                    ee = np.einsum("ki,ki->i", m, m) / 30 - (m.sum(0) / 30) ** 2
                    np.testing.assert_allclose(
                        aa, v[prefix + "conditional_variance"], rtol=1e-13, atol=1e-15
                    )
                    np.testing.assert_allclose(
                        ee, v[prefix + "location_variance"], rtol=1e-8, atol=5e-14
                    )
                    np.testing.assert_allclose(
                        aa + ee, v[prefix + "total_variance"], rtol=1e-10, atol=5e-14
                    )
                    np.testing.assert_allclose(
                        ee / (aa + ee),
                        v[prefix + "dropout_fraction"],
                        rtol=1e-8,
                        atol=5e-12,
                    )
                    nalgebra += n
                    z2 = explicit_forward(y, mp)
                    maxforward = max(maxforward, float(np.max(abs(z2 - z))))
                    np.testing.assert_allclose(z2, z, rtol=2e-10, atol=2e-9)
                    probs = v[prefix + "quantile_probabilities"]
                    q = v[prefix + "quantile_transformed"]
                    raw = v[prefix + "quantile_raw"]
                    valid = v[prefix + "quantile_valid"]
                    inv = explicit_inverse(q, mp)
                    assert np.array_equal(np.isfinite(inv), np.isfinite(raw))
                    np.testing.assert_allclose(
                        inv[valid], raw[valid], rtol=2e-9, atol=5e-12
                    )
                    if valid.any():
                        maxinv = max(
                            maxinv, float(np.max(abs(inv[valid] - raw[valid])))
                        )
                    back = explicit_forward(raw[valid], mp)
                    np.testing.assert_allclose(back, q[valid], rtol=2e-8, atol=2e-8)
                    nraw += int(valid.sum())
                    # Direct checks on all stored endpoints and actual target inclusion.
                    il = np.argmin(abs(probs - 0.05))
                    iu = np.argmin(abs(probs - 0.95))
                    ok = valid[il] & valid[iu]
                    direct = (raw[il] <= y) & (y <= raw[iu])
                    pit90 = (v[prefix + "pit"] >= 0.05) & (v[prefix + "pit"] <= 0.95)
                    assert np.array_equal(direct[ok], pit90[ok])
                    for j in range(len(ids)):
                        sl = (
                            slice(off[j], off[j + 1])
                            if level == "node"
                            else slice(j, j + 1)
                        )
                        row = per[
                            (per.repeat == r)
                            & (per.fold == f)
                            & (per.dataset_idx == ids[j])
                            & (per.level == level)
                            & (per.case == case)
                        ].iloc[0]
                        assert int(row.hits90) == int(pit90[sl].sum()) and int(
                            row.n
                        ) == len(y[sl])
                        np.testing.assert_allclose(
                            row.mean_dropout_fraction,
                            np.mean(v[prefix + "dropout_fraction"][sl]),
                            rtol=1e-13,
                        )
                    for idx in indices:
                        pit = scalar_cdf(z[idx], m[:, idx], s[:, idx])
                        e = abs(pit - v[prefix + "pit"][idx])
                        maxcdf = max(maxcdf, e)
                        assert e < 1e-14
                        terms = [
                            -0.5 * math.log(2 * math.pi)
                            - math.log(float(si))
                            - 0.5 * ((float(z[idx]) - float(mi)) / float(si)) ** 2
                            for mi, si in zip(m[:, idx], s[:, idx])
                        ]
                        top = max(terms)
                        indnll = -(
                            top
                            + math.log(math.fsum(math.exp(t - top) for t in terms) / 30)
                        )
                        assert abs(indnll - v[prefix + "nll_z"][idx]) < 1e-11
                        # Selected fixed quantiles cover both central and extreme levels.
                        checkp = [0.005, 0.05, 0.95, 0.995] + (
                            [0.5] if level == "graph" else []
                        )
                        for p in checkp:
                            qj = int(np.argmin(abs(probs - p)))
                            expected = q[qj, idx]
                            a = float(m[:, idx].min() - 12 * s[:, idx].max())
                            b = float(m[:, idx].max() + 12 * s[:, idx].max())
                            root = brentq(
                                lambda t: scalar_cdf(t, m[:, idx], s[:, idx]) - p,
                                a,
                                b,
                                xtol=2e-13,
                                rtol=1e-14,
                            )
                            err = abs(root - expected)
                            maxroot = max(maxroot, err)
                            assert err < 2e-9
                            points.append(
                                dict(
                                    repeat=r,
                                    fold=f,
                                    level=level,
                                    case=case,
                                    observation=int(idx),
                                    probability=p,
                                    saved_quantile=expected,
                                    brent_quantile=root,
                                    absolute_error=err,
                                )
                            )
                    # Independent support formula; conservative guards never substituted.
                    pt = mp["state"]["power_transformer"]
                    lam = float(pt.lambdas_[0])
                    sc = mp["scaler"]
                    lower = (
                        (1 / (2 - lam) - pt._scaler.mean_[0]) / pt._scaler.scale_[0]
                        if lam > 2
                        else -np.inf
                    )
                    upper = (
                        (-1 / lam - pt._scaler.mean_[0]) / pt._scaler.scale_[0]
                        if lam < 0
                        else np.inf
                    )
                    lower = (lower - sc.mean_[0]) / sc.scale_[0]
                    upper = (upper - sc.mean_[0]) / sc.scale_[0]
                    mathematical = (q > lower) & (q < upper) & np.isfinite(raw)
                    assert np.array_equal(mathematical, valid)
                    ndomain += valid.size
            fingerprints.append(
                dict(
                    key=key,
                    draw_sha256=digest(HERE / f"{key}_draws.npz"),
                    metrics_sha256=digest(HERE / f"{key}_metrics.npz"),
                )
            )
    assert sorted(seen[0]) == sorted(seen[1]) == list(range(940))
    for _, row in aggregate.iterrows():
        mask = (
            (per.repeat == row["repeat"])
            & (per.level == row.level)
            & (per.case == row.case)
        )
        if row["size"] != "all":
            mask &= per.size_um == int(row["size"])
        g = per[mask]
        assert len(g) == row.n_sves and g.n.sum() == row.n
        assert g.hits90.sum() == row.hits90 and g.invalid90.sum() == row.invalid90
        np.testing.assert_allclose(
            np.average(g.mean_dropout_fraction, weights=g.n),
            row.mean_dropout_fraction,
            rtol=1e-13,
        )
        if row.invalid90 == 0:
            np.testing.assert_allclose(
                g.hits90.sum() / g.n.sum(), row.coverage90, rtol=1e-13
            )
            np.testing.assert_allclose(
                g.pit_coverage90.mean(), row.mean_sve_coverage90, rtol=1e-13
            )
    pd.DataFrame(points).to_csv(
        HERE / "independent_quantile_checks.csv", index=False, float_format="%.17g"
    )
    out = dict(
        status="pass",
        unique_sves=940,
        repeats_verified=2,
        repeat_truth_identity=True,
        scalar_erfc_and_Brent_quantile_checks=len(points),
        maximum_Brent_quantile_difference=maxroot,
        maximum_scalar_erfc_PIT_difference=maxcdf,
        maximum_explicit_inverse_difference=maxinv,
        maximum_explicit_forward_difference=maxforward,
        full_array_variance_observations_verified=nalgebra,
        explicit_inverse_endpoint_checks=nraw,
        mathematical_support_endpoint_checks=ndomain,
        aggregate_denominators_and_weighting_verified=True,
        implementation_independence="No analysis imports; math.erfc scalar mixture CDF, scipy.optimize.brentq, explicit log1p/expm1 YJ, second-moment variance algebra.",
        limitation="Numerical verification conditional on archived model/data identity and the capture state/replay assertions; does not establish external-data calibration.",
        verification_script_sha256=digest(__file__),
        analysis_script_sha256=digest(HERE / "analyze.py"),
        files=fingerprints,
    )
    (HERE / "independent_verification.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
