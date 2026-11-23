"""Gated, fixed-recipe evaluation of the twenty matched joint-branch fits."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import sys

sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import ast
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REV = ROOT / "revision_R2"
OLD = REV / "studies_2026_11_15"
RECENT = REV / "studies_2026_11_16"
TRAIN = HERE.parent / "joint_branch"
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = np.array([125, 177, 250, 354, 500])
ARMS = ("detached", "attached")
BOOT = 4000
SEED = 2026111619
FACTOR = np.sqrt(6.0) / np.pi
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source(path):
    path = Path(path)
    digest = sha(path)
    if str(path) in SOURCES:
        assert SOURCES[str(path)] == digest, (path, "source changed")
    SOURCES[str(path)] = digest
    return path


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and (not np.isfinite(x)):
        return None
    return x


def dump(path, value):
    Path(path).write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8"))


def npz(path, keys=None):
    with np.load(source(path), allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def pure_recipe(path, functions, assignments=()):
    """Load only named pure functions/constants, without executing a pipeline."""
    tree = ast.parse(source(path).read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            (isinstance(t, ast.Name) and t.id in assignments for t in node.targets)
        ):
            body.append(node)
    env = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), env)
    assert all((name in env for name in [*functions, *assignments]))
    return env


def context():
    parts = read_json(BASE / "partitions.json")
    raw = npz(BASE / "regressors/raw_features940.npz", ["raw_truth", "size_bucket"])
    target = npz(BASE / "regressors/raw_truth940.npz", ["raw_truth"])["raw_truth"]
    assert np.array_equal(raw["raw_truth"], target)
    metadata = (
        pd.read_csv(source(OLD / "data/all_sve_metadata.csv"))
        .set_index("dataset_idx")
        .loc[np.arange(940)]
    )
    buckets = raw["size_bucket"].astype(int)
    counts = metadata.grain_count.to_numpy(int)
    assert np.array_equal(metadata.size_um.to_numpy(), SIZES[buckets])
    seen = {0: [], 1: []}
    variances = np.full((2, 940, 4), np.nan)
    fit_means = {}
    for p in parts:
        fit, stop, ev = [
            np.array(p[k], dtype=int) for k in ("fitting", "stopping", "evaluation")
        ]
        assert not set(fit) & set(stop) and (not set(np.r_[fit, stop]) & set(ev))
        assert set(np.r_[fit, stop, ev]) == set(range(940))
        saved = npz(
            BASE
            / "gnn"
            / f"repeat{p['repeat']}_fold{p['fold']}"
            / "fitting_statistics.npz",
            ["mean_by_size", "checkpoint_variance_by_size"],
        )
        mu = np.stack([target[fit[buckets[fit] == b]].mean(0) for b in range(5)])
        var = np.stack(
            [target[fit[buckets[fit] == b]].var(0, ddof=1) for b in range(5)]
        )
        np.testing.assert_array_equal(mu, saved["mean_by_size"])
        np.testing.assert_array_equal(var, saved["checkpoint_variance_by_size"])
        variances[p["repeat"], ev] = var[buckets[ev]]
        fit_means[p["repeat"], p["fold"]] = mu
        seen[p["repeat"]].extend(ev.tolist())
    assert all((sorted(ids) == list(range(940)) for ids in seen.values()))
    assert np.isfinite(variances).all() and (variances > 0).all()
    br = pure_recipe(OLD / "baselines/run_moment_calibration.py", ["summaries"])[
        "summaries"
    ]
    sr = pure_recipe(
        RECENT / "boosted_node/stacking/run_stacking.py", ["summaries"], ["NAMES"]
    )
    return dict(
        parts=parts,
        truth=target,
        buckets=buckets,
        counts=counts,
        variance=variances,
        fit_means=fit_means,
        bridge=br,
        summaries=sr["summaries"],
        summary_names=sr["NAMES"],
    )


FIELD_KEYS = [
    "cases",
    "dataset_idx",
    "size_um",
    "size_bucket",
    "grain_count",
    "offsets",
    "y_true",
    "direct_prediction",
    "pred_peak",
    "true_peak",
    "grain_dataset_idx",
    "grain_id",
]


def calibrate(
    x,
    v,
    y,
    ids,
    output_ids,
    label,
    p,
    case,
    alpha=10.0,
    eligibility=False,
    fallback=None,
):
    good = np.isfinite(x).all(1)
    valid = np.isfinite(v).all(1)
    if eligibility:
        assert good.sum() >= max(10, x.shape[1] + 2), (
            label,
            "insufficient eligible stopping rows",
        )
        assert fallback is not None
    else:
        assert good.all() and valid.all()
    sc = StandardScaler().fit(x[good])
    reg = Ridge(alpha=alpha).fit(sc.transform(x[good]), y[good])
    answer = (
        np.asarray(fallback, float).copy() if fallback is not None else np.empty(len(v))
    )
    answer[valid] = reg.predict(sc.transform(v[valid]))
    rebuilt = (v[valid] - sc.mean_) / sc.scale_ @ reg.coef_ + reg.intercept_
    np.testing.assert_allclose(answer[valid], rebuilt, rtol=2e-12, atol=1e-13)
    if x.shape[1] == 1:
        zs = sc.transform(x[good])[:, 0]
        beta = zs @ (y[good] - y[good].mean()) / (zs @ zs + alpha)
        independent = y[good].mean() + sc.transform(v[valid])[:, 0] * beta
        np.testing.assert_allclose(answer[valid], independent, rtol=2e-12, atol=1e-13)
    assert np.isfinite(answer).all()
    record = dict(
        model=label,
        repeat=p["repeat"],
        fold=p["fold"],
        case=case,
        alpha=alpha,
        stopping_ids=ids.tolist(),
        calibration_ids=ids[good].tolist(),
        n_calibration=int(good.sum()),
        evaluation_ids=output_ids.tolist(),
        ineligible_evaluation_ids=output_ids[~valid].tolist(),
        fallback_count=int((~valid).sum()),
        feature_mean=sc.mean_,
        feature_sd=sc.scale_,
        coefficients=reg.coef_,
        intercept=reg.intercept_,
        negative_predictions=int((answer < 0).sum()),
        coefficient_reconstruction=True,
        scalar_closed_form_checked=x.shape[1] == 1,
    )
    return (answer, record)


def stats(y, p):
    valid = np.isfinite(y)
    assert y.shape == p.shape and np.isfinite(p).all()
    a, b = (np.where(valid, y, 0.0), np.where(valid, p, 0.0))
    return np.stack(
        [
            valid.sum(0),
            a.sum(0),
            b.sum(0),
            (a * a).sum(0),
            (b * b).sum(0),
            (a * b).sum(0),
            ((a - b) ** 2).sum(0),
            abs(a - b).sum(0),
        ],
        axis=-1,
    )


def score_stats(s):
    n, sy, sp, sy2, sp2, syp, sse, sae = s
    assert n > 0
    vy, vp = (max(0.0, sy2 / n - (sy / n) ** 2), max(0.0, sp2 / n - (sp / n) ** 2))
    cov = syp / n - sy * sp / (n * n)
    return dict(
        n=int(n),
        r2=1 - sse / (n * vy) if vy > 0 else np.nan,
        rmse=np.sqrt(max(0.0, sse / n)),
        mae=sae / n,
        bias=(sp - sy) / n,
        pearson=cov / np.sqrt(vy * vp) if vy * vp > 0 else np.nan,
        truth_sd=np.sqrt(vy),
        prediction_sd=np.sqrt(vp),
        mse=sse / n,
    )
