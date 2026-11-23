"""Bootstrap expected quantile discrepancy (Murphy, Tawn and Varty, 2025)."""
import numpy as np

try:
    from . import core
except ImportError:
    import core


def discrepancy(excess, seed, B=100, m=500):
    """Compare each resample to its own fitted GPD, at j/(m+1)."""
    y = np.asarray(excess, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(y, size=(B, len(y)), replace=True)
    fits = core.fit_gpd_array(samples)
    order = np.sort(samples, axis=1)
    p = np.arange(1, m + 1, dtype=float) / (m + 1)
    positions = p * (len(y) - 1)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    frac = positions - lower
    empirical = order[:, lower] * (1 - frac) + order[:, upper] * frac
    shape = fits[:, 0, None]
    scale = fits[:, 1, None]
    a = -np.log1p(-p)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        theoretical = np.where(
            abs(shape) < 1e-7, scale * a, scale * np.expm1(shape * a) / shape
        )
    valid = fits[:, 3].astype(bool) & np.isfinite(theoretical).all(axis=1)
    # A failed resample makes this candidate inadmissible. It is not silently
    # deleted from the bootstrap average, and no extra resampling is performed.
    score = float(np.mean(abs(theoretical - empirical))) if valid.all() else None
    return dict(
        score=score,
        failures=int((~valid).sum()),
        boundary_fits=int(fits[:, 4].sum()),
        bootstrap_count=int(B),
        quantile_count=int(m),
    )


def select_eqd(values, seed, candidate_list=None, B=100, m=500):
    x = np.asarray(values, dtype=float)
    cs = core.candidates(x) if candidate_list is None else candidate_list
    scores = []
    for c in cs:
        # Stable candidate-index seeds make runs independent of process order.
        d = discrepancy(
            c["excess"], np.random.SeedSequence([int(seed), int(c["index"])]), B, m
        )
        scores.append(
            dict(
                index=c["index"], quantile=c["quantile"], threshold=c["threshold"], **d
            )
        )
    eligible = [
        i
        for i, d in enumerate(scores)
        if d["score"] is not None and cs[i]["fit"]["success"]
    ]
    if not eligible:
        return dict(
            selected_index=-1,
            selected_threshold=float(x.max()),
            selected_quantile=1.0,
            raw_maximum=float(x.max()),
            status="maximum_fallback",
            scores=scores,
            bootstrap_failures=sum(d["failures"] for d in scores),
            boundary_fits=sum(d["boundary_fits"] for d in scores),
        )
    j = min(eligible, key=lambda i: scores[i]["score"])
    pred, status = core.finite_maximum(cs[j])
    return dict(
        selected_index=int(cs[j]["index"]),
        selected_threshold=cs[j]["threshold"],
        selected_quantile=cs[j]["quantile"],
        raw_maximum=pred,
        status=status,
        scores=scores,
        bootstrap_failures=sum(d["failures"] for d in scores),
        boundary_fits=sum(d["boundary_fits"] for d in scores),
    )
