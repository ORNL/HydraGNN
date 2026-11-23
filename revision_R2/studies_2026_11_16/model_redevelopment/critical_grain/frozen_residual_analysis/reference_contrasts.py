"""Post-gate, prespecified fixed-route context; no model fitting or selection."""
import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import analyze as a
import verify_analysis as v


def main():
    a.gate()
    a.verify_references()
    assert a.load(a.HERE / "independent_verification.json")["status"] == "pass"
    ctx = a.context()
    models = a.load_all_predictions()
    draws = a.bootstrap()
    dest = a.HERE / "fixed_reference_contrasts"
    dest.mkdir(exist_ok=True)
    existing = pd.read_csv(a.HERE / "paired_maximum_comparisons.csv")
    already = set(zip(existing.model, existing.reference))
    comparisons = [
        (arm + "/" + suffix, reference)
        for arm in a.ARMS
        for suffix in a.SUFFIXES
        for reference in ("original/bridge/ridge10", "original/summary/ridge100")
        if (arm + "/" + suffix, reference) not in already
    ]
    moments = {}
    y = ctx["truth"]
    for name in set(sum(([m, r] for m, r in comparisons), [])):
        p = models[name]
        yy = np.broadcast_to(y, p.shape)
        moments[name] = np.stack(
            [
                np.ones_like(yy),
                yy,
                p,
                yy * yy,
                p * p,
                yy * p,
                (p - yy) ** 2,
                abs(p - yy),
            ],
            -1,
        )
    paired = a.paired_stats(moments, comparisons, draws)
    paired.to_csv(dest / "paired_maximum_comparisons.csv", index=False)
    groups = v.scopes(ctx["buckets"], a.SIZES)
    weights = {str(s): v.multiplicities(d, 940) for s, d in zip(a.SIZES, draws)}
    weights["pooled"] = sum(weights.values())
    old_here = v.HERE
    try:
        v.HERE = dest
        verified = v.verify_maximum_pairs(a, models, y, groups, weights)
    finally:
        v.HERE = old_here
    normalized = {
        name: ((p - y) ** 2 / ctx["variance"]).mean(0) for name, p in models.items()
    }
    rows = []
    for arm in a.ARMS:
        name = arm + "/grain_maximum/raw"
        for reference in (
            "cached_original/grain_maximum/raw",
            "original/legacy_direct/raw",
            "original/bridge/ridge10",
            "original/summary/ridge100",
            "reference/size_mean_union",
        ):
            aa, bb = normalized[name], normalized[reference]
            ra, rb = [
                np.mean([x[groups[str(s)]].mean() for s in a.SIZES]) for x in (aa, bb)
            ]
            ba, br = [sum(x[d].mean((1, 2)) for d in draws) / 5 for x in (aa, bb)]
            wa, wr = [
                sum(
                    weights[str(s)] @ x.mean(1) / len(groups[str(s)]) / 5
                    for s in a.SIZES
                )
                for x in (aa, bb)
            ]
            absolute = np.percentile(br - ba, [2.5, 97.5])
            percent = np.percentile(100 * (1 - ba / br), [2.5, 97.5])
            np.testing.assert_allclose(
                absolute, np.percentile(wr - wa, [2.5, 97.5]), rtol=2e-10, atol=1e-12
            )
            np.testing.assert_allclose(
                percent,
                np.percentile(100 * (1 - wa / wr), [2.5, 97.5]),
                rtol=2e-10,
                atol=1e-10,
            )
            rows.append(
                dict(
                    model=name,
                    reference=reference,
                    model_risk=ra,
                    reference_risk=rb,
                    absolute_risk_reduction=rb - ra,
                    absolute_ci_low=absolute[0],
                    absolute_ci_high=absolute[1],
                    normalized_mse_reduction_percent=100 * (1 - ra / rb),
                    reduction_percent_ci_low=percent[0],
                    reduction_percent_ci_high=percent[1],
                    comparison="fixed secondary context",
                )
            )
    pd.DataFrame(rows).to_csv(dest / "equal20_risk_comparisons.csv", index=False)
    a.dump(
        dest / "verification.json",
        dict(
            status="pass",
            core_independent_verification_required=True,
            additional_paired_rows=verified,
            equal20_context_rows=len(rows),
            independent_multiplicity_checks=True,
            primary_unchanged=True,
            new_fits=0,
            prespec_sha256=a.sha(a.HERE / "REFERENCE_CONTRASTS_PRESPEC.txt"),
            script_sha256=a.sha(Path(__file__)),
        ),
    )
    print(json.dumps(a.load(dest / "verification.json"), indent=2))


if __name__ == "__main__":
    main()
