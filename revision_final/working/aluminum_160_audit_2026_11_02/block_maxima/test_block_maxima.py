"""Exploratory per-SVE block-maximum map; does not edit publication files.

Prespecified variants: B=16,32,64; seed101; 1 and 5 random partitions;
remove k=0,1,3 largest peaks before fitting, keep the full maximum target.
Equal block size m=floor((N-k)/B); discard the random remainder. Fit a
Gumbel by moments to B block maxima. Estimate the full-N maximum mean as
mean(block_maxima) + sd(block_maxima)*sqrt(6)/pi * log(N/m).

The N/m count correction is exact for a max-stable Gumbel block parent
under iid grain sampling, including real N/m through its CDF power.
The Gumbel block parent and iid grains are approximations. For k>0,
deletion also truncates the empirical parent: this is an influence test,
not proof that the extrapolation restores the removed tail.
"""
from __future__ import annotations
from fip_release_paths import release_root
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = Path(str(release_root() / "examples/fip_unified/sim_evt_validation"))
CACHE = SOURCE / "results/voxelmax/per_sve_cache.pkl"
SAVED = SOURCE / "results/mixture/per_sve_fitted_q.csv"
BLOCKS = (16, 32, 64)
REMOVE = (0, 1, 3)
REPEATS = 5
SEED = 101


def metrics(frame):
    truth = frame.truth.to_numpy(float)
    pred = frame.prediction.to_numpy(float)
    good = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = (truth[good], pred[good])
    err = pred - truth
    return dict(
        n_total=len(frame),
        n_valid=int(good.sum()),
        R2=float(1 - np.sum(err ** 2) / np.sum((truth - truth.mean()) ** 2)),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        MAE=float(np.mean(abs(err))),
        bias=float(err.mean()),
        truth_mean=float(truth.mean()),
        prediction_mean=float(pred.mean()),
        truth_sd=float(truth.std()),
        prediction_sd=float(pred.std()),
        correlation=float(np.corrcoef(truth, pred)[0, 1]),
    )


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    with CACHE.open("rb") as f:
        source = pickle.load(f)
    saved = pd.read_csv(SAVED)
    assert len(source) == len(saved) == 648
    rows, partition_rows = ([], [])
    total_invalid_values = 0
    for ordinal, (_, rec) in enumerate(source.iterrows()):
        bucket, sve_idx = (int(rec.bucket), int(rec.sve_idx))
        raw = np.asarray(rec.mg, dtype=float)
        good = np.isfinite(raw) & (raw > 0)
        total_invalid_values += int((~good).sum())
        x = raw[good]
        N, truth = (x.size, float(x.max()))
        assert int(saved.iloc[ordinal]["bucket"]) == bucket
        assert int(saved.iloc[ordinal].n_grains) == N
        assert abs(float(saved.iloc[ordinal].M_S_true) - truth) < 1e-14
        rows.append(
            dict(
                bucket=bucket,
                sve_idx=sve_idx,
                N=N,
                truth=truth,
                method="original_fitted_q",
                B=0,
                remove_k=0,
                n_partitions=0,
                prediction=float(saved.iloc[ordinal].M_S_AtM_fit),
                block_size=np.nan,
                remainder=np.nan,
                effective_blocks=np.nan,
                original_max_inclusion_rate=1.0,
            )
        )
        for k in REMOVE:
            retained = np.sort(x)[: N - k]
            predictions = {B: [] for B in BLOCKS}
            inclusions = {B: [] for B in BLOCKS}
            for rep in range(REPEATS):
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, bucket, sve_idx, rep])
                )
                shuffled = retained[rng.permutation(retained.size)]
                for B in BLOCKS:
                    m = retained.size // B
                    remainder = retained.size - B * m
                    assert m >= 2
                    block_values = shuffled[: B * m].reshape(B, m)
                    block_max = block_values.max(axis=1)
                    beta = float(block_max.std(ddof=1) * np.sqrt(6) / np.pi)
                    pred = float(block_max.mean() + beta * np.log(N / m))
                    included = bool(np.any(block_values == truth))
                    predictions[B].append(pred)
                    inclusions[B].append(float(included))
                    partition_rows.append(
                        dict(
                            bucket=bucket,
                            sve_idx=sve_idx,
                            B=B,
                            remove_k=k,
                            repeat=rep,
                            seed=SEED,
                            N=N,
                            n_retained=retained.size,
                            block_size=m,
                            remainder=remainder,
                            effective_blocks=N / m,
                            block_mean=float(block_max.mean()),
                            block_sd=float(block_max.std(ddof=1)),
                            beta=beta,
                            prediction=pred,
                            truth=truth,
                            original_max_included=included,
                        )
                    )
            for B in BLOCKS:
                for nrep in (1, REPEATS):
                    m = retained.size // B
                    rows.append(
                        dict(
                            bucket=bucket,
                            sve_idx=sve_idx,
                            N=N,
                            truth=truth,
                            method=f"block_mom_B{B}_k{k}_r{nrep}",
                            B=B,
                            remove_k=k,
                            n_partitions=nrep,
                            prediction=float(np.mean(predictions[B][:nrep])),
                            block_size=m,
                            remainder=retained.size - B * m,
                            effective_blocks=N / m,
                            original_max_inclusion_rate=float(
                                np.mean(inclusions[B][:nrep])
                            ),
                        )
                    )
    predictions = pd.DataFrame(rows)
    partitions = pd.DataFrame(partition_rows)
    summaries = []
    for method, frame in predictions.groupby("method", sort=False):
        for bucket in [30, 45, 90, 160, "pooled"]:
            sub = frame if bucket == "pooled" else frame[frame.bucket == bucket]
            summaries.append(
                dict(
                    method=method,
                    bucket=bucket,
                    **metrics(sub),
                    median_block_size=float(sub.block_size.median())
                    if method != "original_fitted_q"
                    else None,
                    min_block_size=float(sub.block_size.min())
                    if method != "original_fitted_q"
                    else None,
                    max_remainder=float(sub.remainder.max())
                    if method != "original_fitted_q"
                    else None,
                    max_remainder_fraction=float((sub.remainder / sub.N).max())
                    if method != "original_fitted_q"
                    else None,
                    original_max_inclusion_rate=float(
                        sub.original_max_inclusion_rate.mean()
                    ),
                )
            )
    summary = pd.DataFrame(summaries)
    predictions.to_csv(HERE / "predictions.csv", index=False)
    partitions.to_csv(HERE / "partition_predictions.csv", index=False)
    summary.to_csv(HERE / "metrics.csv", index=False)
    protocol = dict(
        status="exploratory; these SVEs and results have previously been examined",
        source=str(CACHE),
        source_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        historical_result=str(SAVED),
        historical_sha256=hashlib.sha256(SAVED.read_bytes()).hexdigest(),
        n_sves=len(source),
        source_cache_kind=type(source).__name__,
        population="Finite positive per-grain maxima over voxels and twelve FS slip-system values; dimensionless FS.",
        invalid_or_nonpositive_cache_values_removed=total_invalid_values,
        data_alignment="All 648 cache rows match saved fitted-q CSV bucket, N, and exact truth to 1e-14 in preserved order.",
        B=list(BLOCKS),
        remove_k=list(REMOVE),
        partition_counts=[1, REPEATS],
        seed=SEED,
        equal_block_policy="m=floor((N-k)/B); take first B*m grains of fixed seeded permutation; discard remainder. No favorable partition selection.",
        formula="mean(block maxima)+sampleSD(block maxima)*sqrt(6)/pi*log(N/m)",
        repeat_policy="Arithmetic average of 5 separately computed maximum estimates; not one fit pooling correlated block maxima.",
        iid_scope="Random partitions are nonspatial. Treating block maxima as independent iid Gumbel observations is a working approximation.",
        top_exclusion_scope="Original full-N maximum is the unchanged target; N/m count correction does not undo tail truncation from deleting the highest k peaks.",
        information_access="For k=0, the evaluated target maximum contributes to a fitted block maximum whenever not in discarded remainder. This is not input-only prediction or independent validation.",
        training="No across-SVE fitted coefficients, no tuning B or seed to160, no change to published results.",
        source_limit="Reuses historical numeric grain cache; does not reread every raw VTK/source voxel array.",
    )
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(summary.pivot(index="method", columns="bucket", values="R2").to_string())
    print("\nFive-partition RMSE:")
    print(
        summary[
            summary.method.str.endswith("r5") | (summary.method == "original_fitted_q")
        ]
        .pivot(index="method", columns="bucket", values="RMSE")
        .to_string()
    )
    print(
        "\nSource count:",
        len(source),
        "removed cache invalid values:",
        total_invalid_values,
    )


if __name__ == "__main__":
    main()
