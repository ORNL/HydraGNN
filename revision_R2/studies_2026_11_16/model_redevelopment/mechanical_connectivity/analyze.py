"""Secondary raw mechanical-output comparison from completed matched replays."""
import os

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R2 = HERE.parents[2]
SOURCE = R2 / "studies_2026_11_16/hierarchy_output_probe"
PARTS = (
    R2.parent
    / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation/partitions.json"
)
ENCODERS = ["original", "no_message_degree", "no_message_no_degree"]
SIZES = np.array([125, 177, 250, 354, 500])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name, d):
    (HERE / name).write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")


def scores(y, p):
    e = p - y
    return dict(
        r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(e * e))),
        mae=float(np.mean(abs(e))),
        bias=float(e.mean()),
        pearson=float(np.corrcoef(y, p)[0, 1]),
    )


def weights(sizes):
    rng = np.random.default_rng(20261116)
    w = np.zeros((4000, len(sizes)), dtype=np.int32)
    for size in np.unique(sizes):
        ix = np.flatnonzero(sizes == size)
        draws = rng.integers(0, len(ix), size=(4000, len(ix)))
        w[:, ix] = np.array([np.bincount(d, minlength=len(ix)) for d in draws])
    return w


def main():
    assert not (HERE / "completion.json").exists(), "Preserve completed analysis."
    assert json.loads((SOURCE / "verification.json").read_text())["status"] == "pass"
    source_hashes = {
        str(PARTS): sha(PARTS),
        str(SOURCE / "true_auxiliary_cache.npz"): sha(
            SOURCE / "true_auxiliary_cache.npz"
        ),
    }
    parts = json.loads(PARTS.read_text())
    with np.load(SOURCE / "true_auxiliary_cache.npz") as z:
        truth = z["mechanical_true"].copy()
        sizes = SIZES[z["size_bucket"]]
        fields = z["mechanical_names"].astype(str)
    assert truth.shape == (940, 6) and np.isfinite(truth).all()
    preds = np.full((3, 2, 940, 6), np.nan)
    counts = np.zeros((3, 2, 940), dtype=int)
    for ei, encoder in enumerate(ENCODERS):
        for part in parts:
            repeat, fold = part["repeat"], part["fold"]
            folder = SOURCE / encoder / f"repeat{repeat}_fold{fold}"
            assert (folder / "completion.json").exists()
            path = folder / "evaluation.npz"
            source_hashes[str(path)] = sha(path)
            with np.load(path) as z:
                ids = z["dataset_idx"]
                p = z["mechanical_pred"]
                y = z["mechanical_true"]
                assert np.array_equal(ids, np.array(part["evaluation"]))
                assert np.array_equal(z["size_um"], sizes[ids])
                assert np.array_equal(y, truth[ids])
                assert (
                    z["mechanical_true_valid"].all()
                    and z["mechanical_pred_valid"].all()
                )
                assert np.isfinite(p).all() and np.isfinite(y).all()
                preds[ei, repeat, ids] = p
                counts[ei, repeat, ids] += 1
    assert (counts == 1).all() and np.isfinite(preds).all()
    rows = []
    for ei, encoder in enumerate(ENCODERS):
        for repeat in range(2):
            for ci, field in enumerate(fields):
                for size in ["pooled", *map(str, SIZES)]:
                    ix = (
                        np.arange(940)
                        if size == "pooled"
                        else np.flatnonzero(sizes == int(size))
                    )
                    rows.append(
                        dict(
                            encoder=encoder,
                            repeat=repeat,
                            field=field,
                            size_um=size,
                            n_sve=len(ix),
                            **scores(truth[ix, ci], preds[ei, repeat, ix, ci]),
                        )
                    )
    by = pd.DataFrame(rows)
    by.to_csv(HERE / "metrics_by_repeat.csv", index=False)
    cols = ["r2", "rmse", "mae", "bias", "pearson"]
    sm = (
        by.groupby(["encoder", "field", "size_um"])[cols]
        .mean()
        .add_suffix("_mean")
        .reset_index()
    )
    sm.to_csv(HERE / "metrics_summary.csv", index=False)
    prior = pd.read_csv(
        SOURCE / "auxiliary_prediction_metrics_summary.csv",
        dtype={"size_um": str},
        float_precision="round_trip",
    )
    merged = sm.merge(
        prior,
        on=["encoder", "field", "size_um"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert len(merged) == 108
    maxdiff = {
        c: float(np.max(abs(merged[c + "_mean_new"] - merged[c + "_mean_old"])))
        for c in cols
    }
    assert all(v < 1e-11 for v in maxdiff.values()), maxdiff
    losses = np.mean((preds - truth[None, None, :, :]) ** 2, axis=1)
    paired = []
    bootstrap_hashes = {}
    for size in ["pooled", *map(str, SIZES)]:
        ix = np.arange(940) if size == "pooled" else np.flatnonzero(sizes == int(size))
        w = weights(sizes[ix])
        bootstrap_hashes[size] = hashlib.sha256(w.tobytes()).hexdigest()
        cells = losses[:, ix, :].transpose(1, 0, 2).reshape(len(ix), 18)
        means = cells.mean(axis=0).reshape(3, 6)
        boot = (w @ cells / len(ix)).reshape(4000, 3, 6)
        for ei, control in enumerate(ENCODERS[1:], 1):
            delta = 100 * (1 - np.sqrt(means[0] / means[ei]))
            draws = 100 * (1 - np.sqrt(boot[:, 0] / boot[:, ei]))
            lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
            for ci, field in enumerate(fields):
                paired.append(
                    dict(
                        model="original",
                        reference=control,
                        field=field,
                        size_um=size,
                        n_sve=len(ix),
                        rmse_reduction_percent=float(delta[ci]),
                        ci_low=float(lo[ci]),
                        ci_high=float(hi[ci]),
                        model_rmse=float(np.sqrt(means[0, ci])),
                        reference_rmse=float(np.sqrt(means[ei, ci])),
                    )
                )
    pc = pd.DataFrame(paired)
    pc.to_csv(HERE / "paired_rmse.csv", index=False)
    np.savez_compressed(
        HERE / "mechanical_arrays.npz",
        prediction=preds,
        truth=truth,
        size_um=sizes,
        encoders=np.array(ENCODERS),
        fields=fields,
        dataset_idx=np.arange(940),
    )
    report = dict(
        status="complete",
        checkpoints=30,
        physical_sves=940,
        predictions=int(preds.size),
        repeats=2,
        all_source_labels_valid=True,
        all_predictions_valid=True,
        exact_identity_and_truth=True,
        prior_metric_max_abs_differences=maxdiff,
        paired_rows=len(pc),
        bootstrap_draws=4000,
        bootstrap_seed=20261116,
        bootstrap_weight_sha256=bootstrap_hashes,
        source_sha256=source_hashes,
        script_sha256=sha(Path(__file__)),
        prespec_sha256=sha(HERE / "PRESPEC.txt"),
    )
    dump("completion.json", report)
    text = "SECONDARY MECHANICAL CONNECTIVITY COMPARISON\n\n"
    text += "All thirty existing matched checkpoints; no new fitting, calibration or inference.\n"
    text += "Two complete 940-SVE repeats, repeated-maximum selection protocol; distinct from the older47-SVE mechanical assessment.\n\n"
    text += (
        sm[sm.size_um == "pooled"]
        .pivot(index="field", columns="encoder", values="r2_mean")
        .to_string(float_format=lambda x: f"{x:.6f}")
        + "\n\n"
    )
    text += (
        pc[pc.size_um == "pooled"].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        )
        + "\n\n"
    )
    text += f'Graph within-size positive R2 cells: {int((sm[(sm.encoder=="original")&(sm.size_um!="pooled")].r2_mean>0).sum())}/30.\n'
    text += f"Positive interval lower limits: {int((pc.ci_low>0).sum())}/{len(pc)} contrasts.\n"
    text += "Pointwise conditional physical-SVE bootstrap; averages repeat LOSSES, not predictions. Secondary examination after known auxiliary point metrics.\n"
    text += "Ablation removes neighbor operations and makes139264 parameters inactive; does not isolate unique physical adjacency at matched active capacity.\n"
    text += "Mechanical performance is separate from maximum accuracy; neither the47-SVE R2 nor the maximum scores are replaced by these values.\n"
    (HERE / "REPORT.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
