"""Prespecified, paired count diagnostic of saved outer grain predictions."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
DATA = WORKSPACE / "revision_R2/studies_2026_11_15/data"
FROZEN = WORKSPACE / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
CASES = ("lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp")
DRAWS = 100
BOOTSTRAPS = 2000
SEED = 20261116
FEATURES = (
    "mean_y",
    "mean_p",
    "mean_y2",
    "mean_p2",
    "mean_yp",
    "mean_error2",
    "mean_abs_error",
    "mean_y_squared",
    "mean_p_squared",
    "mean_y_times_mean_p",
    "mean_error_squared",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def index_hash(index: np.ndarray) -> bytes:
    return hashlib.sha256(np.asarray(index, dtype="<i4").tobytes()).hexdigest().encode()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_subset(y, p, index):
    """y:Nx4,p:2xNx4; indices sorted by canonical grain ID."""
    ys = y[index]
    ps = p[:, index]
    target = ys.max(axis=0)
    truth_winners = ys == target
    ties = truth_winners.sum(axis=0)
    oracle = (ps * truth_winners[None]).sum(axis=1) / ties[None]
    raw = ps.max(axis=1)
    winner_local = np.argmax(ps, axis=1)
    actual = ys[winner_local, np.arange(4)[None, :]]
    hit1 = actual == target[None]
    rank = np.argsort(-ps, axis=1, kind="stable")[:, : min(5, len(index)), :]
    rank_true = ys[rank, np.arange(4)[None, None, :]]
    hit5 = np.any(rank_true == target[None, None], axis=1)
    chance5 = np.ones(4)
    for j in range(min(5, len(index))):
        chance5 *= np.maximum(len(index) - ties - j, 0) / (len(index) - j)
    capture = np.divide(
        actual, target[None], out=np.ones_like(actual), where=target[None] != 0
    )
    return dict(
        target=target,
        raw_max=raw,
        oracle_location=oracle,
        true_at_predicted_winner=actual,
        predicted_winner_id=index[winner_local],
        hit1=hit1,
        hit5=hit5,
        capture=capture,
        true_max_ties=ties,
        chance1=ties / len(index),
        chance5=1 - chance5,
    )


def self_check():
    y = np.array([[1, 2, 0, 3], [3, 2, 0, 1], [3, 1, 0, 2]], float)
    p = np.array(
        [
            [[9, 2, 5, 1], [7, 2, 3, 4], [6, 1, 1, 0]],
            [[0, 2, 2, 1], [1, 2, 1, 2], [2, 1, 0, 3]],
        ],
        float,
    )
    q = evaluate_subset(y, p, np.arange(3))
    np.testing.assert_array_equal(q["target"], [3, 2, 0, 3])
    np.testing.assert_array_equal(q["predicted_winner_id"][:, 0], [0, 2])
    np.testing.assert_allclose(q["oracle_location"][:, 0], [6.5, 1.5])
    np.testing.assert_array_equal(q["hit1"][:, 0], [False, True])
    np.testing.assert_array_equal(q["hit5"], np.ones((2, 4), bool))
    np.testing.assert_array_equal(q["capture"][:, 2], [1, 1])
    np.testing.assert_allclose(q["chance5"], 1)
    np.testing.assert_array_equal(q["predicted_winner_id"][:, 1], [0, 0])


def load_fields():
    partition_path = FROZEN / "partitions.json"
    partitions = {
        (p["repeat"], p["fold"]): p for p in json.loads(partition_path.read_text())
    }
    truth_path = FROZEN / "regressors/raw_truth940.npz"
    with np.load(truth_path, allow_pickle=False) as z:
        assert tuple(z["cases"]) == CASES
        truth = {
            int(i): (int(s), v.copy())
            for i, s, v in zip(z["dataset_idx"], z["size_um"], z["raw_truth"])
        }
        raw_truth_array_hash = hashlib.sha256(z["raw_truth"].tobytes()).hexdigest()
    assert (
        raw_truth_array_hash
        == "c980bbf556684f657e82c80e809df6ed7d4ed6a65695f4053d274761e9836cfb"
    )
    metadata_path = DATA / "all_sve_metadata.csv"
    with metadata_path.open(newline="") as f:
        metadata = {
            int(row["dataset_idx"]): (int(row["size_um"]), int(row["grain_count"]))
            for row in csv.DictReader(f)
        }
    assert len(metadata) == len(truth) == 940
    records = {}
    sources = []
    totals = []
    parity_max = 0.0
    for r in range(2):
        seen = set()
        total = 0
        for f in range(5):
            path = DATA / f"grain_predictions_repeat{r}_fold{f}.npz"
            sources.append(dict(path=str(path), sha256=sha(path)))
            with np.load(path, allow_pickle=False) as archive:
                # NPZ indexing decompresses on every access; materialize once per file.
                z = {key: archive[key] for key in archive.files}
                assert int(z["repeat"]) == r and int(z["fold"]) == f
                assert str(z["subset"]) == "evaluation"
                assert tuple(z["cases"]) == CASES
                ids = z["dataset_idx"]
                np.testing.assert_array_equal(ids, partitions[(r, f)]["evaluation"])
                assert not set(ids).intersection(partitions[(r, f)]["fitting"])
                assert not set(ids).intersection(partitions[(r, f)]["stopping"])
                assert len(z["offsets"]) == len(ids) + 1
                assert z["offsets"][0] == 0 and z["offsets"][-1] == len(z["true_peak"])
                for j, idx in enumerate(ids):
                    idx = int(idx)
                    assert idx not in seen
                    seen.add(idx)
                    a, b = map(int, z["offsets"][j : j + 2])
                    size, n = metadata[idx]
                    assert b - a == n == int(z["grain_count"][j])
                    assert int(z["size_um"][j]) == size == truth[idx][0]
                    np.testing.assert_array_equal(
                        z["grain_dataset_idx"][a:b], np.full(n, idx)
                    )
                    np.testing.assert_array_equal(z["grain_id"][a:b], np.arange(n))
                    np.testing.assert_array_equal(
                        z["grain_size_um"][a:b], np.full(n, size)
                    )
                    y, p = z["true_peak"][a:b].copy(), z["pred_peak"][a:b].copy()
                    direct = z["direct_prediction"][j].copy()
                    assert (
                        np.isfinite(y).all()
                        and np.isfinite(p).all()
                        and np.isfinite(direct).all()
                    )
                    assert (y >= 0).all()
                    np.testing.assert_array_equal(y.max(0), z["y_true"][j])
                    np.testing.assert_array_equal(y.max(0), truth[idx][1])
                    if r == 0:
                        records[idx] = dict(
                            size=size,
                            n=n,
                            true=y,
                            predictions=[p],
                            direct=[direct],
                            folds=[f],
                        )
                    else:
                        np.testing.assert_array_equal(records[idx]["true"], y)
                        records[idx]["predictions"].append(p)
                        records[idx]["direct"].append(direct)
                        records[idx]["folds"].append(f)
                    total += n
        assert seen == set(metadata)
        totals.append(total)
    assert totals == [467390, 467390]
    for rec in records.values():
        rec["predictions"] = np.stack(rec["predictions"])
        rec["direct"] = np.stack(rec["direct"])
    verification = dict(
        archives=10,
        physical_sves=940,
        grains_by_repeat=totals,
        cases=CASES,
        source_archives=sources,
        metadata_sha256=sha(metadata_path),
        partition_sha256=sha(partition_path),
        truth_archive_sha256=sha(truth_path),
        raw_truth_array_sha256=raw_truth_array_hash,
        exact_true_field_repeat_parity=True,
        exact_true_maximum_parity=True,
        canonical_grain_ids=True,
        outer_partition_disjointness=True,
        all_inputs_finite=True,
        true_peaks_nonnegative=True,
    )
    return records, metadata, verification


def empty_configuration(name, ids, records, k, draws):
    n = len(ids)
    out = dict(
        configuration=np.asarray(name),
        dataset_idx=np.asarray(ids, np.int64),
        size_um=np.asarray([records[i]["size"] for i in ids], np.int16),
        original_n=np.asarray([records[i]["n"] for i in ids], np.int32),
        k=np.asarray([records[i]["n"] if k is None else k for i in ids], np.int32),
        folds=np.asarray([records[i]["folds"] for i in ids], np.int8),
        cases=np.asarray(CASES),
        repeats=np.arange(2),
        draw_id=np.arange(draws),
        index_sha256=np.empty((n, draws), dtype="S64"),
        target=np.empty((n, draws, 4)),
        true_max_ties=np.empty((n, draws, 4), np.int16),
        chance1=np.empty((n, draws, 4)),
        chance5=np.empty((n, draws, 4)),
    )
    for field in ("raw_max", "oracle_location", "true_at_predicted_winner", "capture"):
        out[field] = np.empty((n, 2, draws, 4))
    for field in ("hit1", "hit5"):
        out[field] = np.empty((n, 2, draws, 4), bool)
    out["predicted_winner_id"] = np.empty((n, 2, draws, 4), np.int32)
    if k is None:
        out["original_direct_head"] = np.stack([records[i]["direct"] for i in ids])[
            :, :, None
        ]
    return out


def sufficient(y, p):
    """Per-SVE moments; p:n,repeat,draw,case; y:n,draw,case."""
    yt = np.broadcast_to(y[:, None], p.shape)
    ym = yt.mean((1, 2))
    pm = p.mean((1, 2))
    return np.stack(
        [
            ym,
            pm,
            (yt * yt).mean((1, 2)),
            (p * p).mean((1, 2)),
            (yt * p).mean((1, 2)),
            ((p - yt) ** 2).mean((1, 2)),
            np.abs(p - yt).mean((1, 2)),
            ym ** 2,
            pm ** 2,
            ym * pm,
            (pm - ym) ** 2,
        ],
        axis=-1,
    )


def metrics(m):
    y, p, y2, p2, yp, e2, ae, ym2, pm2, ympm, em2 = np.moveaxis(m, -1, 0)
    vy = np.maximum(y2 - y * y, 0)
    vp = np.maximum(p2 - p * p, 0)
    vym = np.maximum(ym2 - y * y, 0)
    vpm = np.maximum(pm2 - p * p, 0)
    assert np.all(vy > 0) and np.all(vp > 0) and np.all(vym > 0) and np.all(vpm > 0)
    return dict(
        bias=p - y,
        mse=e2,
        rmse=np.sqrt(e2),
        mae=ae,
        draw_repeat_target_sd=np.sqrt(vy),
        draw_repeat_prediction_sd=np.sqrt(vp),
        draw_repeat_correlation=(yp - y * p) / np.sqrt(vy * vp),
        draw_repeat_r2=1 - e2 / vy,
        between_sve_mean_target_sd=np.sqrt(vym),
        between_sve_mean_prediction_sd=np.sqrt(vpm),
        between_sve_mean_correlation=(ympm - y * p) / np.sqrt(vym * vpm),
        between_sve_mean_rmse=np.sqrt(em2),
        between_sve_mean_r2=1 - em2 / vym,
    )


def main():
    started = time.monotonic()
    assert (HERE / "PRESPEC.txt").exists()
    self_check()
    records, metadata, verify = load_fields()
    ids = sorted(records)
    large = [i for i in ids if records[i]["size"] == 500]
    assert len(large) == 141 and min(records[i]["n"] for i in ids) == 26
    confs = {
        "full": empty_configuration("full", ids, records, None, 1),
        "common_26": empty_configuration("common_26", ids, records, 26, DRAWS),
    }
    for k in (30, 83, 233, 665):
        confs[f"large_{k}"] = empty_configuration(
            f"large_{k}", large, records, k, DRAWS
        )
    positions = {
        name: {int(idx): j for j, idx in enumerate(c["dataset_idx"])}
        for name, c in confs.items()
    }
    nested_checks = 0
    for j, idx in enumerate(ids):
        rec = records[idx]
        routes = [("full", None), ("common_26", 26)]
        if rec["size"] == 500:
            routes += [(f"large_{k}", k) for k in (30, 83, 233, 665)]
        for draw in range(DRAWS):
            permutation = np.random.default_rng(
                np.random.SeedSequence([SEED, idx, draw])
            ).permutation(rec["n"])
            previous = None
            for name, k in routes:
                if k is None and draw:
                    continue
                index = (
                    np.arange(rec["n"], dtype=np.int32)
                    if k is None
                    else np.sort(permutation[:k]).astype(np.int32)
                )
                if k is not None:
                    if previous is not None:
                        assert np.isin(previous, index).all()
                        nested_checks += 1
                    previous = index
                q = evaluate_subset(rec["true"], rec["predictions"], index)
                out = confs[name]
                row = positions[name][idx]
                out["index_sha256"][row, draw] = index_hash(index)
                for field, value in q.items():
                    if field in ("target", "true_max_ties", "chance1", "chance5"):
                        out[field][row, draw] = value
                    else:
                        out[field][row, :, draw] = value
        if (j + 1) % 100 == 0:
            print(f"paired subsets: {j+1}/940 SVEs", flush=True)
    stats = {}
    count_checks = {}
    for name, out in confs.items():
        for field in (
            "target",
            "raw_max",
            "oracle_location",
            "true_at_predicted_winner",
            "capture",
            "chance1",
            "chance5",
        ):
            assert np.isfinite(out[field]).all()
        assert np.all(out["true_at_predicted_winner"] <= out["target"][:, None])
        assert np.all((out["capture"] >= 0) & (out["capture"] <= 1))
        count_checks[name] = dict(
            sves=len(out["dataset_idx"]),
            draws=len(out["draw_id"]),
            prediction_records=int(out["raw_max"].size),
            zero_target_draw_case_count=int((out["target"] == 0).sum()),
            tied_true_maximum_draw_case_count=int((out["true_max_ties"] > 1).sum()),
        )
        np.savez_compressed(HERE / f"draws_{name}.npz", **out)
        routes = (
            ("raw_max", "oracle_location", "original_direct_head")
            if name == "full"
            else ("raw_max", "oracle_location")
        )
        ss = {route: sufficient(out["target"], out[route]) for route in routes}
        ss.update(
            dataset_idx=out["dataset_idx"],
            size_um=out["size_um"],
            features=np.asarray(FEATURES),
        )
        np.savez_compressed(HERE / f"sufficient_{name}.npz", **ss)
        stats[name] = ss
    metric_rows = []
    paired_rows = []
    localization_rows = []
    repeat_rows = []
    bootstrap_payload = {}
    for size in (125, 177, 250, 354, 500):
        size_ids = np.asarray([i for i in ids if records[i]["size"] == size])
        n = len(size_ids)
        rng = np.random.default_rng(np.random.SeedSequence([20261117, size]))
        resample = rng.integers(0, n, size=(BOOTSTRAPS, n))
        weights = np.zeros((BOOTSTRAPS, n))
        np.add.at(weights, (np.arange(BOOTSTRAPS)[:, None], resample), 1 / n)
        bootstrap_payload[f"size_{size}_ids"] = size_ids
        bootstrap_payload[f"size_{size}_indices"] = resample.astype(np.int16)
        for name, out in confs.items():
            take = np.flatnonzero(out["size_um"] == size)
            if not len(take):
                continue
            np.testing.assert_array_equal(out["dataset_idx"][take], size_ids)
            routes = (
                ("raw_max", "oracle_location", "original_direct_head")
                if name == "full"
                else ("raw_max", "oracle_location")
            )
            point_by_route = {}
            boot_by_route = {}
            for route in routes:
                sufficient_stats = stats[name][route][take]
                point = metrics(sufficient_stats.mean(0))
                boot = metrics(
                    np.einsum("bs,scf->bcf", weights, sufficient_stats, optimize=True)
                )
                point_by_route[route] = point
                boot_by_route[route] = boot
                for c, case in enumerate(CASES):
                    row = dict(
                        configuration=name,
                        physical_size_um=size,
                        case=case,
                        route=route,
                        n_physical_sves=n,
                        draws_per_sve=len(out["draw_id"]),
                        repeats=2,
                        k="observed_N" if name == "full" else int(out["k"][take[0]]),
                    )
                    for key, val in point.items():
                        row[key] = float(val[c])
                        row[key + "_lo"], row[key + "_hi"] = np.quantile(
                            boot[key][:, c], [0.025, 0.975]
                        ).tolist()
                    metric_rows.append(row)
                for repeat in range(2):
                    pm = metrics(
                        sufficient(
                            out["target"][take], out[route][take, repeat : repeat + 1]
                        ).mean(0)
                    )
                    for c, case in enumerate(CASES):
                        repeat_rows.append(
                            dict(
                                configuration=name,
                                physical_size_um=size,
                                case=case,
                                route=route,
                                repeat=repeat,
                                **{k: float(v[c]) for k, v in pm.items()},
                            )
                        )
            for c, case in enumerate(CASES):
                row = dict(
                    configuration=name,
                    physical_size_um=size,
                    case=case,
                    contrast="oracle_location_minus_raw_max",
                )
                for key in (
                    "rmse",
                    "draw_repeat_correlation",
                    "between_sve_mean_correlation",
                ):
                    delta = (
                        point_by_route["oracle_location"][key][c]
                        - point_by_route["raw_max"][key][c]
                    )
                    bd = (
                        boot_by_route["oracle_location"][key][:, c]
                        - boot_by_route["raw_max"][key][:, c]
                    )
                    row[key + "_difference"] = float(delta)
                    (
                        row[key + "_difference_lo"],
                        row[key + "_difference_hi"],
                    ) = np.quantile(bd, [0.025, 0.975]).tolist()
                paired_rows.append(row)
            loc = {
                "hit1": out["hit1"][take].mean((1, 2)),
                "hit5": out["hit5"][take].mean((1, 2)),
                "capture": out["capture"][take].mean((1, 2)),
                "shortfall": (
                    out["target"][take, None] - out["true_at_predicted_winner"][take]
                ).mean((1, 2)),
                "chance1": out["chance1"][take].mean(1),
                "chance5": out["chance5"][take].mean(1),
            }
            for c, case in enumerate(CASES):
                row = dict(
                    configuration=name,
                    physical_size_um=size,
                    case=case,
                    n_physical_sves=n,
                )
                for key, arr in loc.items():
                    row[key] = float(arr[:, c].mean())
                    row[key + "_lo"], row[key + "_hi"] = np.quantile(
                        weights @ arr[:, c], [0.025, 0.975]
                    ).tolist()
                row["hit5_above_chance"] = row["hit5"] - row["chance5"]
                localization_rows.append(row)
    np.savez_compressed(
        HERE / "physical_sve_bootstrap_indices.npz", **bootstrap_payload
    )
    write_csv(HERE / "metrics.csv", metric_rows)
    write_csv(HERE / "repeat_metrics.csv", repeat_rows)
    write_csv(HERE / "paired_oracle_comparisons.csv", paired_rows)
    write_csv(HERE / "localization.csv", localization_rows)
    verify.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        numpy_version=np.__version__,
        prespec_sha256=sha(HERE / "PRESPEC.txt"),
        script_sha256=sha(Path(__file__)),
        seed=SEED,
        bootstrap_seed=20261117,
        bootstrap_replicates=BOOTSTRAPS,
        paired_nested_subset_checks=nested_checks,
        subset_configurations=count_checks,
        synthetic_tie_and_oracle_tests_passed=True,
        all_diagnostic_predictions_finite=True,
        no_retraining=True,
        no_calibration=True,
        no_new_model_inference=True,
        metric_rows=len(metric_rows),
        metric_intervals_finite=True,
        uncertainty_unit="physical SVE; within-SVE draw/repeat moments averaged first",
    )
    for row in metric_rows:
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, float))
    verify["outputs"] = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in HERE.iterdir()
        if p.suffix in (".npz", ".csv")
    }
    (HERE / "verification.json").write_text(json.dumps(verify, indent=2))
    print(
        json.dumps(
            {
                k: verify[k]
                for k in ("status", "physical_sves", "metric_rows", "elapsed_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
