"""Read-only replay of frozen outer-fold SAGE checkpoints and raw grain export.

No fitting, scaler refitting, model selection, or manuscript/source mutation.
"""
from fip_release_paths import release_root
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

WORK = Path(str(release_root() / ""))
OUT = WORK / "revision_R2/studies_2026_11_15"
FROZEN = WORK / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
ORIGINAL = Path(
    str(
        release_root()
        / "examples/fip_revision_restart_20261005/final_repeated_evaluation/gnn/runner.py"
    )
)
spec = importlib.util.spec_from_file_location("frozen_runner", ORIGINAL)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
joint = runner.joint
CASES, SIZES = (joint.CASES, joint.SIZES)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def components_for(ds):
    result = []
    for c in range(4):
        scaler = ds._scaler_pergraph_y if c == 0 else ds._scaler_aux_pergraph_y[c - 1]
        state = (
            ds._pergraph_target_state
            if c == 0
            else ds._aux_pergraph_target_states[c - 1]
        )
        pt = state["power_transformer"]
        result.append(
            dict(
                outer_mean=float(scaler.mean_[0]),
                outer_sd=float(scaler.scale_[0]),
                inner_mean=float(pt._scaler.mean_[0]),
                inner_sd=float(pt._scaler.scale_[0]),
                lam=float(pt.lambdas_[0]),
                clip_low=float(state["z_min_safe"]),
                clip_high=float(state["z_max_safe"]),
                fallback=float(state["y_mean"]),
            )
        )
    return result


def export_subset(model, ds, part, pair, subset, stat, truth_contract):
    ids = np.asarray(
        part["evaluation" if subset == "evaluation" else "stopping"], dtype=np.int64
    )
    graphs = [ds._transform(ds._raw[int(index)]) for index in ids]
    truth = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(
                    ds._raw[int(index)], c + "_pergraph_max"
                )
                for c in CASES
            ]
            for index in ids
        ],
        dtype=np.float64,
    )
    assert np.array_equal(truth, truth_contract[ids])
    buckets = np.array([int(g.size_bucket[0]) for g in graphs], dtype=np.int16)
    values = joint.build_graphs(
        graphs, truth, buckets, stat["mean_by_size"], stat["sd_by_size"], False
    )
    components = components_for(ds)
    direct, peak_pred, peak_true = ([], [], [])
    observed_ids = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(values, batch_size=16, shuffle=False, num_workers=0):
            batch = batch.to(device)
            pn, pg = model(batch)
            direct.append(
                joint.inverse_graph(
                    pg,
                    batch.size_bucket.flatten().long(),
                    False,
                    stat["mean_by_size"],
                    stat["sd_by_size"],
                    components,
                )
                .cpu()
                .numpy()
            )
            positions = batch.study_position.cpu().tolist()
            observed_ids.extend(ids[positions].tolist())
            transformed = pn[:, :4].double().cpu().numpy()
            pred_columns = []
            for c in range(4):
                inv = (
                    ds.inverse_pergrain_target
                    if c == 0
                    else lambda value, c=c: ds.inverse_per_grain_aux_target(
                        c - 1, value
                    )
                )
                pred_columns.append(inv(transformed[:, c]))
            peak_pred.append(np.column_stack(pred_columns))
            peak_true.append(
                np.concatenate(
                    [
                        np.column_stack(
                            [
                                getattr(ds._raw[int(ids[p])], c + "_fip_dist")[
                                    :, 0
                                ].numpy()
                                for c in CASES
                            ]
                        )
                        for p in positions
                    ]
                ).astype(np.float64)
            )
    assert np.array_equal(observed_ids, ids)
    direct, peak_pred, peak_true = map(np.concatenate, [direct, peak_pred, peak_true])
    expected = pd.read_csv(
        pair / "baseline" / f"{subset}_predictions.csv", float_precision="round_trip"
    )
    expected = (
        expected.pivot(index="dataset_idx", columns="case", values="prediction")
        .loc[ids, CASES]
        .to_numpy()
    )
    error = direct - expected
    scale = stat["sd_by_size"][buckets]
    max_scaled_error = float(np.max(np.abs(error) / scale))
    assert max_scaled_error < 0.0001, (
        part["repeat"],
        part["fold"],
        subset,
        max_scaled_error,
    )
    counts = np.array([ds._raw[int(i)].num_nodes for i in ids], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    assert offsets[-1] == len(peak_pred) == len(peak_true)
    assert np.isfinite(peak_pred).all() and np.isfinite(peak_true).all()
    for j in range(len(ids)):
        assert np.array_equal(peak_true[offsets[j] : offsets[j + 1]].max(0), truth[j])
    grain_dataset_idx = np.repeat(ids, counts)
    grain_id = np.concatenate([np.arange(n, dtype=np.int32) for n in counts])
    size_um = np.asarray(SIZES, dtype=np.int16)[buckets]
    name = f"grain_predictions_{('' if subset == 'evaluation' else subset + '_')}repeat{part['repeat']}_fold{part['fold']}"
    dest = OUT / "data" / (name + ".npz")
    np.savez_compressed(
        dest,
        cases=np.array(CASES),
        repeat=np.array(part["repeat"]),
        fold=np.array(part["fold"]),
        subset=np.array(subset),
        grain_dataset_idx=grain_dataset_idx,
        grain_id=grain_id,
        grain_size_um=np.repeat(size_um, counts),
        pred_peak=peak_pred,
        true_peak=peak_true,
        dataset_idx=ids,
        size_um=size_um,
        size_bucket=buckets,
        grain_count=counts,
        y_true=truth,
        direct_prediction=direct,
        offsets=offsets,
    )
    pd.DataFrame(
        dict(
            dataset_idx=ids,
            size_um=size_um,
            grain_count=counts,
            repeat=part["repeat"],
            fold=part["fold"],
            subset=subset,
        )
    ).to_csv(dest.with_suffix(".csv"), index=False)
    report = dict(
        status="pass",
        subset=subset,
        repeat=part["repeat"],
        fold=part["fold"],
        n_sves=len(ids),
        n_grains=len(peak_pred),
        max_absolute_direct_replay_error_by_case=np.abs(error).max(0).tolist(),
        max_direct_replay_error_in_fitting_sd=max_scaled_error,
        direct_replay_tolerance_in_fitting_sd=0.0001,
        grain_truth_max_equals_sve_truth=True,
        cases=CASES,
        archive_sha256=joint.sha(dest),
        grain_id_definition="zero-based row in the original per-SVE grain tensor; stable across repeats",
        prediction_inputs="saved fitting-only transformations and frozen checkpoint; no refitting",
        stopping_note="Stopping predictions share the set used for checkpoint selection; any subsequent calibration is internal to the outer split but not independent of stopping selection.",
    )
    save_json(dest.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="optional repeat,fold")
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(33500 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "inference").mkdir(parents=True, exist_ok=True)
    parts = json.loads((FROZEN / "partitions.json").read_text())
    assert joint.sha(FROZEN / "partitions.json") == runner.EXPECTED_PARTITIONS_SHA
    raw = None
    truth_file = np.load(FROZEN / "regressors/raw_truth940.npz")
    assert np.array_equal(truth_file["cases"], CASES)
    reports = []
    start = time.perf_counter()
    for part in parts:
        key = f"{part['repeat']},{part['fold']}"
        if args.only and args.only != key:
            continue
        pair = FROZEN / "gnn" / f"repeat{part['repeat']}_fold{part['fold']}"
        preparation = json.loads((pair / "preparation.json").read_text())
        completion = json.loads((pair / "baseline/completion.json").read_text())
        cfg = preparation["original_source_config"]
        with (pair / "fitted_preprocessing.pkl").open("rb") as stream:
            ds = pickle.load(stream)
        assert (
            ds.splits.train == part["fitting"]
            and ds.splits.val == part["stopping"]
            and (ds.splits.test == part["evaluation"])
        )
        if raw is None:
            ds._raw = None
            ds._load()
            raw = ds._raw
            assert len(raw) == 940
            metadata = pd.DataFrame(
                dict(
                    dataset_idx=np.arange(940),
                    size_um=[SIZES[int(g.size_bucket[0])] for g in raw],
                    grain_count=[g.num_nodes for g in raw],
                )
            )
            metadata.to_csv(OUT / "data/all_sve_metadata.csv", index=False)
            print(
                "RAW_LOADED",
                len(raw),
                "graphs",
                int(metadata.grain_count.sum()),
                "grains",
                flush=True,
            )
        else:
            ds._raw = raw
        checkpoint = pair / "baseline/best.pt"
        assert joint.sha(checkpoint) == completion["checkpoint_sha256"]
        device = torch.device(completion["device"])
        model = runner.construct(cfg, part["initialization_seed"], device)
        state = torch.load(checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        stat = np.load(pair / "fitting_statistics.npz")
        fold_reports = []
        for subset in ["evaluation", "stopping"]:
            report = export_subset(
                model, ds, part, pair, subset, stat, truth_file["raw_truth"]
            )
            report.update(
                device=str(device),
                checkpoint_sha256=completion["checkpoint_sha256"],
                preprocessing_sha256=joint.sha(pair / "fitted_preprocessing.pkl"),
            )
            fold_reports.append(report)
            print(json.dumps(report), flush=True)
        reports.extend(fold_reports)
        save_json(
            OUT / "inference" / "progress.json",
            dict(
                status="running",
                elapsed_seconds=time.perf_counter() - start,
                completed=reports,
            ),
        )
        del model, state, ds
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = dict(
        status="complete",
        cases=CASES,
        partitions_sha256=runner.EXPECTED_PARTITIONS_SHA,
        original_runner_sha256=joint.sha(ORIGINAL),
        original_joint_helper_sha256=joint.sha(joint.HERE / "run.py"),
        exporter_sha256=joint.sha(__file__),
        total_seconds=time.perf_counter() - start,
        checkpoints=len(reports) // 2,
        datasets=reports,
        no_model_or_preprocessing_refit=True,
        original_device_and_batch_size_preserved=True,
        conditional_scope="Expanded replay of a recipe developed using this dataset; not independent new-data validation or nested selection evaluation.",
    )
    save_json(
        OUT
        / "inference"
        / (
            "replay_manifest"
            + ("_" + args.only.replace(",", "_") if args.only else "")
            + ".json"
        ),
        manifest,
    )
    torch.distributed.destroy_process_group()
    print(
        "REPLAY_COMPLETE",
        json.dumps(
            dict(checkpoints=len(reports) // 2, total_seconds=manifest["total_seconds"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
