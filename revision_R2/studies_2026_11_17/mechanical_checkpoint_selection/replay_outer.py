"""Read-only CPU replay of the two saved checkpoints after freezing selection."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="",
)
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import importlib.util, json, pickle, time
import numpy as np
import pandas as pd
import torch
from fip_release_paths import matches_published_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
PROBE = (
    ROOT
    / "revision_R2/studies_2026_11_16/model_redevelopment/stopping_checkpoint_probe"
)
spec = importlib.util.spec_from_file_location(
    "verified_stopping_probe", PROBE / "probe.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
runner, joint, sha = old.runner, old.joint, old.sha
CASES, SIZES = old.CASES, old.SIZES


def dump(name, x):
    (HERE / name).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def extract(ds, ids, stat):
    """Generalize the audited stopping extractor to an arbitrary ordered cohort."""
    graphs = [ds._transform(ds._raw[int(i)]) for i in ids]
    buckets = np.array([int(g.size_bucket[0]) for g in graphs], dtype=np.int64)
    counts = np.array([g.num_nodes for g in graphs], dtype=np.int64)
    truth = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(ds._raw[int(i)], c + "_pergraph_max")
                for c in CASES
            ]
            for i in ids
        ],
        dtype=np.float64,
    )
    values = joint.build_graphs(
        graphs, truth, buckets, stat["mean_by_size"], stat["sd_by_size"], False
    )
    for i, g in zip(ids, values):
        g.sve_id = torch.tensor([i], dtype=torch.long)
    node_truth = np.concatenate(
        [
            np.column_stack(
                [
                    getattr(ds._raw[int(i)], ds.per_grain_target_field)[
                        :, ds.per_grain_target_col
                    ].numpy()
                ]
                + [
                    getattr(ds._raw[int(i)], s.get("field", "fs_fip"))[
                        :, int(s["col"])
                    ].numpy()
                    for s in ds.per_grain_aux_targets
                ]
            )
            for i in ids
        ]
    ).astype(np.float64)
    true_peak = np.concatenate(
        [
            np.column_stack(
                [getattr(ds._raw[int(i)], c + "_fip_dist")[:, 0].numpy() for c in CASES]
            )
            for i in ids
        ]
    ).astype(np.float64)
    node_truth[:, :4] = true_peak
    mask = np.concatenate([ds._raw[int(i)].per_grain_aux_nan_mask.numpy() for i in ids])
    assert mask.shape == (len(node_truth), 15)
    node_truth[:, 1:][mask] = np.nan
    graph_truth = np.column_stack(
        [truth[:, 0]]
        + [
            np.array(
                [
                    ds._pergraph_target_raw_by_source(ds._raw[int(i)], s["source"])
                    for i in ids
                ]
            )
            for s in ds.aux_pergraph_targets
        ]
    )
    assert graph_truth.shape == (len(ids), 14) and np.array_equal(
        graph_truth[:, :4], truth
    )
    assert np.isfinite(graph_truth).all()
    offsets = np.r_[0, np.cumsum(counts)]
    for j in range(len(ids)):
        assert np.array_equal(true_peak[offsets[j] : offsets[j + 1]].max(0), truth[j])
    return dict(
        ids=ids,
        values=values,
        truth=truth,
        buckets=buckets,
        counts=counts,
        offsets=offsets,
        node_truth=node_truth,
        graph_truth=graph_truth,
        true_peak=true_peak,
        mask=mask,
        node_names=np.array(
            ["lcf_fs_peak"] + [s["name"] for s in ds.per_grain_aux_targets]
        ),
        graph_names=np.array(
            ["lcf_fs_max"] + [s["name"] for s in ds.aux_pergraph_targets]
        ),
    )


def save(path, inp, result, part, label, epoch):
    np.savez_compressed(
        path,
        checkpoint=label,
        epoch_zero_based=epoch,
        repeat=part["repeat"],
        fold=part["fold"],
        subset="evaluation",
        cases=np.array(CASES),
        dataset_idx=inp["ids"],
        size_bucket=inp["buckets"],
        size_um=SIZES[inp["buckets"]],
        grain_count=inp["counts"],
        offsets=inp["offsets"],
        grain_dataset_idx=np.repeat(inp["ids"], inp["counts"]),
        grain_id=np.concatenate([np.arange(n, dtype=np.int32) for n in inp["counts"]]),
        node_transformed=result["pn"],
        graph_transformed=result["pg"],
        node_target_transformed=result["nt"],
        graph_target_transformed=result["gt"],
        node_mean_raw=result["raw_node"],
        node_target_raw=inp["node_truth"],
        node_mean_names=inp["node_names"],
        node_aux_source_nan_mask=inp["mask"],
        graph_mean_raw=result["raw_graph"],
        graph_target_raw=inp["graph_truth"],
        graph_mean_names=inp["graph_names"],
        pred_peak=result["raw_node"][:, :4],
        true_peak=inp["true_peak"],
        y_true=inp["truth"],
        direct_prediction=result["maximum"],
        bridge_prediction=result["operational"],
        maximum_predicted_grain=result["grain_max"],
        bridge_fallback=result["fallback"],
        bridge_selected_count=result["selected_count"],
        raw_maximum_equal20_score=result["score"],
        raw_maximum_equal20_cells=result["cells"],
    )


def parity(inp, result, part, stat):
    key = f'repeat{part["repeat"]}_fold{part["fold"]}'
    p = ROOT / "revision_R2/studies_2026_11_15/data" / f"grain_predictions_{key}.npz"
    with np.load(p) as z:
        assert str(z["subset"]) == "evaluation"
        for a, b in [
            ("dataset_idx", "ids"),
            ("size_bucket", "buckets"),
            ("grain_count", "counts"),
            ("offsets", "offsets"),
            ("y_true", "truth"),
            ("true_peak", "true_peak"),
        ]:
            assert np.array_equal(z[a], inp[b]), a
        de = np.abs(result["maximum"] - z["direct_prediction"])
        pe = np.abs(result["raw_node"][:, :4] - z["pred_peak"])
    quality = (
        pd.read_csv(BASE / "gnn" / key / "baseline/stopping_grain_quality.csv")
        .set_index("case")
        .loc[CASES]
    )
    p = (
        ROOT
        / "revision_R2/studies_2026_11_16/hierarchy_output_probe/original"
        / key
        / "evaluation.npz"
    )
    with np.load(p) as z:
        assert np.array_equal(z["dataset_idx"], inp["ids"]) and np.array_equal(
            z["mechanical_true"], inp["graph_truth"][:, 8:14]
        )
        me = np.abs(result["raw_graph"][:, 8:14] - z["mechanical_pred"])
        mscale = z["mechanical_true"].std(0, ddof=1)
    ratios = dict(
        direct=float((de / stat["sd_by_size"][inp["buckets"]]).max()),
        peak=float((pe / quality.fitting_case_sd.to_numpy()).max()),
        mechanical=float((me / mscale).max()),
    )
    assert max(ratios.values()) < 1e-4, ratios
    return dict(
        status="pass",
        scaled_max_discrepancies=ratios,
        direct_max_abs_by_case=de.max(0).tolist(),
        peak_max_abs_by_case=pe.max(0).tolist(),
        mechanical_max_abs_by_target=me.max(0).tolist(),
    )


def main():
    start = time.monotonic()
    selection = json.loads((HERE / "selection.json").read_text())
    assert selection["status"] == "frozen_before_outer_replay"
    assert not (
        HERE / "inference_completion.json"
    ).exists(), "Completed replay is immutable."
    for p, digest in selection["hashes"].items():
        assert matches_published_hash(p, digest), p
    sources = [
        Path(__file__),
        HERE / "selection.json",
        Path(old.__file__),
        Path(old.replay.__file__),
        Path(runner.__file__),
        Path(joint.__file__),
        Path(joint.dataset.__file__),
        Path(joint.losses.__file__),
        Path(joint.train.__file__),
        joint.CONTROL / "create_seed_controlled.py",
        old.BRIDGE_PATH,
    ]
    hashes = {str(p): sha(p) for p in sources}
    dump(
        "inference_manifest.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            before_outer_inference=True,
            hashes=hashes,
            selection_sha256=sha(HERE / "selection.json"),
        ),
    )
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(36600 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    (HERE / "outer").mkdir(exist_ok=True)
    parts = json.loads((BASE / "partitions.json").read_text())
    raw = None
    checks = []
    losses = []
    for part, choice in zip(parts, selection["decisions"]):
        r, f = part["repeat"], part["fold"]
        assert (r, f) == (choice["repeat"], choice["fold"])
        key = f"repeat{r}_fold{f}"
        pair = BASE / "gnn" / key
        prep = json.loads((pair / "preparation.json").read_text())
        assert prep["partition"] == part
        with (pair / "fitted_preprocessing.pkl").open("rb") as h:
            ds = pickle.load(h)
        assert (
            ds.splits.train == part["fitting"]
            and ds.splits.val == part["stopping"]
            and ds.splits.test == part["evaluation"]
        )
        if raw is None:
            raw_sha = sha(__import__("fip_release_paths").rebase_path(ds.pt_path))
            assert (
                raw_sha
                == "cf6a9c5a2b3b210a58f11429f9dc1bd2f8e923c681983d6d4a10c754cdee7615"
            )
            ds._raw = None
            ds._load()
            raw = ds._raw
        else:
            ds._raw = raw
        with np.load(pair / "fitting_statistics.npz") as z:
            stat = {k: z[k] for k in z.files}
        inp = extract(ds, np.array(part["evaluation"], dtype=np.int64), stat)
        assert not set(inp["ids"]) & set(choice["stopping_ids"])
        for label in ["best", "last"]:
            model = runner.construct(
                prep["original_source_config"],
                part["initialization_seed"],
                torch.device("cpu"),
            )
            cp = pair / "baseline" / f"{label}.pt"
            state = torch.load(cp, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=True)
            # The audited helper handles arbitrary cohort lengths. Its three
            # reported loss means divide by75, so normalize by actual cohort size.
            result = old.infer(model, ds, inp, stat)
            for name in ["joint_loss", "node_loss", "graph_loss"]:
                result[name] *= 75 / len(inp["ids"])
            check = parity(inp, result, part, stat) if label == "best" else None
            out = HERE / "outer" / f"{key}_{label}.npz"
            save(out, inp, result, part, label, int(state["epoch"]))
            losses.extend(
                [
                    dict(repeat=r, fold=f, checkpoint=label, **b)
                    for b in result["batches"]
                ]
            )
            checks.append(
                dict(
                    repeat=r,
                    fold=f,
                    checkpoint=label,
                    selected=label == choice["selected"],
                    epoch_zero_based=int(state["epoch"]),
                    checkpoint_sha256=sha(cp),
                    output_sha256=sha(out),
                    output_path=str(out),
                    n_sves=len(inp["ids"]),
                    n_grains=len(inp["node_truth"]),
                    best_reference_parity=check,
                    all_predicted_channels_finite=True,
                    source_aux_missing_entries=int(inp["mask"].sum()),
                    joint_loss=result["joint_loss"],
                    node_loss=result["node_loss"],
                    graph_loss=result["graph_loss"],
                )
            )
            dump(
                "progress.json",
                dict(
                    status="running",
                    completed_checkpoints=len(checks),
                    seconds=time.monotonic() - start,
                ),
            )
            print(
                f'outer replay complete {key} {label} epoch{state["epoch"]}', flush=True
            )
            del model, state, result
        del ds, inp
    assert all(
        matches_published_hash(p, digest) for p, digest in selection["hashes"].items()
    )
    assert all(sha(p) == digest for p, digest in hashes.items())
    pd.DataFrame(losses).to_csv(
        HERE / "outer_batch_losses.csv", index=False, float_format="%.17g"
    )
    dump(
        "inference_completion.json",
        dict(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic() - start,
            no_neural_training=True,
            cpu_only=True,
            original_inputs_unchanged=True,
            selection_unchanged=True,
            raw_source_sha256=raw_sha,
            source_hashes=hashes,
            checks=checks,
            stopping_cache_note="Full stopping predictions reused from immutable paths and hashes in selection.json",
        ),
    )
    dump(
        "progress.json",
        dict(
            status="complete",
            completed_checkpoints=20,
            seconds=time.monotonic() - start,
        ),
    )
    torch.distributed.destroy_process_group()
    print("OUTER_REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
