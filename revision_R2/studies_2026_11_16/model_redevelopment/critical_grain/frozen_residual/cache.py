"""Read-only original checkpoint replay and gated frozen feature caches."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
from pathlib import Path
import importlib.util
import json
import sys
import traceback

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from data_contract import add_raw_grain_contract

_legacy_path = PARENT.parent / "joint_branch" / "study.py"
sys.path.append(str(_legacy_path.parent))
_spec = importlib.util.spec_from_file_location(
    "frozen_residual_legacy_support", _legacy_path
)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
sha, dump = legacy.sha, legacy.dump
CASES, SIZES = legacy.CASES, legacy.SIZES


def inputs(part):
    return add_raw_grain_contract(legacy.fold_inputs(part), CASES)


def original_model(inp, device):
    path = inp["pair"] / "baseline" / "best.pt"
    completion = json.loads(path.with_name("completion.json").read_text())
    assert (
        completion["status"] == "complete"
        and sha(path) == completion["checkpoint_sha256"]
    )
    model = legacy.runner.construct(
        inp["cfg"], inp["part"]["initialization_seed"], device
    )
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    return model, dict(
        path=str(path),
        sha256=sha(path),
        full_state_sha256=legacy.joint.state_hash(model),
        original_device=completion["device"],
        selected_epoch=completion["selected_epoch_zero_based"],
    )


def validate_selection_seal(inp, selection_seal):
    if selection_seal is None:
        raise RuntimeError("Outer transforms/cache require a selected-head seal")
    seal = json.loads(Path(selection_seal).read_text())
    assert seal["partition"] == inp["part"] and seal["strict_reloaded"] is True
    assert sha(seal["selected_head_checkpoint"]) == seal["selected_head_sha256"]
    return seal


def subset(inp, name, selection_seal=None):
    if name == "evaluation":
        validate_selection_seal(inp, selection_seal)
    return legacy.subset_data(inp, name)


def immutable_shared_cache(directory, expected_provenance, build):
    """One immutable tensor payload; failures/partials are never overwritten."""
    directory = Path(directory)
    payload = directory / "cache.pt"
    manifest = directory / "cache.json"
    if directory.exists():
        if not manifest.exists():
            raise RuntimeError(f"Partial/failed shared cache preserved: {directory}")
        report = json.loads(manifest.read_text())
        assert (
            report["status"] == "complete"
            and report["expected_provenance"] == expected_provenance
        )
        assert sha(payload) == report["cache_sha256"]
        return torch.load(payload, weights_only=True, map_location="cpu"), report
    directory.mkdir(parents=True, exist_ok=False)
    try:
        values, build_report = build()
        torch.save(values, payload)
        report = dict(
            status="complete",
            expected_provenance=expected_provenance,
            cache_path=str(payload),
            cache_sha256=sha(payload),
            build_report=build_report,
        )
        dump(manifest, report)
        return values, report
    except Exception as error:
        dump(
            directory / "failure.json",
            dict(error=repr(error), traceback=traceback.format_exc()),
        )
        raise


def cache_subset(inp, model, name, device, selection_seal=None, explicit=None):
    if explicit is None:
        graphs, ids, truth, buckets = subset(inp, name, selection_seal)
    else:
        assert name.startswith("technical_")
        graphs, ids, truth, buckets = explicit
        assert set(ids.tolist()) <= set(inp["part"]["fitting"])
    before = legacy.joint.state_hash(model)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    captured = []

    def hook(_module, args, kwargs):
        captured.append((kwargs["x"] if "x" in kwargs else args[0]).detach())

    handle = model.heads_NN[0]["branch-0"].register_forward_pre_hook(
        hook, with_kwargs=True
    )
    states, node, graph, node_target, graph_target, direct, seen = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    clips = dict(
        node_peak=np.zeros(4, dtype=np.int64),
        legacy_maximum=np.zeros(4, dtype=np.int64),
    )
    try:
        with torch.no_grad():
            for batch in DataLoader(
                graphs, batch_size=16, shuffle=False, num_workers=0
            ):
                batch = batch.to(device)
                captured.clear()
                pn, pg = model(batch)
                assert len(captured) == 1 and captured[0].shape == (
                    batch.num_nodes,
                    256,
                )
                assert not captured[0].requires_grad
                states.append(captured[0].cpu())
                node.append(pn.cpu())
                graph.append(pg.cpu())
                node_target.append(batch.pn_target.cpu())
                graph_target.append(batch.pg_target.cpu())
                direct.append(
                    legacy.joint.inverse_graph(
                        pg,
                        batch.size_bucket.flatten().long(),
                        False,
                        inp["mean"],
                        inp["population_sd"],
                        inp["components"],
                    ).cpu()
                )
                seen.extend(batch.sve_id.cpu().tolist())
                counted = legacy.clip_counts(pn, pg, inp)
                for key in clips:
                    clips[key] += counted[key]
    finally:
        handle.remove()
    assert seen == ids.tolist()
    assert legacy.joint.state_hash(model) == before
    assert all(p.grad is None for p in model.parameters())
    ds = inp["ds"]
    counts = np.array([ds._raw[int(i)].num_nodes for i in ids], dtype=np.int64)
    offsets = np.r_[0, counts.cumsum()]
    nb = np.repeat(buckets, counts)
    pn, pg = torch.cat(node), torch.cat(graph)
    pn64, pg64 = pn.double().numpy(), pg.double().numpy()
    raw_node = np.column_stack(
        [ds.inverse_pergrain_target(pn64[:, 0], bucket_ids=nb)]
        + [
            ds.inverse_per_grain_aux_target(c - 1, pn64[:, c], bucket_ids=nb)
            for c in range(1, 16)
        ]
    )
    raw_graph = np.column_stack(
        [ds.inverse_pergraph_target(pg64[:, 0])]
        + [
            ds.inverse_aux_pergraph_target(c - 1, pg64[:, c], bucket_ids=buckets)
            for c in range(1, 14)
        ]
    )
    true_peak = np.concatenate(
        [
            np.column_stack(
                [getattr(ds._raw[int(i)], c + "_fip_dist")[:, 0].numpy() for c in CASES]
            )
            for i in ids
        ]
    ).astype(np.float64)
    raw_node_truth = np.concatenate(
        [
            np.column_stack(
                [
                    getattr(ds._raw[int(i)], ds.per_grain_target_field)[
                        :, ds.per_grain_target_col
                    ].numpy()
                ]
                + [
                    getattr(ds._raw[int(i)], spec.get("field", "fs_fip"))[
                        :, int(spec["col"])
                    ].numpy()
                    for spec in ds.per_grain_aux_targets
                ]
            )
            for i in ids
        ]
    ).astype(np.float64)
    raw_node_truth[:, :4] = true_peak
    masks = np.concatenate(
        [ds._raw[int(i)].per_grain_aux_nan_mask.numpy() for i in ids]
    )
    raw_node_truth[:, 1:][masks] = np.nan
    raw_graph_truth = np.column_stack(
        [truth[:, 0]]
        + [
            np.array(
                [
                    ds._pergraph_target_raw_by_source(ds._raw[int(i)], spec["source"])
                    for i in ids
                ]
            )
            for spec in ds.aux_pergraph_targets
        ]
    )
    vtk_ids = np.concatenate(
        [ds._raw[int(i)].vtk_grain_ids.numpy().reshape(-1) for i in ids]
    ).astype(np.int64)
    for i in range(len(ids)):
        np.testing.assert_array_equal(
            true_peak[offsets[i] : offsets[i + 1]].max(0), truth[i]
        )
        assert len(np.unique(vtk_ids[offsets[i] : offsets[i + 1]])) == counts[i]
    assert all(
        np.isfinite(x).all() for x in (raw_node, raw_graph, true_peak, raw_graph_truth)
    )
    legacy_direct = torch.cat(direct)
    np.testing.assert_allclose(
        raw_graph[:, :4], legacy_direct.numpy(), rtol=1e-5, atol=1e-8
    )
    arrays = dict(
        features=torch.cat(states),
        original_prediction=torch.from_numpy(raw_node[:, :4].copy()),
        true_peak=torch.from_numpy(true_peak),
        y_true=torch.from_numpy(truth),
        dataset_idx=torch.from_numpy(ids),
        size_bucket=torch.from_numpy(np.asarray(buckets, dtype=np.int64)),
        grain_count=torch.from_numpy(counts),
        offsets=torch.from_numpy(offsets),
        grain_id=torch.from_numpy(np.concatenate([np.arange(n) for n in counts])),
        vtk_grain_id=torch.from_numpy(vtk_ids),
        legacy_node_transformed=pn,
        legacy_graph_transformed=pg,
        legacy_node_target_transformed=torch.cat(node_target),
        legacy_graph_target_transformed=torch.cat(graph_target),
        legacy_node_mean_raw=torch.from_numpy(raw_node),
        legacy_graph_mean_raw=torch.from_numpy(raw_graph),
        node_target_raw=torch.from_numpy(raw_node_truth),
        graph_target_raw=torch.from_numpy(raw_graph_truth),
        node_aux_source_nan_mask=torch.from_numpy(masks),
        legacy_direct_prediction=legacy_direct,
    )
    assert (
        arrays["features"].dtype == torch.float32
        and torch.isfinite(arrays["features"]).all()
    )
    report = dict(
        subset=name,
        ids=ids.tolist(),
        n_sves=len(ids),
        n_grains=len(true_peak),
        device=str(device),
        original_state_sha256=before,
        original_state_unchanged=True,
        original_gradients_absent=True,
        canonical_grain_ids_unique=True,
        peak_maximum_identity_exact=True,
        inverse_clip_counts={k: v.tolist() for k, v in clips.items()},
        original_negative_raw_peak_counts=(raw_node[:, :4] < 0).sum(0).tolist(),
        source_aux_missing_entries=int(masks.sum()),
    )
    # Only fitting/stopping canonical predictions are opened before all20.
    if name in ("fitting", "stopping"):
        expected = pd.read_csv(
            inp["pair"] / "baseline" / f"{name}_predictions.csv",
            float_precision="round_trip",
        )
        expected = (
            expected.pivot(index="dataset_idx", columns="case", values="prediction")
            .loc[ids, CASES]
            .to_numpy()
        )
        error = float(
            np.max(
                np.abs(legacy_direct.numpy() - expected) / inp["population_sd"][buckets]
            )
        )
        assert error < 1e-4
        report.update(
            canonical_original_direct_max_error_fitting_sd=error,
            canonical_direct_tolerance=1e-4,
        )
        if name == "stopping":
            reference = (
                legacy.WORK
                / "revision_R2/studies_2026_11_15/data"
                / (
                    f"grain_predictions_stopping_repeat{inp['part']['repeat']}_fold{inp['part']['fold']}.npz"
                )
            )
            with np.load(reference) as z:
                np.testing.assert_array_equal(z["dataset_idx"], ids)
                np.testing.assert_array_equal(z["true_peak"], true_peak)
                error = float(
                    np.max(
                        np.abs(raw_node[:, :4] - z["pred_peak"])
                        / inp["grain_variance"][nb] ** 0.5
                    )
                )
                assert error < 1e-4
                report.update(
                    canonical_raw_grain_max_error_fitting_sd=error,
                    canonical_grain_tolerance=1e-4,
                )
    return arrays, report


def batches(cache, order=None, batch_size=16, device="cpu", include_targets=True):
    order = list(range(len(cache["dataset_idx"]))) if order is None else list(order)
    offsets = cache["offsets"]
    for start in range(0, len(order), batch_size):
        selected = order[start : start + batch_size]
        pieces = [slice(int(offsets[i]), int(offsets[i + 1])) for i in selected]
        counts = [int(cache["grain_count"][i]) for i in selected]
        batch = dict(
            features=torch.cat([cache["features"][sl] for sl in pieces]),
            original_prediction=torch.cat(
                [cache["original_prediction"][sl] for sl in pieces]
            ),
            membership=torch.repeat_interleave(
                torch.arange(len(selected)), torch.tensor(counts)
            ),
            buckets=cache["size_bucket"][selected],
            dataset_idx=cache["dataset_idx"][selected],
        )
        if include_targets:
            batch.update(
                target=torch.cat([cache["true_peak"][sl] for sl in pieces]),
                maximum_truth=cache["y_true"][selected],
            )
        yield {key: value.to(device) for key, value in batch.items()}
