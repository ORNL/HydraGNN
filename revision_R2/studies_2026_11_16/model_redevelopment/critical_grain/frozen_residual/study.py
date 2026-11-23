"""Versioned preparation, head-only gated fits and complete frozen-route exports."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import torch
from cache import (
    HERE,
    PARENT,
    CASES,
    SIZES,
    legacy,
    sha,
    dump,
    inputs,
    original_model,
    cache_subset,
    batches,
    validate_selection_seal,
    immutable_shared_cache,
)
from model import FrozenResidualHead

ARMS = ("bulk_control", "critical_treatment")
MAX_EPOCHS, PATIENCE = (120, 50)


def sources():
    paths = [
        HERE / name
        for name in ("study.py", "cache.py", "model.py", "smoke.py", "PRESPEC.txt")
    ]
    paths += [
        PARENT / name for name in ("grain_head.py", "objectives.py", "data_contract.py")
    ]
    result = legacy.source_manifest()
    result.update({str(path): sha(path) for path in paths})
    return result


def version():
    return hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()[
        :16
    ]


def preparation_dir(part):
    return (
        HERE / "preparation" / version() / f"repeat{part['repeat']}_fold{part['fold']}"
    )


def head_seed(part):
    return 2026111700 + 10 * int(part["repeat"]) + int(part["fold"])


def construct(inp, device):
    before = torch.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    model = FrozenResidualHead(
        inp["grain_variance"],
        inp["variance"],
        inp["cell_weights"],
        head_seed(inp["part"]),
    ).to(device)
    assert torch.equal(before, torch.get_rng_state())
    assert all((torch.equal(a, b) for a, b in zip(gpu, torch.cuda.get_rng_state_all())))
    return model


def source_device(inp):
    completion = json.loads((inp["pair"] / "baseline/completion.json").read_text())
    return torch.device(completion["device"])


def prepare(inp):
    part = inp["part"]
    dest = preparation_dir(part)
    if dest.exists():
        report = json.loads((dest / "preparation.json").read_text())
        assert report["source_manifest"] == sources() and report["partition"] == part
        for name, digest in report["files"].items():
            assert sha(dest / name) == digest
        assert (
            sha(inp["pair"] / "baseline/best.pt")
            == report["original_checkpoint"]["sha256"]
        )
        assert (
            sha(inp["pair"] / "fitted_preprocessing.pkl")
            == report["preprocessing_sha256"]
        )
        return report
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        original, checkpoint = original_model(inp, source_device(inp))
        reports = []
        for subset in ("fitting", "stopping"):
            values, report = cache_subset(inp, original, subset, source_device(inp))
            torch.save(values, dest / f"{subset}_cache.pt")
            reports.append(report)
            del values
        assert legacy.joint.state_hash(original) == checkpoint["full_state_sha256"]
        assert sha(checkpoint["path"]) == checkpoint["sha256"]
        a, b = (construct(inp, "cpu"), construct(inp, "cpu"))
        assert legacy.joint.state_hash(a) == legacy.joint.state_hash(b)
        torch.save(a.state_dict(), dest / "initial_head.pt")
        np.savez_compressed(
            dest / "fitting_statistics.npz",
            grain_mean=inp["grain_mean"],
            grain_variance=inp["grain_variance"],
            grain_variance_floor=inp["grain_variance_floor"],
            grain_pooled_variance=inp["grain_pooled_variance"],
            maximum_variance=inp["variance"],
            maximum_mean=inp["mean"],
            maximum_variance_floor=inp["variance_floor"],
            size_weights=inp["cell_weights"],
            cell_counts=inp["cell_counts"],
            permutations=inp["permutations"],
        )
        names = (
            "fitting_cache.pt",
            "stopping_cache.pt",
            "initial_head.pt",
            "fitting_statistics.npz",
        )
        report = dict(
            status="prepared_not_trained",
            partition=part,
            source_manifest=sources(),
            manifest_version=version(),
            files={name: sha(dest / name) for name in names},
            original_checkpoint=checkpoint,
            paired_initial_head_sha256=legacy.joint.state_hash(a),
            head_seed=head_seed(part),
            head_parameters=16708,
            original_parameters_trainable=0,
            preprocessing_sha256=sha(inp["pair"] / "fitted_preprocessing.pkl"),
            original_preparation_sha256=sha(inp["pair"] / "preparation.json"),
            raw_source_sha256=inp["raw_source_sha256"],
            saved_batch_orders_sha256=legacy.array_sha(inp["permutations"]),
            grain_and_maximum_variance_floors_active=False,
            cache_reports=reports,
            outer_targets_extracted=False,
            outer_graphs_transformed=False,
            outer_fields_cached=False,
            elapsed_seconds=time.monotonic() - started,
        )
        dump(dest / "preparation.json", report)
        return report
    except Exception as error:
        dump(
            dest / "failure.json",
            dict(
                error=repr(error),
                traceback=traceback.format_exc(),
                scientific_fits_started=0,
            ),
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict(model, cache, compute_losses=False):
    model.eval()
    device = next(model.parameters()).device
    prediction, maxima, allgrain, topfour, maximum, ids = ([], [], [], [], [], [])
    with torch.no_grad():
        for batch in batches(cache, device=device, include_targets=compute_losses):
            p = model(
                batch["features"],
                batch["original_prediction"],
                batch["buckets"][batch["membership"]],
            )
            predicted_maximum = torch.stack(
                [
                    p[batch["membership"] == graph].amax(dim=0)
                    for graph in range(len(batch["buckets"]))
                ]
            )
            prediction.append(p.cpu().numpy())
            maxima.append(predicted_maximum.cpu().numpy())
            if compute_losses:
                cells = model.cells(p, batch)
                allgrain.append(cells["allgrain"].cpu().numpy())
                topfour.append(cells["topfour"].cpu().numpy())
                maximum.append(cells["maximum"].cpu().numpy())
            ids.extend(batch["dataset_idx"].cpu().tolist())
    assert ids == cache["dataset_idx"].tolist()
    result = dict(
        prediction=np.concatenate(prediction), maximum_prediction=np.concatenate(maxima)
    )
    if compute_losses:
        result.update(
            allgrain=np.concatenate(allgrain),
            topfour=np.concatenate(topfour),
            maximum=np.concatenate(maximum),
        )
    return result


def stopping(model, cache):
    result = predict(model, cache, compute_losses=True)
    buckets = cache["size_bucket"].numpy()
    cells = {
        key: np.stack([result[key][buckets == b].mean(0) for b in range(5)])
        for key in ("allgrain", "topfour", "maximum")
    }
    assert all(
        (np.isfinite(value).all() for value in cells.values())
    ), "All20 stopping cells required"
    score = float(0.5 * cells["allgrain"].mean() + 0.5 * cells["maximum"].mean())
    if not np.isfinite(score):
        raise FloatingPointError("Nonfinite stopping composite")
    return dict(
        score=score,
        cells={key: value.tolist() for key, value in cells.items()},
        negative_raw_grain_counts=(result["prediction"] < 0).sum(0).tolist(),
    )


def optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=0.0001586668862464421, weight_decay=1e-05
    )
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len({id(p) for p in owned})
    assert {id(p) for p in owned} == {id(p) for p in model.parameters()}
    assert sum((p.numel() for p in owned)) == 16708
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=15, min_lr=1e-07
    )
    return (optimizer, scheduler)


def export(model, cache, inp, arm, subset, dest):
    result = predict(model, cache)
    arrays = {key: value.numpy() for key, value in cache.items() if key != "features"}
    ids, counts = (arrays["dataset_idx"], arrays["grain_count"])
    arrays.update(
        cases=np.array(CASES),
        repeat=inp["part"]["repeat"],
        fold=inp["part"]["fold"],
        subset=subset,
        arm=arm,
        size_um=SIZES[arrays["size_bucket"]],
        grain_dataset_idx=np.repeat(ids, counts),
        grain_size_um=np.repeat(SIZES[arrays["size_bucket"]], counts),
        legacy_pred_peak=arrays["original_prediction"],
        new_raw_grain_prediction=result["prediction"],
        new_raw_grain_maximum_prediction=result["maximum_prediction"],
        new_head_has_no_uncertainty=np.array(True),
        legacy_node_mean_names=np.array(
            ["lcf_fs_peak"] + [s["name"] for s in inp["ds"].per_grain_aux_targets]
        ),
        legacy_graph_mean_names=np.array(
            ["lcf_fs_max"] + [s["name"] for s in inp["ds"].aux_pergraph_targets]
        ),
    )
    path = dest / f"{subset}_joint_predictions.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path) as restored:
        for key, expected in arrays.items():
            actual = restored[key]
            assert (
                np.array_equal(actual, expected, equal_nan=True)
                if actual.dtype.kind in "fc"
                else np.array_equal(actual, expected)
            ), key
    report = dict(
        subset=subset,
        path=str(path),
        archive_sha256=sha(path),
        n_sves=len(ids),
        n_grains=int(counts.sum()),
        original_outputs_identical_to_frozen_cache=True,
        canonical_identity_readback=True,
        new_negative_raw_grain_counts=(result["prediction"] < 0).sum(0).tolist(),
        scientific_outer_metrics_deferred_to_all20_gate=subset == "evaluation",
    )
    dump(dest / f"{subset}_export.json", report)
    return report


def validate_release(path):
    release = json.loads(Path(path).read_text())
    assert release["source_manifest"] == sources()
    assert release["arms"] == list(ARMS) and release["target_fits"] == 20
    assert release["head_training_device"] == "cuda:0"
    assert release["partitions_sha256"] == sha(legacy.FROZEN / "partitions.json")
    expected = {
        f"repeat{p['repeat']}_fold{p['fold']}" for p in legacy.load_partitions()
    }
    assert set(release["preparations"]) == expected
    for part in legacy.load_partitions():
        key = f"repeat{part['repeat']}_fold{part['fold']}"
        assert (
            sha(preparation_dir(part) / "preparation.json")
            == release["preparations"][key]
        )
    return release


def sealed_outer_cache(inp, prepared, seal_path):
    seal = validate_selection_seal(inp, seal_path)
    part = inp["part"]
    directory = (
        HERE / "outer_cache" / version() / f"repeat{part['repeat']}_fold{part['fold']}"
    )
    expected = dict(
        partition=part,
        source_manifest=sources(),
        original_checkpoint=prepared["original_checkpoint"],
        preprocessing_sha256=prepared["preprocessing_sha256"],
        cache_device=str(source_device(inp)),
    )

    def build():
        original, checkpoint_info = original_model(inp, source_device(inp))
        assert checkpoint_info == prepared["original_checkpoint"]
        values, report = cache_subset(
            inp, original, "evaluation", source_device(inp), selection_seal=seal_path
        )
        assert sha(checkpoint_info["path"]) == checkpoint_info["sha256"]
        assert legacy.joint.state_hash(original) == checkpoint_info["full_state_sha256"]
        report.update(
            created_after_first_arm_seal=str(seal_path),
            first_arm_seal_sha256=sha(seal_path),
            first_selected_head_sha256=seal["selected_head_sha256"],
        )
        return (values, report)

    values, report = immutable_shared_cache(directory, expected, build)
    assert values["dataset_idx"].tolist() == part["evaluation"]
    return (values, report)


def fit(inp, arm, device, release):
    part = inp["part"]
    assert str(device) == release["head_training_device"] == "cuda:0"
    assert release["source_manifest"] == sources(), "Released source changed before fit"
    pair_key = f"repeat{part['repeat']}_fold{part['fold']}"
    assert (
        sha(preparation_dir(part) / "preparation.json")
        == release["preparations"][pair_key]
    )
    dest = (
        HERE / "runs" / version() / arm / f"repeat{part['repeat']}_fold{part['fold']}"
    )
    if dest.exists():
        raise RuntimeError(
            f"Existing completed/failed/partial run is preserved: {dest}"
        )
    prepared = prepare(inp)
    prepdir = preparation_dir(part)
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        training = torch.load(
            prepdir / "fitting_cache.pt", weights_only=True, map_location="cpu"
        )
        stop = torch.load(
            prepdir / "stopping_cache.pt", weights_only=True, map_location="cpu"
        )
        model = construct(inp, device)
        model.load_state_dict(
            torch.load(
                prepdir / "initial_head.pt", weights_only=True, map_location=device
            ),
            strict=True,
        )
        assert legacy.joint.state_hash(model) == prepared["paired_initial_head_sha256"]
        optimizer, scheduler = optimizer_and_scheduler(model)
        dump(
            dest / "protocol.json",
            dict(
                arm=arm,
                partition=part,
                release=release,
                initial_head_sha256=legacy.joint.state_hash(model),
                original_checkpoint=prepared["original_checkpoint"],
                optimizer_parameter_count=16708,
                identity_epoch_minus_one_eligible=True,
                no_original_model_present_in_training_loop=True,
                cache_device=str(source_device(inp)),
                head_training_device=str(device),
            ),
        )
        initial = stopping(model, stop)
        dump(dest / "identity_stopping.json", initial)
        best, best_epoch = (initial["score"], -1)

        def checkpoint(epoch, score, path):
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch,
                    stopping_score=score,
                    arm=arm,
                    source_manifest=sources(),
                    original_checkpoint_sha256=prepared["original_checkpoint"][
                        "sha256"
                    ],
                ),
                path,
            )

        checkpoint(-1, best, dest / "best.pt")
        history = []
        for epoch in range(MAX_EPOCHS):
            model.train()
            sums = dict(total=0.0, allgrain=0.0, topfour=0.0, maximum=0.0)
            count = 0
            for batch in batches(training, inp["permutations"][epoch], device=device):
                optimizer.zero_grad(set_to_none=True)
                p = model(
                    batch["features"],
                    batch["original_prediction"],
                    batch["buckets"][batch["membership"]],
                )
                losses = model.losses(p, batch, treatment=arm == "critical_treatment")
                losses["total"].backward()
                if not all(
                    (
                        p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                ):
                    raise FloatingPointError(
                        "Nonfinite head gradient; identity is not a fallback"
                    )
                optimizer.step()
                n = len(batch["buckets"])
                for key in sums:
                    sums[key] += float(losses[key].detach()) * n
                count += n
            selected = stopping(model, stop)
            score = selected["score"]
            scheduler.step(score)
            if score < best:
                best, best_epoch = (score, epoch)
                checkpoint(epoch, score, dest / "best.pt")
            row = dict(
                epoch_zero_based=epoch,
                fitting_losses={key: value / count for key, value in sums.items()},
                stopping_score=score,
                stopping_cells=selected["cells"],
                best_epoch_zero_based=best_epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                negative_raw_grain_counts=selected["negative_raw_grain_counts"],
                elapsed_seconds=time.monotonic() - started,
            )
            history.append(row)
            dump(dest / "history.json", history)
            print(
                json.dumps(
                    dict(arm=arm, repeat=part["repeat"], fold=part["fold"], **row)
                ),
                flush=True,
            )
            if epoch - best_epoch >= PATIENCE:
                break
        checkpoint(epoch, score, dest / "last.pt")
        state = torch.load(dest / "best.pt", weights_only=True, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        np.testing.assert_allclose(
            stopping(model, stop)["score"], best, rtol=1e-12, atol=1e-12
        )
        assert (
            release["source_manifest"] == sources()
        ), "Released source changed during fit"
        assert sha(prepdir / "preparation.json") == release["preparations"][pair_key]
        seal = dest / "selected_head.json"
        dump(
            seal,
            dict(
                partition=part,
                selected_head_checkpoint=str(dest / "best.pt"),
                selected_head_sha256=sha(dest / "best.pt"),
                selected_epoch_zero_based=best_epoch,
                strict_reloaded=True,
                before_outer_target_extraction=True,
                before_outer_transform_cache=True,
            ),
        )
        reports = [
            export(model, data, inp, arm, name, dest)
            for name, data in (("fitting", training), ("stopping", stop))
        ]
        outer, cache_report = sealed_outer_cache(inp, prepared, seal)
        dump(dest / "evaluation_cache_provenance.json", cache_report)
        reports.append(export(model, outer, inp, arm, "evaluation", dest))
        checkpoint_info = prepared["original_checkpoint"]
        assert sha(checkpoint_info["path"]) == checkpoint_info["sha256"]
        assert (
            release["source_manifest"] == sources()
        ), "Released source changed before completion"
        completion = dict(
            status="complete",
            arm=arm,
            partition=part,
            actual_update_epochs=len(history),
            selected_epoch_zero_based=best_epoch,
            selected_identity=best_epoch == -1,
            identity_stopping_score=initial["score"],
            selected_stopping_score=best,
            head_checkpoint_sha256=sha(dest / "best.pt"),
            original_checkpoint=checkpoint_info,
            original_outputs_preserved=True,
            optimizer_head_only=True,
            exports=reports,
            paired_outer_inputs_and_legacy_arrays_exact=True,
            shared_outer_cache_sha256=cache_report["cache_sha256"],
            stop_reason="budget_cap120" if len(history) == MAX_EPOCHS else "patience50",
            elapsed_seconds=time.monotonic() - started,
            outer_comparison_not_performed=True,
        )
        dump(dest / "completion.json", completion)
        return completion
    except Exception as error:
        dump(
            dest / "failure.json",
            dict(
                status="failed",
                arm=arm,
                partition=part,
                error=repr(error),
                traceback=traceback.format_exc(),
                no_identity_fallback=True,
                no_failed_sves_dropped=True,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="0,0")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--arm", choices=("both",) + ARMS, default="both")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if not (args.prepare or args.smoke or args.fit):
        parser.error("Choose an explicit operation")
    if args.fit and (not args.release):
        parser.error("Scientific fits require --release training configuration")
    if args.fit and args.device != "cuda:0":
        parser.error("All20 new-head fits are fixed to --device cuda:0")
    release = validate_release(args.release) if args.fit else None
    legacy.init_distributed()
    try:
        for part in legacy.load_partitions():
            if args.pair != "all" and args.pair != f"{part['repeat']},{part['fold']}":
                continue
            inp = inputs(part)
            if args.prepare:
                prepare(inp)
                print("PREPARED", part["repeat"], part["fold"], flush=True)
            if args.smoke:
                from smoke import run

                run(inp, torch.device(args.device))
            if args.fit:
                for arm in ARMS if args.arm == "both" else (args.arm,):
                    fit(inp, arm, torch.device(args.device), release)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del inp
            gc.collect()
        if args.prepare:
            paths = {
                f"repeat{p['repeat']}_fold{p['fold']}": preparation_dir(p)
                / "preparation.json"
                for p in legacy.load_partitions()
            }
            if all((path.exists() for path in paths.values())):
                template = dict(
                    source_manifest=sources(),
                    partitions_sha256=sha(legacy.FROZEN / "partitions.json"),
                    arms=list(ARMS),
                    target_fits=20,
                    head_training_device="cuda:0",
                    preparations={key: sha(path) for key, path in paths.items()},
                )
                dump(HERE / f"training_config_{version()}.json", template)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
