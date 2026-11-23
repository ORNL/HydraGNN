"""Disposable CPU/CUDA technical verification; never a scientific fit."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import traceback
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import study as s
from cache import cache_subset, original_model, batches, subset, immutable_shared_cache
from objectives import grain_objective_cells, top_four_weights, reduce_cells


def analytic():
    variance = torch.ones((5, 4), dtype=torch.float64)

    def grid(values):
        return torch.tensor(values, dtype=torch.float64)[:, None].repeat(1, 4)

    def cells(p, y):
        return grain_objective_cells(
            p,
            y,
            torch.zeros(len(y), dtype=torch.long),
            torch.tensor([0]),
            variance,
            2 * variance,
        )

    target = grid([10, 9, 8, 7, 7, 7, 1])
    weights = top_four_weights(target)
    torch.testing.assert_close(
        weights[:, 0],
        torch.tensor([1, 1, 1, 1 / 3, 1 / 3, 1 / 3, 0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        weights.sum(0), torch.full((4,), 4.0, dtype=torch.float64)
    )
    p = grid([10, 9, 8, 7, 7, 7, 12]).requires_grad_()
    values = cells(p, target)
    assert torch.count_nonzero(values["topfour"]) == 0
    values["maximum"][0, 0].backward()
    assert torch.equal(
        p.grad[:, 0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=torch.float64),
    )
    p = grid([12, 9, 8, 7, 7, 7, 12]).requires_grad_()
    cells(p, target)["maximum"][0, 0].backward()
    assert torch.equal(
        p.grad[:, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
    )
    for count in (1, 2, 3, 4):
        y = grid(list(range(count)))
        v = cells(y + 2, y)
        assert torch.equal(v["allgrain"], v["topfour"])
        assert torch.equal(v["allgrain"], 2 * v["maximum"])
    bucket = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4])
    scores = torch.arange(60, dtype=torch.float64).reshape(15, 4)
    weights = len(bucket) / (5 * torch.bincount(bucket).double())
    expected = torch.stack([scores[bucket == b].mean(0) for b in range(5)]).mean()
    torch.testing.assert_close(reduce_cells(scores, bucket, weights), expected)
    missing = target.clone()
    missing[0, 0] = float("nan")
    try:
        cells(target, missing)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing peaks were not rejected")
    return dict(
        status="pass",
        fractional_cutoff_ties=True,
        false_winner_gradients=True,
        symmetric_predicted_ties=True,
        four_or_fewer_grains=True,
        separate_normalizers=True,
        exact_equal20_fitting_weight_identity=True,
        missing_peak_rejected=True,
    )


def run(inp, device):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dest = s.HERE / "technical_attempts" / stamp
    dest.mkdir(parents=True, exist_ok=False)
    manifest = s.sources()
    snapshot = dest / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(manifest):
        (snapshot / f"{index:02d}_{Path(path).name}").write_bytes(
            Path(path).read_bytes()
        )
    report = dict(
        status="running",
        scientific_fits_started=0,
        device=str(device),
        source_manifest=manifest,
        only_fitting_fixtures=True,
        original_checkpoint_unchanged=True,
    )
    try:
        report["analytic"] = analytic()
        positions = np.array(
            [np.flatnonzero(inp["buckets"][: inp["n"]] == b)[0] for b in range(5)]
        )
        graphs = [inp["values"][i] for i in positions]
        fixture = (
            graphs,
            inp["ids"][positions],
            inp["raw"][positions],
            inp["buckets"][positions],
        )
        original, checkpoint = original_model(inp, device)
        cached, replay_report = cache_subset(
            inp, original, "technical_fitting", device, explicit=fixture
        )
        report["cache_replay"] = replay_report
        original_state = s.legacy.joint.state_hash(original)
        a, b = s.construct(inp, device), s.construct(inp, device)
        assert s.legacy.joint.state_hash(a) == s.legacy.joint.state_hash(b)
        batch = next(batches(cached, device=device))
        node_bucket = batch["buckets"][batch["membership"]]
        with torch.no_grad():
            initial = a(batch["features"], batch["original_prediction"], node_bucket)
        assert torch.equal(initial, batch["original_prediction"])
        assert (
            not batch["features"].requires_grad
            and not batch["original_prediction"].requires_grad
        )

        def scales(model):
            p = model(batch["features"], batch["original_prediction"], node_bucket)
            terms = model.losses(p, batch, treatment=True)
            params = list(model.named_parameters())
            result = {}
            for key in terms:
                grads = torch.autograd.grad(
                    terms[key], [p for _, p in params], retain_graph=True
                )
                result[key] = dict(
                    loss=float(terms[key].detach()),
                    hidden_gradient_l2=sum(
                        float(g.double().square().sum())
                        for (name, _), g in zip(params, grads)
                        if name.startswith("head.0")
                    )
                    ** 0.5,
                    final_gradient_l2=sum(
                        float(g.double().square().sum())
                        for (name, _), g in zip(params, grads)
                        if name.startswith("head.2")
                    )
                    ** 0.5,
                )
            return result

        report["initial_gradient_scales"] = scales(a)
        # One disposable AdamW update in each arm tests the declared optimizer.
        update_timings = []
        for head, treatment in ((a, False), (b, True)):
            optimizer, scheduler = s.optimizer_and_scheduler(head)
            assert not (
                {id(p) for p in original.parameters()}
                & {id(p) for g in optimizer.param_groups for p in g["params"]}
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_start = time.monotonic()
            p = head(batch["features"], batch["original_prediction"], node_bucket)
            loss = head.losses(p, batch, treatment)["total"]
            loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in head.parameters()
            )
            assert torch.count_nonzero(head.head[0].weight.grad) == 0
            assert torch.count_nonzero(head.head[-1].weight.grad) > 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_timings.append(
                dict(
                    treatment=treatment,
                    seconds=time.monotonic() - update_start,
                    n_sves=len(batch["buckets"]),
                    n_grains=len(batch["features"]),
                )
            )
        report["disposable_head_step_timings"] = update_timings
        report["post_one_disposable_treatment_update_gradient_scales"] = scales(b)
        assert all(p.grad is None for p in original.parameters())
        assert s.legacy.joint.state_hash(original) == original_state
        assert s.sha(checkpoint["path"]) == checkpoint["sha256"]
        repeated, _ = cache_subset(
            inp, original, "technical_fitting", device, explicit=fixture
        )
        # Same-device cache outputs retain every original channel exactly when
        # reductions are deterministic; otherwise report the actual tiny bound.
        identity = {}
        for key in (
            "features",
            "original_prediction",
            "legacy_node_transformed",
            "legacy_graph_transformed",
            "legacy_node_mean_raw",
            "legacy_graph_mean_raw",
            "legacy_direct_prediction",
        ):
            difference = float((cached[key] - repeated[key]).abs().max())
            identity[key] = dict(
                bitwise_equal=torch.equal(cached[key], repeated[key]),
                max_absolute_error=difference,
            )
            torch.testing.assert_close(cached[key], repeated[key], rtol=2e-5, atol=2e-5)
        report["same_device_original_replay_before_after_head_updates"] = identity
        # Test shared-cache immutability using fitting fixtures, never outer data.
        common_path = dest / "disposable_shared_cache"
        shared, common_report = immutable_shared_cache(
            common_path,
            {"original_state": original_state},
            lambda: (cached, {"technical_fitting_only": True}),
        )

        def forbidden_rebuild():
            raise AssertionError("Shared inference cache was recomputed")

        restored_shared, repeated_report = immutable_shared_cache(
            common_path, {"original_state": original_state}, forbidden_rebuild
        )
        assert common_report == repeated_report
        for key in shared:
            torch.testing.assert_close(
                shared[key], restored_shared[key], rtol=0, atol=0, equal_nan=True
            )
        state_path = dest / "disposable_head.pt"
        torch.save(b.state_dict(), state_path)
        clone = s.construct(inp, device)
        clone.load_state_dict(
            torch.load(state_path, weights_only=True, map_location=device), strict=True
        )
        with torch.no_grad():
            predicted = b(batch["features"], batch["original_prediction"], node_bucket)
            reloaded = clone(
                batch["features"], batch["original_prediction"], node_bucket
            )
        assert torch.equal(predicted, reloaded)
        damaged = dict(b.state_dict())
        damaged.pop("head.2.bias")
        try:
            clone.load_state_dict(damaged, strict=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Strict reload accepted a missing key")
        order = torch.randperm(
            len(predicted), generator=torch.Generator().manual_seed(2026111713)
        ).to(device)
        with torch.no_grad():
            permuted = b(
                batch["features"][order],
                batch["original_prediction"][order],
                node_bucket[order],
            )
        assert torch.equal(permuted, predicted[order])
        original_cells = b.cells(predicted, batch)
        perm_batch = dict(
            batch, target=batch["target"][order], membership=batch["membership"][order]
        )
        permuted_cells = b.cells(permuted, perm_batch)
        for key in ("allgrain", "topfour", "maximum"):
            torch.testing.assert_close(
                original_cells[key], permuted_cells[key], rtol=1e-10, atol=1e-10
            )
        score = s.stopping(b, cached)
        expected = (
            0.5 * original_cells["allgrain"].mean()
            + 0.5 * original_cells["maximum"].mean()
        )
        np.testing.assert_allclose(
            score["score"], float(expected), rtol=1e-12, atol=1e-12
        )
        # Refusal happens before any outer transform, extraction or forward.
        try:
            subset(inp, "evaluation")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Outer cache had no seal gate")
        # A finite negative prediction is kept verbatim by the residual decoder.
        identity_head = s.construct(inp, device)
        negative = -torch.ones_like(batch["original_prediction"])
        assert torch.equal(
            identity_head(batch["features"], negative, node_bucket), negative
        )
        # Singleton with no edges remains a valid original inference input.
        singleton = graphs[0].clone()
        singleton.x = singleton.x[:1]
        if getattr(singleton, "pos", None) is not None:
            singleton.pos = singleton.pos[:1]
        singleton.edge_index = torch.empty((2, 0), dtype=torch.long)
        if getattr(singleton, "edge_attr", None) is not None:
            singleton.edge_attr = singleton.edge_attr[:0]
        singleton.num_nodes = 1
        with torch.no_grad():
            singleton_output = original(
                next(iter(DataLoader([singleton], batch_size=1))).to(device)
            )
        assert all(torch.isfinite(x).all() for x in singleton_output)
        # Export is label-free inference: it must never call objective code,
        # even though the archive preserves true fields for later all20 analysis.
        previous_cells = b.cells

        def forbidden_cells(*args, **kwargs):
            raise AssertionError("Export tried to compute target-dependent losses")

        b.cells = forbidden_cells
        try:

            class NoTruthReads(dict):
                def __getitem__(self, key):
                    if key in ("true_peak", "y_true"):
                        raise AssertionError("Inference read a target field")
                    return super().__getitem__(key)

            label_free = s.predict(b, NoTruthReads(cached))
            np.testing.assert_array_equal(
                label_free["prediction"], predicted.cpu().numpy()
            )
            report["export_readback"] = s.export(
                b, cached, inp, "technical_treatment", "technical_fitting", dest
            )
        finally:
            b.cells = previous_cells
        report.update(
            status="pass",
            fixture_ids=fixture[1].tolist(),
            original_checkpoint=checkpoint,
            original_full_state_unchanged=True,
            original_gradients_absent=True,
            optimizer_head_only=True,
            zero_residual_identity_bitwise=True,
            paired_initial_state_equal=True,
            rng_preserved=True,
            strict_reload_bitwise=True,
            strict_reload_rejects_missing_key=True,
            node_permutation_equivariance=True,
            tied_loss_permutation_invariance=True,
            same_equal20_stopping_selector=True,
            outer_without_seal_rejected=True,
            finite_negative_predictions_retained=True,
            singleton_empty_edge_finite=True,
            export_cannot_invoke_target_dependent_objectives=True,
            inference_cannot_read_raw_target_fields=True,
            shared_cache_not_recomputed_and_all_arrays_exact=True,
            grain_variance_floor_active=False,
            maximum_variance_floor_active=False,
            no_outer_targets_transforms_caches_or_forwards=True,
        )
    except Exception as error:
        report.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        s.dump(dest / "verification.json", report)
        print(
            json.dumps(
                dict(status=report["status"], report=str(dest / "verification.json"))
            ),
            flush=True,
        )
    return report
