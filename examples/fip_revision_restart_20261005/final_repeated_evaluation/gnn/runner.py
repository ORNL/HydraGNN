"""Matched fold-specific full joint SAGE recipe; outer inference after selection only."""
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
import dataclasses
import hashlib
import json
import pickle
import shutil
import sys
import time
import types
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
WSL_RESTART = Path(str(release_root() / "examples/fip_revision_restart_20261005"))
JOINT = WSL_RESTART / "joint_normalization_fcc"
WINDOWS = Path(
    str(
        release_root()
        / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
    )
)
sys.path.insert(0, str(JOINT))
import run as joint

CASES, SIZES = (joint.CASES, joint.SIZES)
EXPECTED_PARTITIONS_SHA = (
    "a6c7cf5a79114d6fb296a3511b155a43ede3932ebd2d266030925ad6aa9a0c39"
)


def sync(dest):
    for path in dest.rglob("*"):
        if path.is_file() and path.suffix in [
            ".json",
            ".csv",
            ".npz",
            ".pt",
            ".py",
            ".log",
            ".pkl",
        ]:
            target = WINDOWS / "gnn" / path.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def load_partition(repeat, fold):
    path = WINDOWS / "partitions.json"
    assert joint.sha(path) == EXPECTED_PARTITIONS_SHA
    return next(
        (
            part
            for part in json.loads(path.read_text())
            if part["repeat"] == repeat and part["fold"] == fold
        )
    )


def fold_inputs(part):
    cfg = json.loads(
        (joint.CONTROL / "results/sage_sorted_v2_seed101/config.json").read_text()
    )
    recipe = cfg["Recipe"]
    fields = {
        field.name
        for field in dataclasses.fields(joint.dataset.FIPDataset)
        if field.init and (not field.name.startswith("_"))
    }
    kwargs = {key: value for key, value in recipe.items() if key in fields}
    kwargs["seed"] = part["partition_seed"]
    ds = joint.dataset.FIPDataset(**kwargs)
    ds._load()
    assert len(ds._raw) == 940
    ds.splits = joint.dataset.FIPSplits(
        train=part["fitting"], val=part["stopping"], test=part["evaluation"]
    )
    assert not set(ds.splits.train) & set(ds.splits.val)
    assert not (set(ds.splits.train) | set(ds.splits.val)) & set(ds.splits.test)
    assert set(ds.splits.train) | set(ds.splits.val) | set(ds.splits.test) == set(
        range(940)
    )
    ds._fit_scalers()
    ids = np.r_[ds.splits.train, ds.splits.val]
    graphs = [ds._transform(ds._raw[int(index)]) for index in ids]
    raw = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(
                    ds._raw[int(index)], case + "_pergraph_max"
                )
                for case in CASES
            ]
            for index in ids
        ],
        dtype=np.float64,
    )
    truth_contract = np.load(WINDOWS / "regressors/raw_truth940.npz")
    assert np.array_equal(truth_contract["dataset_idx"], np.arange(940))
    assert np.array_equal(truth_contract["cases"], np.array(CASES))
    assert np.array_equal(raw, truth_contract["raw_truth"][ids])
    buckets = np.array([int(g.size_bucket[0]) for g in graphs])
    n = len(ds.splits.train)
    mean = np.stack([raw[:n][buckets[:n] == b].mean(0) for b in range(5)])
    sd = np.stack([raw[:n][buckets[:n] == b].std(0, ddof=0) for b in range(5)])
    variance = np.stack([raw[:n][buckets[:n] == b].var(0, ddof=1) for b in range(5)])
    assert np.all(sd > 0) and np.all(variance > 0)
    grains = np.concatenate(
        [
            np.column_stack(
                [
                    getattr(ds._raw[int(index)], case + "_fip_dist")[:, 0].numpy()
                    for case in CASES
                ]
            )
            for index in ds.splits.train
        ]
    ).astype(np.float64)
    grain_sd = grains.std(0, ddof=0)
    components = []
    for c in range(4):
        scaler = ds._scaler_pergraph_y if c == 0 else ds._scaler_aux_pergraph_y[c - 1]
        state = (
            ds._pergraph_target_state
            if c == 0
            else ds._aux_pergraph_target_states[c - 1]
        )
        pt = state["power_transformer"]
        components.append(
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
    inputs = (
        cfg,
        ds,
        ids,
        graphs,
        raw,
        buckets,
        mean,
        sd,
        variance,
        grain_sd,
        components,
    )
    return inputs


def construct(cfg, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    factory = types.ModuleType("hydragnn.models.final_joint_factory")
    factory.__package__ = "hydragnn.models"
    path = joint.CONTROL / "create_seed_controlled.py"
    exec(compile(path.read_text(), str(path), "exec"), factory.__dict__)
    model = factory.create_model_config(
        config=cfg["NeuralNetwork"], verbosity=0, use_gpu=False
    )
    assert joint.train._install_residual_node_head(model) == 1
    model._ensure_graph_concat_projector(5, 256, torch.device("cpu"), torch.float32)
    joint.losses.override_model_loss(model, cfg["Recipe"]["loss_recipe"])
    assert sum((p.numel() for p in model.parameters())) == 377974
    assert [float(w) for w in model.loss_weights] == [0.5, 0.5]
    return model.to(device)


def technical_smoke(inputs, part):
    (
        cfg,
        ds,
        ids,
        graphs,
        raw,
        buckets,
        mean,
        sd,
        variance,
        grain_sd,
        components,
    ) = inputs
    rows = []
    for variant in joint.ARMS:
        normalized, augmented = (
            variant in ["normalization", "both"],
            variant in ["fcc", "both"],
        )
        values = joint.build_graphs(
            graphs[:2], raw[:2], buckets[:2], mean, sd, normalized
        )
        fitting = joint.FittingDataset(ds, values, augmented)
        batch = next(
            iter(DataLoader(fitting, batch_size=2, shuffle=False, num_workers=0))
        )
        model = construct(cfg, part["initialization_seed"], torch.device("cpu"))
        optimizer, _ = joint.train.build_optimizer_and_scheduler(model, cfg)
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
            id(p) for p in model.parameters()
        }
        pn, pg = model(batch)
        actual = 0.5 * model.loss_function(
            pn, batch.pn_target
        ) + 0.5 * model.loss_function(pg, batch.pg_target)
        from hydragnn.train.train_validate_test import get_head_indices

        source, _ = model.loss(
            [pn, pg],
            batch.y,
            get_head_indices(types.SimpleNamespace(module=model), batch),
        )
        assert torch.equal(actual, source)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        actual.backward()
        assert torch.isfinite(actual) and all(
            (
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        optimizer.step()
        changed = [
            name
            for name, p in model.named_parameters()
            if not torch.equal(p, before[name])
        ]
        for prefix in ["heads_NN.0.", "heads_NN.1.", "graph_concat_projector."]:
            assert any((name.startswith(prefix) for name in changed))
        rows.append(
            dict(
                variant=variant,
                source_joint_loss_exact=True,
                finite_gradients=True,
                optimizer_tracks_all_parameters=True,
                node_graph_context_updates=True,
                changed_parameter_tensors=len(changed),
                technical_batch_size=2,
            )
        )
    rng = np.random.default_rng(2026100518)
    positions = np.concatenate(
        [np.flatnonzero(buckets[: len(ds.splits.train)] == b)[:2] for b in range(5)]
    )
    for position in positions:
        original = graphs[position]
        source = ds._raw[int(ids[position])]
        operators = rng.integers(0, 24, size=source.num_nodes, dtype=np.uint8)
        q, schmid, _ = joint.equivalent_features(source, operators)
        selected = ds._select_x(source.x).numpy().copy()
        selected[:, :4], selected[:, 4:16] = (q, schmid)
        fast = torch.from_numpy(ds._scaler_x.transform(selected).astype(np.float32))
        full, _ = joint.transformed_equivalent(ds, source, operators)
        assert torch.equal(fast, full.x)
        assert torch.equal(fast[:, 4:], original.x[:, 4:])
        assert torch.equal(full.y, original.y)
        assert torch.equal(full.edge_index, original.edge_index)
        assert torch.equal(full.graph_attr, original.graph_attr)
    for normalized in [False, True]:
        values = joint.build_graphs(graphs, raw, buckets, mean, sd, normalized)
        restored = joint.inverse_graph(
            torch.stack([g.pg_target[0] for g in values]),
            torch.tensor(buckets),
            normalized,
            mean,
            sd,
            components,
        ).numpy()
        assert np.allclose(restored, raw, rtol=2e-06, atol=1e-10)
    return dict(
        status="pass",
        four_actual_joint_gradient_smokes=rows,
        fold_local_fcc_fast_full_exact=True,
        fixed_fixture_positions=positions.tolist(),
        fixture_ids=ids[positions].tolist(),
        preserves_nonquaternion_features_targets_geometry_context=True,
        fitting_stopping_inverse_roundtrips=True,
        no_outer_transform_or_forward=True,
    )


def output_endpoints(model, graphs, normalized, inputs, part, name, subset, dest):
    (
        cfg,
        ds,
        ids,
        originals,
        raw,
        buckets,
        mean,
        sd,
        variance,
        grain_sd,
        components,
    ) = inputs
    endpoint = joint.evaluate(model, graphs, normalized, inputs, grain=True)
    positions = endpoint["positions"]
    frame = []
    for i, position in enumerate(positions):
        for c, case in enumerate(CASES):
            frame.append(
                dict(
                    model=name,
                    repeat=part["repeat"],
                    fold=part["fold"],
                    dataset_idx=int(ids[position]),
                    size_um=SIZES[int(buckets[position])],
                    case=case,
                    truth=float(raw[position, c]),
                    prediction=float(endpoint["pred"][i, c]),
                    target="SVE maximum",
                    units="dimensionless" if case.endswith("fs") else "MPa",
                )
            )
    pd.DataFrame(frame).to_csv(dest / (subset + "_predictions.csv"), index=False)
    np.savez(
        dest / (subset + "_predictions.npz"),
        dataset_idx=ids[positions],
        size_bucket=buckets[positions],
        y_true=raw[positions],
        y_pred=endpoint["pred"],
        cases=CASES,
    )
    rows = []
    for c, case in enumerate(CASES):
        quality = endpoint["grain_quality"]
        rows.append(
            dict(
                model=name,
                repeat=part["repeat"],
                fold=part["fold"],
                subset=subset,
                case=case,
                target="grain peak FIP",
                observations=quality["observations"],
                n_grains=quality["n_grains"],
                raw_rmse=quality["raw_rmse"][c],
                fitting_case_sd=quality["fitting_case_sd"][c],
                normalized_rmse=quality["normalized_rmse"][c],
                units="dimensionless" if case.endswith("fs") else "MPa",
            )
        )
    pd.DataFrame(rows).to_csv(dest / (subset + "_grain_quality.csv"), index=False)
    return endpoint


def fit(name, variant, inputs, part, device, permutations, pair_hash):
    (
        cfg,
        ds,
        ids,
        graphs,
        raw,
        buckets,
        mean,
        sd,
        variance,
        grain_sd,
        components,
    ) = inputs
    n = len(ds.splits.train)
    dest = HERE / f"repeat{part['repeat']}_fold{part['fold']}" / name
    if (dest / "completion.json").exists():
        sealed = json.loads((dest / "completion.json").read_text())
        assert sealed["model"] == name and sealed["variant"] == variant
        assert sealed["repeat"] == part["repeat"] and sealed["fold"] == part["fold"]
        assert sealed["device"] == str(
            device
        ), "Partial pair must resume on its declared matched device"
        assert (dest / "evaluation_predictions.csv").exists()
        print(f"SEALED_MODEL_SKIPPED {dest}", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    normalized, augmented = (
        variant in ["normalization", "both"],
        variant in ["fcc", "both"],
    )
    values = joint.build_graphs(graphs, raw, buckets, mean, sd, normalized)
    dataset = joint.FittingDataset(ds, values[:n], augmented)
    model = construct(cfg, part["initialization_seed"], device)
    assert joint.state_hash(model) == pair_hash
    optimizer, scheduler = joint.train.build_optimizer_and_scheduler(model, cfg)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
    joint.dump(
        dest / "initialization.json",
        dict(
            full_state_sha256=pair_hash,
            full_model_parameters=377974,
            seed=part["initialization_seed"],
            device=str(device),
            task_weights=[0.5, 0.5],
            optimizer_tracks_all_parameters=True,
        ),
    )
    start = time.perf_counter()
    history, best, best_epoch = ([], float("inf"), -1)
    for epoch in range(120):
        model.train()
        sums, count = (np.zeros(3), 0)
        for batch in DataLoader(
            dataset,
            batch_size=16,
            sampler=permutations[epoch].tolist(),
            num_workers=0,
            pin_memory=device.type == "cuda",
        ):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pn, pg = model(batch)
            node_loss, graph_loss = (
                model.loss_function(pn, batch.pn_target),
                model.loss_function(pg, batch.pg_target),
            )
            loss = 0.5 * node_loss + 0.5 * graph_loss
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            sums += (
                np.array([loss.item(), node_loss.item(), graph_loss.item()])
                * batch.num_graphs
            )
            count += batch.num_graphs
        stopping = joint.evaluate(model, values[n:], normalized, inputs)
        score = stopping["score"]
        scheduler.step(score)
        if score < best:
            best, best_epoch = (score, epoch)
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch,
                    raw_stopping_score=score,
                ),
                dest / "best.pt",
            )
        row = dict(
            epoch_zero_based=epoch,
            raw_stopping_maximum_score=score,
            fitting_joint_loss=float(sums[0] / count),
            fitting_node_loss=float(sums[1] / count),
            fitting_graph_loss=float(sums[2] / count),
            best_epoch_zero_based=best_epoch,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            seconds=time.perf_counter() - start,
        )
        history.append(row)
        joint.dump(dest / "history.json", history)
        print(
            json.dumps(
                dict(model=name, repeat=part["repeat"], fold=part["fold"], **row)
            ),
            flush=True,
        )
        if epoch - best_epoch >= 50:
            break
    torch.save(
        dict(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
        ),
        dest / "last.pt",
    )
    model.load_state_dict(
        torch.load(dest / "best.pt", weights_only=False, map_location=device)[
            "model_state_dict"
        ],
        strict=True,
    )
    checkpoint_sha = joint.sha(dest / "best.pt")
    output_endpoints(model, values[:n], normalized, inputs, part, name, "fitting", dest)
    output_endpoints(
        model, values[n:], normalized, inputs, part, name, "stopping", dest
    )
    outer_ids = np.array(part["evaluation"])
    outer_graphs = [ds._transform(ds._raw[int(index)]) for index in outer_ids]
    outer_raw = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(
                    ds._raw[int(index)], case + "_pergraph_max"
                )
                for case in CASES
            ]
            for index in outer_ids
        ],
        dtype=np.float64,
    )
    assert np.array_equal(
        outer_raw,
        np.load(WINDOWS / "regressors/raw_truth940.npz")["raw_truth"][outer_ids],
    )
    outer_buckets = np.array([int(graph.size_bucket[0]) for graph in outer_graphs])
    outer_inputs = (
        cfg,
        ds,
        outer_ids,
        outer_graphs,
        outer_raw,
        outer_buckets,
        mean,
        sd,
        variance,
        grain_sd,
        components,
    )
    outer_values = joint.build_graphs(
        outer_graphs, outer_raw, outer_buckets, mean, sd, normalized
    )
    output_endpoints(
        model, outer_values, normalized, outer_inputs, part, name, "evaluation", dest
    )
    assert joint.sha(dest / "best.pt") == checkpoint_sha
    joint.dump(
        dest / "completion.json",
        dict(
            status="complete",
            model=name,
            variant=variant,
            repeat=part["repeat"],
            fold=part["fold"],
            device=str(device),
            actual_epochs=len(history),
            selected_epoch_zero_based=best_epoch,
            stop_reason="budget cap" if len(history) == 120 else "stopping patience50",
            elapsed_seconds=time.perf_counter() - start,
            checkpoint_sha256=checkpoint_sha,
            outer_forward_only_after_checkpoint_frozen=True,
            counts=[len(ds.splits.train), len(ds.splits.val), len(ds.splits.test)],
            fcc_accesses=dataset.accesses,
            fcc_max_schmid_error=dataset.max_schmid_error,
        ),
    )
    sync(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(31500 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    part = load_partition(args.repeat, args.fold)
    inputs = fold_inputs(part)
    (
        cfg,
        ds,
        ids,
        graphs,
        raw,
        buckets,
        mean,
        sd,
        variance,
        grain_sd,
        components,
    ) = inputs
    pair = HERE / f"repeat{part['repeat']}_fold{part['fold']}"
    pair.mkdir(parents=True, exist_ok=True)
    first = construct(cfg, part["initialization_seed"], torch.device("cpu"))
    second = construct(cfg, part["initialization_seed"], torch.device("cpu"))
    pair_hash = joint.state_hash(first)
    assert pair_hash == joint.state_hash(second)
    runtime_gates = technical_smoke(inputs, part)
    joint.dump(pair / "runtime_gates.json", runtime_gates)
    n = len(ds.splits.train)
    permutations = np.stack(
        [
            np.random.default_rng(
                2026100601 + part["repeat"] * 100 + part["fold"] + epoch * 1000
            ).permutation(n)
            for epoch in range(120)
        ]
    )
    np.savez(
        pair / "fitting_statistics.npz",
        dataset_idx=ids,
        size_bucket=buckets,
        mean_by_size=mean,
        sd_by_size=sd,
        checkpoint_variance_by_size=variance,
        grain_case_sd=grain_sd,
        permutations=permutations,
    )
    joint.dump(
        pair / "preparation.json",
        dict(
            partition=part,
            partitions_sha256=EXPECTED_PARTITIONS_SHA,
            full_initial_state_sha256=pair_hash,
            full_model_parameters=377974,
            all_preprocessing_fitted_on_fitting_ids_only=True,
            outer_graphs_not_transformed_or_forwarded=True,
            original_source_config=cfg,
            joint_helper_sha256=joint.sha(JOINT / "run.py"),
            runner_sha256=joint.sha(__file__),
        ),
    )
    slim = __import__("copy").copy(ds)
    slim._raw, slim._processed = ([], [])
    with (pair / "fitted_preprocessing.pkl").open("wb") as stream:
        pickle.dump(slim, stream)
    sync(pair)
    if args.prepare_only:
        torch.distributed.destroy_process_group()
        print("FOLD_PREPARATION_ONLY_COMPLETE", flush=True)
        return
    protocol = json.loads((WINDOWS / "protocol.json").read_text())
    assert (
        protocol.get("selection_frozen") is True
    ), "The selected variant must be fixed before any outer fits"
    assert protocol.get("selected_variant_frozen") is True
    assert (
        json.loads((WINDOWS / "frozen_selection.json").read_text())["status"]
        == "frozen"
    )
    assert protocol["selected_variant"] in joint.ARMS
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    fit("baseline", "neither", inputs, part, device, permutations, pair_hash)
    if protocol["selected_variant"] != "neither":
        fit(
            "selected",
            protocol["selected_variant"],
            inputs,
            part,
            device,
            permutations,
            pair_hash,
        )
    joint.dump(
        pair / "completion.json",
        dict(
            status="complete",
            repeat=part["repeat"],
            fold=part["fold"],
            selected_variant=protocol["selected_variant"],
            paired_device=str(device),
        ),
    )
    sync(pair)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
