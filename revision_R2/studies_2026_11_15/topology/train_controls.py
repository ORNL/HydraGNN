"""Prespecified matched controls: remove adjacency, then explicit degree.

Uses each reference fold's saved transformations, original initialization,
minibatch permutations, stopping rule, objective, and original CPU/CUDA device.
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
import importlib.util
import json
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(str(release_root() / ""))
STUDY = ROOT / "revision_R2/studies_2026_11_15"
FROZEN = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
sys.path.insert(0, str(STUDY / "inference"))
import replay_saved as replay

runner, joint = (replay.runner, replay.joint)
CASES, SIZES = (joint.CASES, joint.SIZES)
ARMS = ["no_message_degree", "no_message_no_degree"]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def controls(graphs, remove_degree):
    result = []
    for g in graphs:
        x = g.clone()
        x.edge_index = x.edge_index[:, :0].clone()
        x.edge_attr = x.edge_attr[:0].clone()
        if remove_degree:
            x.x[:, 19] = 0.0
        result.append(x)
    return result


def export(model, ds, part, pair, stat, subset, name):
    ids = np.asarray(
        part["evaluation" if subset == "evaluation" else "stopping"], dtype=np.int64
    )
    raw = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(ds._raw[int(i)], c + "_pergraph_max")
                for c in CASES
            ]
            for i in ids
        ]
    )
    truth_contract = np.load(FROZEN / "regressors/raw_truth940.npz")
    assert np.array_equal(raw, truth_contract["raw_truth"][ids])
    originals = [ds._transform(ds._raw[int(i)]) for i in ids]
    buckets = np.array([int(g.size_bucket[0]) for g in originals])
    values = controls(
        joint.build_graphs(
            originals, raw, buckets, stat["mean_by_size"], stat["sd_by_size"], False
        ),
        name == "no_message_no_degree",
    )
    components = replay.components_for(ds)
    inputs = (
        None,
        ds,
        ids,
        originals,
        raw,
        buckets,
        stat["mean_by_size"],
        stat["sd_by_size"],
        stat["checkpoint_variance_by_size"],
        stat["grain_case_sd"],
        components,
    )
    device = next(model.parameters()).device
    graph_pred, grain_pred, grain_true = ([], [], [])
    seen = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(values, batch_size=16, shuffle=False, num_workers=0):
            batch = batch.to(device)
            pn, pg = model(batch)
            graph_pred.append(
                joint.inverse_graph(
                    pg,
                    batch.size_bucket.flatten().long(),
                    False,
                    inputs[6],
                    inputs[7],
                    components,
                )
                .cpu()
                .numpy()
            )
            positions = batch.study_position.cpu().tolist()
            seen.extend(ids[positions])
            pred = pn[:, :4].double().cpu().numpy()
            columns = []
            for c in range(4):
                inv = (
                    ds.inverse_pergrain_target
                    if c == 0
                    else lambda value, c=c: ds.inverse_per_grain_aux_target(
                        c - 1, value
                    )
                )
                columns.append(inv(pred[:, c]))
            grain_pred.append(np.column_stack(columns))
            grain_true.append(
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
    graph_pred, grain_pred, grain_true = map(
        np.concatenate, [graph_pred, grain_pred, grain_true]
    )
    assert np.array_equal(ids, seen)
    assert np.isfinite(graph_pred).all() and np.isfinite(grain_pred).all()
    counts = np.array([ds._raw[int(i)].num_nodes for i in ids], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    for j in range(len(ids)):
        assert np.array_equal(grain_true[offsets[j] : offsets[j + 1]].max(0), raw[j])
    size_um = np.asarray(SIZES)[buckets]
    stem = f"grain_predictions_{('' if subset == 'evaluation' else subset + '_')}repeat{part['repeat']}_fold{part['fold']}"
    dest = STUDY / "topology" / name / "data" / (stem + ".npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        cases=np.array(CASES),
        repeat=part["repeat"],
        fold=part["fold"],
        subset=np.array(subset),
        grain_dataset_idx=np.repeat(ids, counts),
        grain_id=np.concatenate([np.arange(n, dtype=np.int32) for n in counts]),
        grain_size_um=np.repeat(size_um, counts),
        pred_peak=grain_pred,
        true_peak=grain_true,
        dataset_idx=ids,
        size_um=size_um,
        size_bucket=buckets,
        grain_count=counts,
        y_true=raw,
        direct_prediction=graph_pred,
        offsets=offsets,
    )
    records = [
        dict(
            model=name,
            repeat=part["repeat"],
            fold=part["fold"],
            dataset_idx=int(i),
            size_um=int(size_um[j]),
            case=c,
            truth=float(raw[j, k]),
            prediction=float(graph_pred[j, k]),
            subset=subset,
        )
        for j, i in enumerate(ids)
        for k, c in enumerate(CASES)
    ]
    pd.DataFrame(records).to_csv(
        pair / name / (subset + "_predictions.csv"), index=False
    )
    meta = dict(
        status="pass",
        model=name,
        repeat=part["repeat"],
        fold=part["fold"],
        subset=subset,
        n_sves=len(ids),
        n_grains=len(grain_pred),
        grain_truth_max_equals_sve_truth=True,
        archive_sha256=joint.sha(dest),
        device=str(device),
        no_adjacency=True,
        explicit_degree_removed=name == "no_message_no_degree",
    )
    dump(dest.with_suffix(".json"), meta)
    return meta


def smoke(cfg, part, graphs, device, remove_degree, expected_hash):
    model = runner.construct(cfg, part["initialization_seed"], device)
    assert joint.state_hash(model) == expected_hash
    optimizer, _ = joint.train.build_optimizer_and_scheduler(model, cfg)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
    batch = next(
        iter(DataLoader(controls(graphs[:2], remove_degree), batch_size=2))
    ).to(device)
    assert batch.edge_index.shape[1] == 0 and batch.edge_attr.shape[0] == 0
    if remove_degree:
        assert (batch.x[:, 19] == 0).all()
    pn, pg = model(batch)
    loss = 0.5 * model.loss_function(pn, batch.pn_target) + 0.5 * model.loss_function(
        pg, batch.pg_target
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        (torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    )
    neighbors = [
        dict(
            name=n,
            count=p.numel(),
            zero_gradient=p.grad is None or bool((p.grad == 0).all()),
        )
        for n, p in model.named_parameters()
        if ".lin_l.weight" in n
    ]
    assert len(neighbors) == 3 and all((x["zero_gradient"] for x in neighbors))
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    for prefix in ["heads_NN.0.", "heads_NN.1.", "graph_concat_projector."]:
        assert any((n.startswith(prefix) for n in changed))
    model.eval()
    with torch.no_grad():
        first = model(batch)
        variant = batch.clone()
        variant.edge_index = variant.edge_index[:, :0]
        if remove_degree:
            variant.x[:, 19] = 0
        second = model(variant)
    assert all((torch.equal(a, b) for a, b in zip(first, second)))
    return dict(
        status="pass",
        original_full_initial_state_sha256=expected_hash,
        full_parameters=sum((p.numel() for p in model.parameters())),
        inactive_neighbor_weight_parameters=sum((x["count"] for x in neighbors)),
        neighbor_weights=neighbors,
        finite_joint_gradient_step=True,
        node_graph_and_context_updated=True,
        zero_degree=remove_degree,
        graph_attr="five size-bucket indicators",
        retains_pooling=True,
    )


def fit(
    name,
    cfg,
    ds,
    part,
    source_pair,
    pair,
    stat,
    base_inputs,
    values,
    device,
    expected_hash,
):
    dest = pair / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "completion.json").exists():
        return json.loads((dest / "completion.json").read_text())
    remove_degree = name == "no_message_no_degree"
    runtime_gate = smoke(cfg, part, values, device, remove_degree, expected_hash)
    dump(dest / "runtime_gate.json", runtime_gate)
    n = len(part["fitting"])
    control_values = controls(values, remove_degree)
    model = runner.construct(cfg, part["initialization_seed"], device)
    assert joint.state_hash(model) == expected_hash
    optimizer, scheduler = joint.train.build_optimizer_and_scheduler(model, cfg)
    history, best, best_epoch = ([], float("inf"), -1)
    start = time.perf_counter()
    for epoch in range(120):
        model.train()
        sums, count = (np.zeros(3), 0)
        for batch in DataLoader(
            control_values[:n],
            batch_size=16,
            sampler=stat["permutations"][epoch].tolist(),
            num_workers=0,
            pin_memory=device.type == "cuda",
        ):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pn, pg = model(batch)
            nl = model.loss_function(pn, batch.pn_target)
            gl = model.loss_function(pg, batch.pg_target)
            loss = 0.5 * nl + 0.5 * gl
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), nl.item(), gl.item()]) * batch.num_graphs
            count += batch.num_graphs
        stopping = joint.evaluate(model, control_values[n:], False, base_inputs)
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
        dump(dest / "history.json", history)
        print(
            json.dumps(
                dict(model=name, repeat=part["repeat"], fold=part["fold"], **row)
            ),
            flush=True,
        )
        if epoch - best_epoch >= 50:
            break
    model.load_state_dict(
        torch.load(dest / "best.pt", weights_only=False, map_location=device)[
            "model_state_dict"
        ],
        strict=True,
    )
    fixed_hash = joint.sha(dest / "best.pt")
    endpoints = [
        export(model, ds, part, pair, stat, subset, name)
        for subset in ["stopping", "evaluation"]
    ]
    assert joint.sha(dest / "best.pt") == fixed_hash
    result = dict(
        status="complete",
        model=name,
        repeat=part["repeat"],
        fold=part["fold"],
        device=str(device),
        reference_device=str(device),
        same_device_as_reference=True,
        actual_epochs=len(history),
        selected_epoch_zero_based=best_epoch,
        stop_reason="budget cap" if len(history) == 120 else "stopping patience50",
        elapsed_seconds=time.perf_counter() - start,
        checkpoint_sha256=fixed_hash,
        initial_state_sha256=expected_hash,
        preprocessing_sha256=joint.sha(source_pair / "fitted_preprocessing.pkl"),
        permutations_sha256=joint.sha(source_pair / "fitting_statistics.npz"),
        runtime_gate=runtime_gate,
        endpoints=endpoints,
        outer_evaluation_after_checkpoint_frozen=True,
    )
    dump(dest / "completion.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(35500 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    part = runner.load_partition(a.repeat, a.fold)
    source_pair = FROZEN / "gnn" / f"repeat{a.repeat}_fold{a.fold}"
    pair = STUDY / "topology" / f"repeat{a.repeat}_fold{a.fold}"
    prep = json.loads((source_pair / "preparation.json").read_text())
    cfg = prep["original_source_config"]
    reference = json.loads((source_pair / "baseline/completion.json").read_text())
    device = torch.device(reference["device"])
    with (source_pair / "fitted_preprocessing.pkl").open("rb") as s:
        ds = pickle.load(s)
    assert ds.graph_attr_source == "size_bucket_onehot"
    indices = []
    for block in ds.use_x_blocks:
        sl = joint.dataset.X_COL_LAYOUT[block]
        indices.extend(range(sl.start, sl.stop))
    assert (
        len(indices) == 32 and indices[19] == joint.dataset.X_COL_LAYOUT["degree"].start
    )
    ds._raw = None
    ds._load()
    assert len(ds._raw) == 940
    assert (
        ds.splits.train == part["fitting"]
        and ds.splits.val == part["stopping"]
        and (ds.splits.test == part["evaluation"])
    )
    ids = np.r_[part["fitting"], part["stopping"]]
    graphs = [ds._transform(ds._raw[int(i)]) for i in ids]
    raw = np.array(
        [
            [
                ds._pergraph_target_raw_by_source(ds._raw[int(i)], c + "_pergraph_max")
                for c in CASES
            ]
            for i in ids
        ]
    )
    assert np.array_equal(
        raw, np.load(FROZEN / "regressors/raw_truth940.npz")["raw_truth"][ids]
    )
    buckets = np.array([int(g.size_bucket[0]) for g in graphs])
    stat = np.load(source_pair / "fitting_statistics.npz")
    assert np.array_equal(stat["dataset_idx"], ids)
    inputs = (
        cfg,
        ds,
        ids,
        graphs,
        raw,
        buckets,
        stat["mean_by_size"],
        stat["sd_by_size"],
        stat["checkpoint_variance_by_size"],
        stat["grain_case_sd"],
        replay.components_for(ds),
    )
    values = joint.build_graphs(graphs, raw, buckets, inputs[6], inputs[7], False)
    dump(
        pair / "preparation.json",
        dict(
            status="frozen",
            partition=part,
            original_preparation_sha256=joint.sha(source_pair / "preparation.json"),
            source_reference_checkpoint_sha256=reference["checkpoint_sha256"],
            device=str(device),
            fitted_preprocessing_reused=True,
            graph_attr_source=ds.graph_attr_source,
            features=ds.use_x_blocks,
            degree_selected_column=19,
            topology_features_removed="all physical edges, and degree input in no_message_no_degree; geometry is retained",
            arms=ARMS,
            epochs_cap=120,
            patience=50,
            criterion="equal-weight20 cell raw max MSE normalized by fitting variance",
            batch_size=16,
            code_sha256=joint.sha(__file__),
        ),
    )
    results = [
        fit(
            name,
            cfg,
            ds,
            part,
            source_pair,
            pair,
            stat,
            inputs,
            values,
            device,
            prep["full_initial_state_sha256"],
        )
        for name in ARMS
    ]
    dump(
        pair / "completion.json",
        dict(
            status="complete",
            repeat=a.repeat,
            fold=a.fold,
            device=str(device),
            results=results,
        ),
    )
    torch.distributed.destroy_process_group()
    print("PAIR_COMPLETE", a.repeat, a.fold, flush=True)


if __name__ == "__main__":
    main()
