"""Four full paper joint models: maximum target transform x physical FCC fitting augmentation."""
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
import copy
import datetime
import hashlib
import json
import pickle
import shutil
import sys
import time
import types
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = Path(str(release_root() / ""))
CONTROL = ROOT / "examples/fip_revision_restart_20261005/controlled_gnn"
FCC = CONTROL.parent / "fcc_node_invariance"
WINDOWS = Path(
    str(
        release_root()
        / "revision_restart_2026_10_05/new_studies/joint_normalization_fcc"
    )
)
sys.path[:0] = [str(CONTROL / "code"), str(FCC), str(ROOT)]
import dataset
import train
import losses
from fcc_orientation import equivalent_features, transformed_equivalent

CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]
SIZES = [125, 177, 250, 354, 500]
ARMS = ["neither", "normalization", "fcc", "both"]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for (name, value) in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sync():
    for path in HERE.rglob("*"):
        if path.is_file() and path.suffix in [
            ".py",
            ".json",
            ".csv",
            ".npz",
            ".pt",
            ".png",
            ".pdf",
            ".log",
        ]:
            dest = WINDOWS / path.relative_to(HERE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def load_inputs():
    with (CONTROL / "results/sage_sorted_v2_seed101/fitted_dataset_state.pkl").open(
        "rb"
    ) as stream:
        ds = pickle.load(stream)
    cfg = json.loads(
        (CONTROL / "results/sage_sorted_v2_seed101/config.json").read_text()
    )
    assert [len(ds.splits.train), len(ds.splits.val), len(ds.splits.test)] == [
        846,
        47,
        47,
    ]
    ids = np.r_[ds.splits.train, ds.splits.val]
    graphs = ds._processed[:893]
    assert len(graphs) == 893
    assert [spec["source"] for spec in ds.pergraph_targets[:4]] == [
        case + "_pergraph_max" for case in CASES
    ]
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
    buckets = np.array([int(g.size_bucket[0]) for g in graphs])
    assert np.isfinite(raw).all()
    fitting = raw[:846]
    mean = np.stack([fitting[buckets[:846] == b].mean(0) for b in range(5)])
    sd = np.stack([fitting[buckets[:846] == b].std(0, ddof=0) for b in range(5)])
    assert np.all(sd > 0)
    variance = np.stack([fitting[buckets[:846] == b].var(0, ddof=1) for b in range(5)])
    assert np.all(variance > 0)
    all_grains = np.concatenate(
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
    grain_sd = all_grains.std(0, ddof=0)
    assert np.isfinite(all_grains).all() and np.all(grain_sd > 0)
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
    return (
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


def inverse_graph(output, bucket, normalized, mean, sd, components):
    output = output.double()
    if normalized:
        return (
            output[:, :4] * torch.as_tensor(sd, device=output.device)[bucket]
            + torch.as_tensor(mean, device=output.device)[bucket]
        )
    answer = []
    for (c, p) in enumerate(components):
        z = (output[:, c] * p["outer_sd"] + p["outer_mean"]).clamp(
            p["clip_low"], p["clip_high"]
        )
        u = z * p["inner_sd"] + p["inner_mean"]
        positive = u >= 0
        v = torch.empty_like(u)
        lam = p["lam"]
        v[positive] = (
            torch.expm1(u[positive])
            if abs(lam) < 1e-12
            else torch.expm1(torch.log1p(lam * u[positive]) / lam)
        )
        q = 2 - lam
        v[~positive] = (
            -torch.expm1(-u[~positive])
            if abs(q) < 1e-12
            else -torch.expm1(torch.log1p(-q * u[~positive]) / q)
        )
        answer.append(
            torch.where(torch.isfinite(v), v, torch.full_like(v, p["fallback"]))
        )
    return torch.stack(answer, 1)


def build_graphs(graphs, raw, buckets, mean, sd, normalized):
    answer = []
    for (i, original) in enumerate(graphs):
        g = original.clone()
        node_end = int(g.y_loc[0, 1])
        g.pn_target = g.y[:node_end].reshape(-1, 20).clone()
        g.pg_target = g.y[node_end:].reshape(1, 18).clone()
        if normalized:
            g.pg_target[0, :4] = torch.from_numpy(
                ((raw[i] - mean[buckets[i]]) / sd[buckets[i]]).astype(np.float32)
            )
            g.y[node_end : node_end + 4] = g.pg_target[0, :4]
        assert torch.equal(g.pn_target.flatten(), original.y[:node_end])
        assert torch.equal(g.pg_target[0, 4:], original.y[node_end + 4 :])
        g.maximum_raw = torch.tensor(raw[i : i + 1], dtype=torch.float64)
        g.study_position = torch.tensor([i], dtype=torch.long)
        answer.append(g)
    return answer


class FittingDataset(Dataset):
    def __init__(self, ds, graphs, enabled):
        (self.ds, self.graphs, self.enabled) = (ds, graphs, enabled)
        self.rng = np.random.default_rng(2026100517)
        self.accesses = 0
        self.max_schmid_error = 0.0

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        original = self.graphs[index]
        if not self.enabled:
            return original
        raw = self.ds._raw[int(self.ds.splits.train[index])]
        operators = self.rng.integers(0, 24, size=raw.num_nodes, dtype=np.uint8)
        (q, schmid, error) = equivalent_features(raw, operators)
        selected = self.ds._select_x(raw.x).numpy().copy()
        (selected[:, :4], selected[:, 4:16]) = (q, schmid)
        out = original.clone()
        out.x = torch.from_numpy(
            self.ds._scaler_x.transform(selected).astype(np.float32)
        )
        assert torch.equal(out.x[:, 4:], original.x[:, 4:])
        self.accesses += 1
        self.max_schmid_error = max(self.max_schmid_error, error)
        return out


def construct(cfg, device):
    torch.manual_seed(101)
    np.random.seed(101)
    factory = types.ModuleType("hydragnn.models.joint_factorial_factory")
    factory.__package__ = "hydragnn.models"
    path = CONTROL / "create_seed_controlled.py"
    exec(compile(path.read_text(), str(path), "exec"), factory.__dict__)
    model = factory.create_model_config(
        config=cfg["NeuralNetwork"], verbosity=0, use_gpu=False
    )
    assert train._install_residual_node_head(model) == 1
    model._ensure_graph_concat_projector(5, 256, torch.device("cpu"), torch.float32)
    losses.override_model_loss(model, cfg["Recipe"]["loss_recipe"])
    assert sum((p.numel() for p in model.parameters())) == 377974
    return model.to(device)


def selection_score(truth, pred, buckets, variance):
    values = np.stack(
        [
            ((pred[buckets == b] - truth[buckets == b]) ** 2).mean(0) / variance[b]
            for b in range(5)
        ]
    )
    return (float(values.mean()), values)


def evaluate(model, graphs, normalized, inputs, grain=False):
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
    model.eval()
    (predictions, positions, node_errors) = ([], [], [])
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False, num_workers=0):
            batch = batch.to(device)
            (pn, pg) = model(batch)
            bucket = batch.size_bucket.flatten().long()
            predictions.append(
                inverse_graph(pg, bucket, normalized, mean, sd, components)
                .cpu()
                .numpy()
            )
            batch_positions = batch.study_position.cpu().tolist()
            positions.extend(batch_positions)
            if grain:
                prediction = pn[:, :4].double().cpu().numpy()
                truth = np.concatenate(
                    [
                        np.column_stack(
                            [
                                getattr(
                                    ds._raw[int(ids[position])], case + "_fip_dist"
                                )[:, 0].numpy()
                                for case in CASES
                            ]
                        )
                        for position in batch_positions
                    ]
                ).astype(np.float64)
                columns = []
                for c in range(4):
                    inv = (
                        ds.inverse_pergrain_target
                        if c == 0
                        else lambda value, c=c: ds.inverse_per_grain_aux_target(
                            c - 1, value
                        )
                    )
                    columns.append(inv(prediction[:, c]) - truth[:, c])
                node_errors.append(np.column_stack(columns))
    positions = np.array(positions)
    prediction = np.concatenate(predictions)
    (score, cells) = selection_score(
        raw[positions], prediction, buckets[positions], variance
    )
    result = dict(score=score, cells=cells, pred=prediction, positions=positions)
    if grain:
        error = np.concatenate(node_errors)
        rmse = np.sqrt((error ** 2).mean(0))
        result["grain_quality"] = dict(
            target="grain peak FIP",
            observations="pooled grains across evaluated SVEs; not within-grain voxel accuracy",
            n_grains=len(error),
            raw_rmse=rmse.tolist(),
            fitting_case_sd=grain_sd.tolist(),
            normalized_rmse=(rmse / grain_sd).tolist(),
        )
    return result


def preflight(inputs, permutations):
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
    (init, smoke) = ([], [])
    for arm in ARMS:
        model = construct(cfg, torch.device("cpu"))
        init.append(state_hash(model))
        assert [float(w) for w in model.loss_weights] == [0.5, 0.5]
        (normalized, augmented) = (
            arm in ["normalization", "both"],
            arm in ["fcc", "both"],
        )
        values = build_graphs(graphs[:2], raw[:2], buckets[:2], mean, sd, normalized)
        fitting = FittingDataset(ds, values, augmented)
        batch = next(
            iter(DataLoader(fitting, batch_size=2, shuffle=False, num_workers=0))
        )
        model.train()
        (optimizer, _) = train.build_optimizer_and_scheduler(model, cfg)
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
            id(p) for p in model.parameters()
        }
        (pn, pg) = model(batch)
        joint = 0.5 * model.loss_function(
            pn, batch.pn_target
        ) + 0.5 * model.loss_function(pg, batch.pg_target)
        from hydragnn.train.train_validate_test import get_head_indices

        (reference, _) = model.loss(
            [pn, pg],
            batch.y,
            get_head_indices(types.SimpleNamespace(module=model), batch),
        )
        assert torch.equal(joint, reference)
        before_parameters = {
            name: value.detach().clone() for (name, value) in model.named_parameters()
        }
        joint.backward()
        assert all(
            (
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        optimizer.step()
        changed = [
            name
            for (name, p) in model.named_parameters()
            if not torch.equal(p, before_parameters[name])
        ]
        assert any((name.startswith("heads_NN.0.") for name in changed))
        assert any((name.startswith("heads_NN.1.") for name in changed))
        assert any((name.startswith("graph_concat_projector.") for name in changed))
        smoke.append(
            dict(
                arm=arm,
                source_joint_loss_exact_parity=True,
                requested_task_weight_ratio=[1.0, 1.0],
                executed_normalized_task_weights=[0.5, 0.5],
                finite_loss=float(joint.detach()),
                finite_gradients=True,
                optimizer_tracks_all377974_parameters=True,
                graph_and_node_and_context_parameters_updated=True,
                changed_parameter_tensors=len(changed),
            )
        )
    assert len(set(init)) == 1
    for normalized in [False, True]:
        values = build_graphs(graphs, raw, buckets, mean, sd, normalized)
        target = torch.stack([g.pg_target[0] for g in values])
        restored = inverse_graph(
            target, torch.tensor(buckets), normalized, mean, sd, components
        ).numpy()
        assert np.allclose(restored, raw, rtol=2e-06, atol=1e-10)
    rng = np.random.default_rng(2026100518)
    before = torch.get_rng_state().clone()
    global_np = np.random.get_state()
    max_error = 0.0
    for (index, original) in enumerate(graphs[:846]):
        source = ds._raw[int(ds.splits.train[index])]
        operators = rng.integers(0, 24, size=source.num_nodes, dtype=np.uint8)
        (q, schmid, error) = equivalent_features(source, operators)
        selected = ds._select_x(source.x).numpy().copy()
        (selected[:, :4], selected[:, 4:16]) = (q, schmid)
        fast = torch.from_numpy(ds._scaler_x.transform(selected).astype(np.float32))
        (full, _) = transformed_equivalent(ds, source, operators)
        assert torch.equal(fast, full.x)
        assert torch.equal(fast[:, 4:], original.x[:, 4:])
        max_error = max(max_error, error)
    assert torch.equal(before, torch.get_rng_state())
    after_np = np.random.get_state()
    assert (
        global_np[0] == after_np[0]
        and np.array_equal(global_np[1], after_np[1])
        and (global_np[2:] == after_np[2:])
    )
    gate = dict(
        status="pass",
        initial_full_state_sha256=init[0],
        all_four_initial_hashes=init,
        full_model_parameters=377974,
        lazy_projection_materialized_before_optimizer=True,
        fitting_validation_counts=[846, 47],
        fitting_counts_by_size=np.bincount(buckets[:846], minlength=5).tolist(),
        validation_counts_by_size=np.bincount(buckets[846:], minlength=5).tolist(),
        no_test_graphs_newly_transformed_or_forwarded=True,
        same_nodes_topology_geometry_inputs_and_nonmaximum_targets=True,
        all_target_inverse_roundtrips_pass=True,
        all846_fcc_transforms_equal_complete_source_transform=True,
        nonquaternion_scaled_features_exactly_unchanged=True,
        physical_source_schmid_max_error=max_error,
        augmentation_leaves_global_numpy_and_torch_rng_unchanged=True,
        permutations_sha256=sha(HERE / "epoch_permutations.npz"),
        shared_initialization_and_epoch_order=True,
    )
    dump(HERE / "input_gates.json", gate)
    gate["joint_loss_and_gradient_smokes"] = smoke
    dump(HERE / "input_gates.json", gate)
    return gate


def run_arm(arm, inputs, permutations, gate):
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
    dest = HERE / "results" / arm
    assert not (
        dest / "completion.json"
    ).exists(), "Complete arm must not be duplicated"
    dest.mkdir(parents=True, exist_ok=True)
    (normalized, augmented) = (arm in ["normalization", "both"], arm in ["fcc", "both"])
    values = build_graphs(graphs, raw, buckets, mean, sd, normalized)
    fit = FittingDataset(ds, values[:846], augmented)
    model = construct(cfg, torch.device("cuda:0"))
    assert state_hash(model) == gate["initial_full_state_sha256"]
    (optimizer, scheduler) = train.build_optimizer_and_scheduler(model, cfg)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
    dump(
        dest / "initialization.json",
        dict(
            state_sha256=state_hash(model),
            full_model_parameters=377974,
            optimizer_tracks_all_parameters=True,
        ),
    )
    (history, best_score, best_epoch) = ([], float("inf"), -1)
    start = time.perf_counter()
    for epoch in range(120):
        model.train()
        (loss_sum, count) = (np.zeros(3), 0)
        loader = DataLoader(
            fit,
            batch_size=16,
            sampler=permutations[epoch].tolist(),
            num_workers=0,
            pin_memory=True,
        )
        for batch in loader:
            batch = batch.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            (pn, pg) = model(batch)
            node_loss = model.loss_function(pn, batch.pn_target)
            graph_loss = model.loss_function(pg, batch.pg_target)
            total = 0.5 * node_loss + 0.5 * graph_loss
            assert torch.isfinite(total)
            total.backward()
            optimizer.step()
            number = batch.num_graphs
            loss_sum += (
                np.array([total.item(), node_loss.item(), graph_loss.item()]) * number
            )
            count += number
        val = evaluate(model, values[846:], normalized, inputs)
        score = val["score"]
        scheduler.step(score)
        improved = score < best_score
        if improved:
            (best_score, best_epoch) = (score, epoch)
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch,
                    raw_selection_score=score,
                    arm=arm,
                ),
                dest / "best.pt",
            )
        record = dict(
            epoch_zero_based=epoch,
            fitting_total_loss=float(loss_sum[0] / count),
            fitting_node_loss=float(loss_sum[1] / count),
            fitting_graph_loss=float(loss_sum[2] / count),
            raw_validation_maximum_score=score,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            best_epoch_zero_based=best_epoch,
            seconds=time.perf_counter() - start,
        )
        history.append(record)
        dump(dest / "history.json", history)
        dump(HERE / "progress.json", dict(arm=arm, **record))
        print(json.dumps(dict(arm=arm, **record)), flush=True)
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
    sealed = torch.load(dest / "best.pt", weights_only=False, map_location="cuda:0")
    model.load_state_dict(sealed["model_state_dict"], strict=True)
    for (subset, group) in [("fitting", values[:846]), ("validation", values[846:])]:
        endpoint = evaluate(model, group, normalized, inputs, grain=True)
        np.savez(
            dest / (subset + "_predictions.npz"),
            dataset_idx=ids[endpoint["positions"]],
            size_bucket=buckets[endpoint["positions"]],
            y_true=raw[endpoint["positions"]],
            y_pred=endpoint["pred"],
            cases=CASES,
        )
        dump(
            dest / (subset + "_endpoint.json"),
            dict(
                raw_maximum_selection_score=endpoint["score"],
                normalized_cell_mse=endpoint["cells"].tolist(),
                grain_quality=endpoint["grain_quality"],
            ),
        )
    dump(
        dest / "effective_config.json",
        dict(
            source_full_joint_config=cfg,
            arm=arm,
            maximum_transform="raw fitting-size affine"
            if normalized
            else "original fitting Yeo-Johnson plus outer scaler",
            fcc_fitting_augmentation=augmented,
            checkpoint_scheduler_score="mean raw MSE / fixed fitting size variance over20 cells",
            epochs_cap=120,
            patience=50,
            seed=101,
            split_seed=42,
            task_weights=[1.0, 1.0],
            no_test_forward=True,
        ),
    )
    dump(
        dest / "completion.json",
        dict(
            status="complete",
            arm=arm,
            actual_epochs=len(history),
            selected_epoch_zero_based=best_epoch,
            epochs_since_best=epoch - best_epoch,
            stop_reason="budget cap"
            if len(history) == 120
            else "validation patience50",
            elapsed_seconds=time.perf_counter() - start,
            checkpoint_sha256=sha(dest / "best.pt"),
            final_checkpoint_sha256=sha(dest / "last.pt"),
            full_model_parameters=377974,
            fcc_graph_accesses=fit.accesses,
            fcc_max_schmid_error=fit.max_schmid_error,
            test_forward=False,
        ),
    )
    sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(30500 + os.getpid() % 500),
        WORLD_SIZE="1",
        RANK="0",
    )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", init_method="env://")
    HERE.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    rng = np.random.default_rng(2026100601)
    permutations = np.stack([rng.permutation(846) for _ in range(120)])
    if (HERE / "epoch_permutations.npz").exists():
        assert np.array_equal(
            np.load(HERE / "epoch_permutations.npz")["permutations"], permutations
        )
    else:
        np.savez(HERE / "epoch_permutations.npz", permutations=permutations)
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
    np.savez(
        HERE / "fitting_statistics.npz",
        mean_by_size=mean,
        sd_by_size=sd,
        checkpoint_variance_by_size=variance,
        grain_case_sd=grain_sd,
        dataset_idx=ids,
        size_bucket=buckets,
        raw_truth=raw,
    )
    dump(
        HERE / "source_manifest.json",
        [
            {"path": str(path), "sha256": sha(path)}
            for path in [
                Path(__file__),
                FCC / "fcc_orientation.py",
                CONTROL / "create_seed_controlled.py",
                CONTROL / "code/dataset.py",
                CONTROL / "code/train.py",
                CONTROL / "code/losses.py",
                CONTROL / "results/sage_sorted_v2_seed101/config.json",
            ]
        ],
    )
    gate = preflight(inputs, permutations)
    sync()
    if args.gate_only:
        print("FOUR_ARM_PREFLIGHT_PASS", flush=True)
        torch.distributed.destroy_process_group()
        return
    for arm in [args.arm] if args.arm else ARMS:
        run_arm(arm, inputs, permutations, gate)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
