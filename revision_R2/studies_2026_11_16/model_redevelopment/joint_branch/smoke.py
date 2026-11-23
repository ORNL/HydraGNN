"""Technical tests on independently constructed disposable models only."""
from pathlib import Path
import gc
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

import study as s


def copy_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fresh(inp, arm, device, state=None):
    model = s.construct(inp, arm, device)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def gradients(model):
    return {
        n: None if p.grad is None else p.grad.detach().cpu().clone()
        for n, p in model.named_parameters()
    }


def reset_rng(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_smoke(inp, device):
    part = inp["part"]
    dest = (
        s.HERE
        / "technical_smoke"
        / f"repeat{part['repeat']}_fold{part['fold']}"
        / str(device)
    )
    dest.mkdir(parents=True, exist_ok=True)
    positions = np.array(
        [np.flatnonzero(inp["buckets"][: inp["n"]] == b)[0] for b in range(5)]
    )
    graphs = [inp["values"][i] for i in positions]
    batch = next(
        iter(DataLoader(graphs, batch_size=5, shuffle=False, num_workers=0))
    ).to(device)
    assert set(batch.sve_id.cpu().tolist()) <= set(part["fitting"])
    report = dict(
        status="running",
        scientific_training=False,
        device=str(device),
        repeat=part["repeat"],
        fold=part["fold"],
        fixture_ids=batch.sve_id.cpu().tolist(),
        fixture_nodes=batch.num_nodes,
        only_fitting_fixtures=True,
        independently_constructed_models=True,
        no_deepcopy_of_pool_closure=True,
    )
    s.dump(dest / "verification.json", report)

    # Initial functions/state and RNG are checked before any technical update.
    d = fresh(inp, "detached", device)
    a = fresh(inp, "attached", device)
    base = s.runner.construct(inp["cfg"], part["initialization_seed"], device)
    base2 = s.runner.construct(inp["cfg"], part["initialization_seed"], device)
    expected = inp["prep"]["full_initial_state_sha256"]
    assert (
        s.joint.state_hash(d.core)
        == s.joint.state_hash(a.core)
        == s.joint.state_hash(base)
        == expected
    )
    assert s.joint.state_hash(d) == s.joint.state_hash(a)
    d.eval()
    a.eval()
    base.eval()
    base2.eval()
    with torch.no_grad():
        reset_rng(2026111609)
        pn0, pg0 = base(batch)
        reset_rng(2026111609)
        pn02, pg02 = base2(batch)
        reset_rng(2026111609)
        pnd, pgd, rd = d(batch)
        reset_rng(2026111609)
        pna, pga, ra = a(batch)
    pairs = ((pn0, pnd), (pg0, pgd), (pnd, pna), (pgd, pga), (rd, ra))
    source_pairs = ((pn0, pn02), (pg0, pg02))
    exact_initial = all(torch.equal(x, y) for x, y in pairs)
    source_exact = all(torch.equal(x, y) for x, y in source_pairs)
    report.update(
        initial_function_max_abs_errors=[float((x - y).abs().max()) for x, y in pairs],
        identical_unwrapped_source_max_abs_errors=[
            float((x - y).abs().max()) for x, y in source_pairs
        ],
        initial_function_exact=exact_initial,
        identical_unwrapped_source_exact=source_exact,
        transformed_gpu_tolerance=dict(atol=2e-5, rtol=2e-5),
    )
    s.dump(dest / "verification.json", report)
    if device.type == "cpu":
        assert exact_initial and source_exact
    else:
        assert all(
            torch.allclose(x, y, atol=2e-5, rtol=2e-5) for x, y in pairs + source_pairs
        )
    assert torch.equal(rd, d.fit_mean[batch.size_bucket.flatten().long()])
    assert s.joint.state_hash(base) == s.joint.state_hash(d.core) == expected
    report.update(
        initial_original_function_matches=True,
        initial_paired_function_matches=True,
        initial_raw_head_equals_fitting_mean=True,
        eval_does_not_change_core_state=True,
        core_full_state_sha256=expected,
        paired_full_state_sha256=s.joint.state_hash(d),
        parameters=dict(core=377974, raw_head=16772, total=394746),
        branch_initialization_preserves_cpu_cuda_rng=True,
    )

    # A single head-only technical update makes its encoder derivative testable.
    # The zero last layer correctly yields zero extra encoder gradient at step0.
    d.zero_grad(set_to_none=True)
    initial_lmax = d.losses(d(batch), batch)[-1]
    initial_lmax.backward()
    assert all(p.grad is None for p in d.core.parameters())
    head_opt = torch.optim.SGD(d.scoring_head.parameters(), lr=0.01)
    head_opt.step()
    head_updated_state = copy_state(d)
    assert s.joint.state_hash(d.core) == expected
    del d, a, base, base2
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Fresh independent objects avoid copied hook closures. Identical stochastic
    # streams make the detached full-loss comparison exactly L0-only.
    d = fresh(inp, "detached", device, head_updated_state)
    a = fresh(inp, "attached", device, head_updated_state)
    base = s.runner.construct(inp["cfg"], part["initialization_seed"], device)
    for m in (d, a, base):
        m.train()
        m.zero_grad(set_to_none=True)
    reset_rng(2026111607)
    pn, pg = base(batch)
    original_joint = 0.5 * base.loss_function(
        pn, batch.pn_target
    ) + 0.5 * base.loss_function(pg, batch.pg_target)
    original_joint.backward()
    bg = gradients(base)
    reset_rng(2026111607)
    outd = d(batch)
    detached_losses = d.losses(outd, batch)
    detached_losses[0].backward()
    if device.type == "cpu":
        assert torch.equal(detached_losses[1], original_joint)
    else:
        assert torch.allclose(detached_losses[1], original_joint, atol=2e-5, rtol=2e-5)
    dg = gradients(d.core)
    difference_sq = base_sq = 0.0
    gradient_max_abs = 0.0
    exact_gradients = True
    for name in bg:
        assert (bg[name] is None) == (dg[name] is None), name
        if bg[name] is not None:
            exact_gradients &= torch.equal(bg[name], dg[name])
            difference_sq += float(
                (bg[name].double() - dg[name].double()).square().sum()
            )
            base_sq += float(bg[name].double().square().sum())
            gradient_max_abs = max(
                gradient_max_abs, float((bg[name] - dg[name]).abs().max())
            )
            if device.type == "cpu":
                assert torch.equal(bg[name], dg[name]), name
    gradient_relative_l2 = (difference_sq / max(base_sq, 1e-30)) ** 0.5
    assert gradient_relative_l2 < 2e-5
    reset_rng(2026111607)
    outa = a(batch)
    attached_losses = a.losses(outa, batch)
    attached_losses[0].backward()
    ag = gradients(a.core)
    if device.type == "cpu":
        assert all(torch.equal(x, y) for x, y in zip(outd, outa))
    else:
        assert all(
            torch.allclose(x, y, atol=2e-5, rtol=2e-5) for x, y in zip(outd, outa)
        )
    changed_encoder = [
        name
        for name in bg
        if bg[name] is not None
        and (
            name.startswith("graph_convs") or name.startswith("graph_concat_projector")
        )
        and not torch.equal(bg[name], ag[name])
    ]
    assert changed_encoder
    assert all(torch.isfinite(g).all() for g in ag.values() if g is not None)
    exact_bn = (
        s.joint.state_hash(base)
        == s.joint.state_hash(d.core)
        == s.joint.state_hash(a.core)
    )
    if device.type == "cpu":
        assert exact_bn
    else:
        assert all(
            torch.allclose(x, y, atol=2e-5, rtol=2e-5)
            for x, y in zip(base.state_dict().values(), d.core.state_dict().values())
        )
    optimizer, scheduler = s.joint.train.build_optimizer_and_scheduler(a, inp["cfg"])
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in optimized} == {id(p) for p in a.parameters()}
    assert sum(p.numel() for p in optimized) == 394746
    shape_grads, mechanical_grads = [], []
    for name, module in base.named_modules():
        if isinstance(module, nn.Linear) and module.weight.grad is not None:
            if module.out_features == 20 and name.startswith("heads_NN.0"):
                shape_grads.append(float(module.weight.grad[8:16].abs().sum()))
            if module.out_features == 18 and name.startswith("heads_NN.1"):
                mechanical_grads.append(float(module.weight.grad[8:14].abs().sum()))
    assert shape_grads and max(shape_grads) > 0
    assert mechanical_grads and max(mechanical_grads) > 0
    report.update(
        detached_core_gradients_exact_L0=exact_gradients,
        detached_gradient_max_absolute_difference=gradient_max_abs,
        detached_gradient_relative_l2_difference=gradient_relative_l2,
        detached_gradient_relative_l2_tolerance=2e-5,
        same_train_mode_core_outputs=True,
        attached_changes_encoder_gradients=True,
        changed_encoder_parameter_names=changed_encoder,
        original_shape_output_gradients=shape_grads,
        original_mechanical_output_gradients=mechanical_grads,
        optimizer_parameter_count=sum(p.numel() for p in optimized),
        optimizer_complete=True,
        finite_original_and_attached_gradients=True,
        matched_BatchNorm_updates=True,
        bitwise_equal_BatchNorm_updates=exact_bn,
        head_update_is_technical_only=True,
    )
    del base, d
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Evaluation invariance to within-SVE node relabeling; no targets select nodes.
    a.eval()
    with torch.no_grad():
        before = a(batch)
    permuted = []
    rng = np.random.default_rng(2026111608)
    node_order = []
    offset = 0
    for g in graphs:
        perm = rng.permutation(g.num_nodes)
        inv = np.argsort(perm)
        gp = g.clone()
        gp.x = gp.x[torch.as_tensor(perm)]
        if getattr(gp, "pos", None) is not None:
            gp.pos = gp.pos[torch.as_tensor(perm)]
        gp.edge_index = torch.as_tensor(inv, dtype=torch.long)[gp.edge_index]
        permuted.append(gp)
        node_order.extend((offset + perm).tolist())
        offset += g.num_nodes
    bp = next(
        iter(DataLoader(permuted, batch_size=5, shuffle=False, num_workers=0))
    ).to(device)
    with torch.no_grad():
        after = a(bp)
    node_error = float((after[0] - before[0][node_order]).abs().max())
    graph_error = float((after[1] - before[1]).abs().max())
    raw_scaled_error = float(
        (
            (after[2] - before[2])
            / a.fit_variance[batch.size_bucket.flatten().long()].sqrt()
        )
        .abs()
        .max()
    )
    assert torch.allclose(after[0], before[0][node_order], atol=2e-5, rtol=2e-5)
    assert torch.allclose(after[1], before[1], atol=2e-5, rtol=2e-5)
    assert raw_scaled_error < 2e-5

    restored = (
        s.joint.inverse_graph(
            batch.pg_target,
            batch.size_bucket.flatten().long(),
            False,
            inp["mean"],
            inp["population_sd"],
            inp["components"],
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(restored, inp["raw"][positions], rtol=2e-6, atol=1e-10)
    checkpoint = dest / "technical_state.pt"
    torch.save(a.state_dict(), checkpoint)
    clone = fresh(inp, "attached", device)
    clone.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location=device), strict=True
    )
    clone.eval()
    with torch.no_grad():
        replay = clone(batch)
    exact_replay = all(torch.equal(x, y) for x, y in zip(before, replay))
    if device.type == "cpu":
        assert exact_replay
    else:
        assert all(
            torch.allclose(x, y, atol=2e-5, rtol=2e-5) for x, y in zip(before, replay)
        )
    fixture = (
        graphs,
        inp["ids"][positions],
        inp["raw"][positions],
        inp["buckets"][positions],
    )
    export_report = s.export_subset(
        clone, inp, "technical_fitting_fixture", dest, explicit=fixture
    )
    report.update(
        status="pass",
        permutation_invariance=True,
        permutation_max_absolute_node_error=node_error,
        permutation_max_absolute_graph_error=graph_error,
        permutation_raw_head_max_error_fitting_sd=raw_scaled_error,
        legacy_target_inverse_roundtrip=True,
        strict_state_load=True,
        reloaded_model_output_bitwise_equal=exact_replay,
        export_readback=export_report,
        source_manifest=s.source_manifest(),
        no_outer_transforms_forwards_or_scores=True,
        scientific_fits_started=0,
    )
    s.dump(dest / "verification.json", report)
    print(
        "TECHNICAL_SMOKE_PASS",
        json.dumps(dict(repeat=part["repeat"], fold=part["fold"], device=str(device))),
        flush=True,
    )
    return report
