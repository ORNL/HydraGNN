"""Fitting/stopping raw-grain labels and fitting-only equal-SVE moments."""
import numpy as np
import torch


def add_raw_grain_contract(inp, cases):
    """Extend already restricted baseline inputs, never the outer evaluation set."""
    ids = inp["ids"]
    assert ids.tolist() == inp["part"]["fitting"] + inp["part"]["stopping"]
    raw = []
    source_ids = []
    for position, idx in enumerate(ids):
        source = inp["ds"]._raw[int(idx)]
        peak = np.column_stack(
            [getattr(source, c + "_fip_dist")[:, 0].numpy() for c in cases]
        ).astype(np.float64)
        if peak.shape != (source.num_nodes, 4) or not np.isfinite(peak).all():
            raise ValueError(
                f"Incomplete raw grain peaks for fitting/stopping SVE {idx}"
            )
        np.testing.assert_array_equal(peak.max(0), inp["raw"][position])
        if len(np.unique(source.vtk_grain_ids.numpy())) != source.num_nodes:
            raise ValueError(f"Nonunique source grain IDs in SVE {idx}")
        raw.append(peak)
        source_ids.append(source.vtk_grain_ids.numpy().astype(np.int64).reshape(-1))
        inp["values"][position].raw_grain_target = torch.from_numpy(peak)
        inp["values"][position].grain_row_id = torch.arange(
            source.num_nodes, dtype=torch.long
        )
        inp["values"][position].grain_source_id = torch.from_numpy(source_ids[-1])
    n = inp["n"]
    graph_means = np.stack([y.mean(0) for y in raw[:n]])
    mean = np.stack([graph_means[inp["buckets"][:n] == b].mean(0) for b in range(5)])
    # Stable centered second moment: mixture draws an SVE uniformly then a grain
    # uniformly within that SVE. It is a population variance, with no ddof1.
    variance = np.stack(
        [
            np.stack(
                [
                    ((raw[i] - mean[b]) ** 2).mean(0)
                    for i in range(n)
                    if inp["buckets"][i] == b
                ]
            ).mean(0)
            for b in range(5)
        ]
    )
    pooled_mean = graph_means.mean(0)
    pooled_variance = np.stack(
        [((y - pooled_mean) ** 2).mean(0) for y in raw[:n]]
    ).mean(0)
    floor = np.maximum(1e-6 * pooled_variance, 1e-30)
    protected = np.maximum(variance, floor[None, :])
    # Keep a declared protection but never silently alter this fixed-cohort study.
    assert np.array_equal(
        protected, variance
    ), "Grain variance floor activated; protocol review required"
    assert (
        np.isfinite(mean).all() and np.isfinite(variance).all() and (variance > 0).all()
    )
    inp.update(
        raw_grain=raw,
        grain_source_ids=source_ids,
        grain_mean=mean,
        grain_variance=variance,
        grain_variance_floor=floor,
        grain_pooled_variance=pooled_variance,
    )
    return inp
