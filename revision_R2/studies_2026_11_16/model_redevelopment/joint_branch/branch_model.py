"""Matched raw maximum readouts; the core model and joint outputs are untouched."""
from contextlib import contextmanager
import types

import numpy as np
import torch
from torch import nn


@contextmanager
def isolated_torch_rng(seed):
    """Initialize new modules without consuming the core's CPU/CUDA streams."""
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


class JointRawBranch(nn.Module):
    def __init__(
        self, core, arm, mean, variance, count_mean, count_sd, cell_weights, head_seed
    ):
        super().__init__()
        if arm not in ("detached", "attached"):
            raise ValueError(arm)
        self.core, self.arm = core, arm
        for name, value in (
            ("fit_mean", mean),
            ("fit_variance", variance),
            ("count_mean", count_mean),
            ("count_sd", count_sd),
            ("cell_weights", cell_weights),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        if self.fit_mean.shape != (5, 4) or self.fit_variance.shape != (5, 4):
            raise ValueError("Expected five sizes by four cases")
        if (
            not torch.isfinite(self.fit_variance).all()
            or not (self.fit_variance > 0).all()
        ):
            raise ValueError("Invalid fitting variance")
        if not torch.isfinite(self.count_sd) or not self.count_sd > 0:
            raise ValueError("Invalid fitting log-count SD")
        with isolated_torch_rng(head_seed):
            self.scoring_head = nn.Sequential(
                nn.Linear(257, 64), nn.ELU(), nn.Linear(64, 4)
            )
            nn.init.zeros_(self.scoring_head[-1].weight)
            nn.init.zeros_(self.scoring_head[-1].bias)
        assert sum(p.numel() for p in core.parameters()) == 377974
        assert sum(p.numel() for p in self.scoring_head.parameters()) == 16772
        self._pooled = None
        original_pool = core._pool_graph_features

        def capture_pool(core_instance, node_features, graph_indices):
            pooled = original_pool(node_features, graph_indices)
            self._pooled = pooled
            return pooled

        # Construct fresh wrappers for replay/clones. Do not deepcopy this closure.
        core._pool_graph_features = types.MethodType(capture_pool, core)

    def forward(self, batch):
        self._pooled = None
        pn, pg = self.core(batch)
        pooled = self._pooled
        self._pooled = None
        if pooled is None or pooled.shape != (batch.num_graphs, 256):
            raise RuntimeError("Did not capture the original latent MAX pooling")
        if self.arm == "detached":
            pooled = pooled.detach()
        counts = torch.bincount(batch.batch.long(), minlength=batch.num_graphs)
        if not (counts > 0).all():
            raise ValueError("Empty SVE")
        count_z = (counts.double().log() - self.count_mean) / self.count_sd
        z = self.scoring_head(
            torch.cat((pooled, count_z[:, None].to(pooled.dtype)), dim=1)
        )
        bucket = batch.size_bucket.flatten().long()
        raw = self.fit_mean[bucket] + self.fit_variance[bucket].sqrt() * z.double()
        if pn.shape[1] != 20 or pg.shape[1] != 18:
            raise RuntimeError("Original output contract changed")
        if not all(torch.isfinite(v).all() for v in (pn, pg, raw)):
            raise FloatingPointError(
                "Nonfinite model outputs; no automatic mean fallback"
            )
        return pn, pg, raw

    def losses(self, outputs, batch):
        pn, pg, raw = outputs
        node = self.core.loss_function(pn, batch.pn_target)
        graph = self.core.loss_function(pg, batch.pg_target)
        joint = 0.5 * node + 0.5 * graph
        b = batch.size_bucket.flatten().long()
        maximum = (
            ((raw - batch.maximum_raw).square() / self.fit_variance[b]).mean(1)
            * self.cell_weights[b]
        ).mean()
        total = joint + 2.0 * maximum
        if not all(torch.isfinite(v) for v in (node, graph, joint, maximum, total)):
            raise FloatingPointError("Nonfinite training loss")
        return total, joint, node, graph, maximum


def balanced_score(truth, prediction, buckets, variance):
    if truth.shape != prediction.shape or truth.shape[1] != 4:
        raise ValueError("Endpoint shape mismatch")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Invalid stopping checkpoint")
    cells = np.stack(
        [
            ((prediction[buckets == b] - truth[buckets == b]) ** 2).mean(0)
            / variance[b]
            for b in range(5)
        ]
    )
    if not np.isfinite(cells).all():
        raise FloatingPointError("All twenty stopping cells are required")
    return float(cells.mean()), cells
