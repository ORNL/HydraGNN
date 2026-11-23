"""Reusable four-case grain MLP, independent of encoder training policy."""
import torch
from torch import nn


class RawGrainHead(nn.Sequential):
    def __init__(self, seed, hidden_width, zero_final):
        if not isinstance(hidden_width, int) or hidden_width <= 0:
            raise ValueError("Positive grain hidden width required")
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            super().__init__(
                nn.Linear(256, hidden_width), nn.ELU(), nn.Linear(hidden_width, 4)
            )
            if zero_final:
                nn.init.zeros_(self[-1].weight)
                nn.init.zeros_(self[-1].bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 256:
            raise ValueError("Expected final per-grain h256")
        return super().forward(features)


def decode_raw_grain(standardized, node_buckets, mean, variance):
    """Affine decoder shared by live and frozen-feature candidate paths."""
    raw = mean[node_buckets] + variance[node_buckets].sqrt() * standardized.double()
    if not torch.isfinite(raw).all():
        raise FloatingPointError("Nonfinite raw grain output; no fallback")
    return raw
