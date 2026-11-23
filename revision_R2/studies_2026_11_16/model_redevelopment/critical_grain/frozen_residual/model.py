"""Head-only residual grain predictors on a permanently frozen feature cache."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch import nn
from grain_head import RawGrainHead
from objectives import grain_objective_cells, reduce_cells


class FrozenResidualHead(nn.Module):
    def __init__(self, grain_variance, maximum_variance, size_weights, seed):
        super().__init__()
        for name, value in (
            ("grain_variance", grain_variance),
            ("maximum_variance", maximum_variance),
            ("size_weights", size_weights),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        assert self.grain_variance.shape == self.maximum_variance.shape == (5, 4)
        assert self.size_weights.shape == (5,)
        assert all(
            torch.isfinite(v).all() and (v > 0).all()
            for v in (self.grain_variance, self.maximum_variance, self.size_weights)
        )
        self.head = RawGrainHead(seed, 64, True)
        assert sum(p.numel() for p in self.parameters()) == 16708

    def forward(self, features, original_prediction, node_buckets):
        if features.requires_grad or original_prediction.requires_grad:
            raise ValueError(
                "Original features/predictions must be detached cache tensors"
            )
        residual = self.head(features)
        prediction = (
            original_prediction
            + self.grain_variance[node_buckets].sqrt() * residual.double()
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                "Nonfinite new prediction; identity is not a fallback"
            )
        return prediction

    def cells(self, prediction, batch):
        return grain_objective_cells(
            prediction,
            batch["target"],
            batch["membership"],
            batch["buckets"],
            self.grain_variance,
            self.maximum_variance,
        )

    def losses(self, prediction, batch, treatment):
        cells = self.cells(prediction, batch)
        terms = {
            key: reduce_cells(cells[key], batch["buckets"], self.size_weights)
            for key in ("allgrain", "topfour", "maximum")
        }
        total = terms["allgrain"]
        if treatment:
            total = total + 0.5 * terms["topfour"] + 0.5 * terms["maximum"]
        if not all(torch.isfinite(v) for v in (total, *terms.values())):
            raise FloatingPointError("Nonfinite new loss; identity is not a fallback")
        return dict(total=total, **terms)
