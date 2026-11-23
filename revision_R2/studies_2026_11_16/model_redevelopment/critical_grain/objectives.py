"""Parameter-free, equal-SVE raw-grain objective primitives.

This module makes no scientific coefficient, initialization or stopping choice.
All four cases must have complete finite peak labels. Source auxiliary missing
labels belong to the unchanged legacy loss/export contract, not these losses.
"""
import torch


def top_four_weights(target):
    """Fractional cutoff ties, exactly min(4,n) total weight per case.

    Every label above the fourth-largest cutoff gets weight one. All labels equal
    to the cutoff share the remaining slots. This keeps targets and gradients
    equivariant to a relabeling of grains, including ties and n < 4.
    """
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("Expected a nonempty n-grain by four-case target")
    if not torch.isfinite(target).all():
        raise ValueError("Missing/nonfinite raw peak label")
    k = min(4, target.shape[0])
    cutoff = torch.topk(target, k, dim=0).values[-1]
    above = target > cutoff
    tied = target == cutoff
    remaining = k - above.sum(0)
    weights = above.to(target.dtype) + tied.to(target.dtype) * (
        remaining.to(target.dtype) / tied.sum(0).to(target.dtype)
    )
    return weights.detach()


def grain_objective_cells(
    prediction, target, membership, buckets, grain_variance, maximum_variance
):
    """Return [SVE,case] losses before any optional size balancing.

    The all-grain and true-top-four terms average within each SVE/case. The maximum
    term selects the maximum prediction over *all* grains, even a false winner
    outside the true top four. torch.amax shares the derivative among exact ties.
    All-grain/top-four use equal-SVE fitting-grain mixture population variance;
    maximum uses fitting-size/case maximum sample variance.
    """
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 4:
        raise ValueError("Peak prediction/target shape mismatch")
    if not torch.isfinite(target).all():
        raise ValueError("Missing/nonfinite raw peak label")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite raw grain prediction")
    if membership.ndim != 1 or len(membership) != len(target):
        raise ValueError("Node membership shape mismatch")
    if membership.dtype != torch.long or buckets.dtype != torch.long:
        raise ValueError("Membership and size buckets must be int64")
    if buckets.ndim != 1 or len(buckets) == 0:
        raise ValueError("Nonempty one-dimensional size buckets required")
    for variance in (grain_variance, maximum_variance):
        if (
            variance.shape != (5, 4)
            or not torch.isfinite(variance).all()
            or not (variance > 0).all()
        ):
            raise ValueError("Invalid fitting-size/case variance")
    if not ((buckets >= 0) & (buckets < 5)).all():
        raise ValueError("Invalid size bucket")
    if not ((membership >= 0) & (membership < len(buckets))).all():
        raise ValueError("Invalid graph membership")
    allgrain, topfour, maximum, max_prediction = [], [], [], []
    for graph, bucket in enumerate(buckets):
        selected = membership == graph
        if not selected.any():
            raise ValueError("Empty SVE")
        p, y = prediction[selected], target[selected]
        squared = (p - y).square() / grain_variance[bucket]
        weights = top_four_weights(y)
        allgrain.append(squared.mean(0))
        topfour.append((squared * weights).sum(0) / weights.sum(0))
        maxp = p.amax(dim=0)
        max_prediction.append(maxp)
        maximum.append((maxp - y.amax(dim=0)).square() / maximum_variance[bucket])
    return dict(
        allgrain=torch.stack(allgrain),
        topfour=torch.stack(topfour),
        maximum=torch.stack(maximum),
        max_prediction=torch.stack(max_prediction),
    )


def reduce_cells(cells, buckets, size_weights=None):
    """Average SVE/case cells; optional predeclared fitting size weights."""
    if cells.ndim != 2 or cells.shape != (len(buckets), 4):
        raise ValueError("Expected SVE by four-case cells")
    per_sve = cells.mean(1)
    if size_weights is not None:
        if (
            size_weights.shape != (5,)
            or not torch.isfinite(size_weights).all()
            or not (size_weights > 0).all()
        ):
            raise ValueError("Invalid size weights")
        per_sve = per_sve * size_weights[buckets]
    return per_sve.mean()
