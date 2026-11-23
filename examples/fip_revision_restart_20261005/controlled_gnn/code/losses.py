"""Custom loss functions for cp3d_v2 — ported from examples/anik/train_pre.py.

The headline loss `peak_huber_topk_{α×10}_{γ×100}` was the production winner
in the anik per-grain study (peak_huber_topk_05_15: α=0.5, β=0.35, γ=0.15)
giving top-1=0.35, top-10% recall=0.55, per-graph Spearman=0.59. We port
it directly so v2 can match or exceed those results.

Available loss recipes (passed as `Recipe.loss_recipe` in the config):

  - "rank_huber_05"           : 0.5·Huber + 0.5·pairwise rank  (per-grain ranking)
  - "rank_huber_02" / "_08"   : Other α values
  - "weighted_huber"          : Huber weighted by ReLU(target z-score) — hot-spot bias
  - "pinball_085"             : Asymmetric pinball at τ=0.85
  - "peak_huber_topk_05_15"   : 0.5·Huber + 0.35·rank + 0.15·top-k Huber  (production)
  - "peak_huber_topk_05_25"   : γ=0.25 variant (more hot-spot pressure)
  - "peak_huber_topk_{α×10}"  : Balanced β=γ=(1-α)/2
  - "mse_stdmatch_LL"         : MSE + (LL/100)·within-SVE std-match
  - "mse_gumbel_LB_LM"        : MSE + Gumbel β-match + Gumbel μ-match (method-of-moments)
  - "mse_gumbel_tw_AA_KK"     : Per-grain MSE re-weighted by within-SVE Gumbel-CDF percentile

Use via `from losses import build_custom_loss`. Pair with
`install_batch_stash_hook(model)` so the rank/topk losses can read data.batch.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

GAMMA_EULER = 0.5772156649015329
SQRT6_OVER_PI = math.sqrt(6.0) / math.pi

# Module-level stash set by a forward pre-hook on the model; the per-graph
# losses read batch from here without HydraGNN needing to know.
_CURRENT_BATCH: Dict[str, Optional[torch.Tensor]] = {"batch": None}

LOSS_RANK_MARGIN = 0.1  # ranking margin (in target-scaled space)
LOSS_RANK_SUBSAMPLE = 200  # max nodes per graph in pairwise rank (memory bound)
LOSS_ALPHA_WEIGHTED = 2.0  # weighted_huber alpha
LOSS_TAU_PINBALL = 0.85
TOPK_FRACTION = 0.10
TOPK_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _pairwise_rank_term_per_graph(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    margin: float,
    subsample: int,
) -> Optional[torch.Tensor]:
    """Per-graph pairwise ranking margin loss (no Huber component).

    For each graph g, for each ordered pair (i, j) with target_j > target_i + eps,
    apply relu(margin - (pred_j - pred_i)). Subsample to bound O(n^2) memory.
    """
    device = pred.device
    if batch.device != device:
        batch = batch.to(device)

    p = pred.reshape(-1)
    t = target.reshape(-1)
    if p.numel() != batch.numel():
        return None

    rank_terms: List[torch.Tensor] = []
    for g in torch.unique(batch):
        mask = batch == g
        idxs = mask.nonzero(as_tuple=True)[0]
        n = idxs.numel()
        if n < 2:
            continue
        if n > subsample:
            perm = torch.randperm(n, device=device)[:subsample]
            idxs = idxs[perm]
            n = subsample

        pg = p[idxs]
        tg = t[idxs]
        diff_t = tg.unsqueeze(0) - tg.unsqueeze(1)
        diff_p = pg.unsqueeze(0) - pg.unsqueeze(1)
        pair_mask = diff_t > 1e-6
        if not pair_mask.any():
            continue
        penalties = torch.clamp(margin - diff_p, min=0.0)
        rank_terms.append(penalties[pair_mask].mean())

    if not rank_terms:
        return None
    return torch.stack(rank_terms).mean()


# ---------------------------------------------------------------------------
# Loss classes
# ---------------------------------------------------------------------------


class _WeightedHuberLoss:
    def __init__(self, alpha: float = LOSS_ALPHA_WEIGHTED, delta: float = 1.0):
        self.alpha = alpha
        self.delta = delta

    def __call__(self, pred, target):
        w = 1.0 + self.alpha * torch.clamp(target, min=0.0)
        diff = pred - target
        absd = diff.abs()
        huber = torch.where(
            absd < self.delta,
            0.5 * diff * diff,
            self.delta * (absd - 0.5 * self.delta),
        )
        return (w * huber).mean()


class _PinballLoss:
    def __init__(self, tau: float = LOSS_TAU_PINBALL):
        self.tau = tau

    def __call__(self, pred, target):
        diff = target - pred
        return (
            torch.clamp(diff, min=0.0) * self.tau
            + torch.clamp(-diff, min=0.0) * (1.0 - self.tau)
        ).mean()


class _PairwiseRankHuberLoss:
    def __init__(
        self,
        alpha: float = 0.5,
        margin: float = LOSS_RANK_MARGIN,
        delta: float = 1.0,
        subsample: int = LOSS_RANK_SUBSAMPLE,
    ):
        self.alpha = alpha
        self.margin = margin
        self.delta = delta
        self.subsample = subsample

    def __call__(self, pred, target):
        huber_term = torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return huber_term
        rank_term = _pairwise_rank_term_per_graph(
            pred,
            target,
            batch,
            self.margin,
            self.subsample,
        )
        if rank_term is None:
            return huber_term
        return self.alpha * huber_term + (1.0 - self.alpha) * rank_term


class _PerGraphTopKHuberLoss:
    def __init__(
        self,
        k_frac: float = TOPK_FRACTION,
        k_weight: float = TOPK_WEIGHT,
        delta: float = 1.0,
    ):
        self.k_frac = k_frac
        self.k_weight = k_weight
        self.delta = delta

    def __call__(self, pred, target):
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        device = pred.device
        if batch.device != device:
            batch = batch.to(device)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        if p.numel() != batch.numel():
            return torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        diff = p - t
        absd = diff.abs()
        elem = torch.where(
            absd < self.delta,
            0.5 * diff * diff,
            self.delta * (absd - 0.5 * self.delta),
        )

        terms: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            n = int(mask.sum().item())
            if n == 0:
                continue
            t_g = t[mask]
            e_g = elem[mask]
            if n < 10:
                terms.append(e_g.mean())
                continue
            tau_g = torch.quantile(t_g, 1.0 - self.k_frac)
            above = (t_g >= tau_g).to(e_g.dtype)
            w_g = 1.0 + above * (self.k_weight - 1.0)
            terms.append((w_g * e_g).sum() / w_g.sum())

        if not terms:
            return torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        return torch.stack(terms).mean()


class _ListMLELoss:
    """Per-graph ListMLE (Plackett–Luce NLL of true ranking).

    Pure ranking — invariant to additive/multiplicative constants — so we
    blend with α·Huber to keep absolute calibration. anik tested this on the
    corrupted dataset and got negative R²; on v2 with correct orientations
    it might work now.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        delta: float = 1.0,
        subsample: int = LOSS_RANK_SUBSAMPLE,
    ):
        self.alpha = alpha
        self.delta = delta
        self.subsample = subsample

    def __call__(self, pred, target):
        huber_term = torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return huber_term
        device = pred.device
        if batch.device != device:
            batch = batch.to(device)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        if p.numel() != batch.numel():
            return huber_term
        terms: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            idxs = mask.nonzero(as_tuple=True)[0]
            n = idxs.numel()
            if n < 3:
                continue
            if n > self.subsample:
                perm = torch.randperm(n, device=device)[: self.subsample]
                idxs = idxs[perm]
                n = self.subsample
            pg = p[idxs]
            tg = t[idxs]
            order = torch.argsort(tg, descending=True)
            pg_ord = pg[order]
            reversed_scores = torch.flip(pg_ord, dims=[0])
            logcumsum_rev = torch.logcumsumexp(reversed_scores, dim=0)
            logsum_suffix = torch.flip(logcumsum_rev, dims=[0])
            terms.append((logsum_suffix - pg_ord).mean())
        if not terms:
            return huber_term
        list_term = torch.stack(terms).mean()
        return self.alpha * huber_term + (1.0 - self.alpha) * list_term


class _PeakHuberTopKLoss:
    """alpha*Huber + beta*per-graph pairwise rank + gamma*per-graph top-k Huber.

    The headline anik production loss. The Huber term keeps overall calibration;
    the rank term preserves within-graph ranking pressure; the per-graph top-k
    term explicitly upweights errors on the highest-value nodes within each
    graph (the hot-spots that the engineering task cares about).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        k_frac: float = TOPK_FRACTION,
        k_weight: float = TOPK_WEIGHT,
        margin: float = LOSS_RANK_MARGIN,
        delta: float = 1.0,
        subsample: int = LOSS_RANK_SUBSAMPLE,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.delta = delta
        self.subsample = subsample
        self.topk = _PerGraphTopKHuberLoss(
            k_frac=k_frac, k_weight=k_weight, delta=delta
        )

    def __call__(self, pred, target):
        huber = torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return huber
        rank_term = _pairwise_rank_term_per_graph(
            pred,
            target,
            batch,
            self.margin,
            self.subsample,
        )
        topk_term = self.topk(pred, target)
        if rank_term is None:
            return self.alpha * huber + self.gamma * topk_term
        return self.alpha * huber + self.beta * rank_term + self.gamma * topk_term


class _RankTopKHuberMC:
    """Huber on every channel + per-graph pairwise-rank + per-graph top-K
    hot-spot Huber upweight + per-graph |Δσ| L1 stdmatch, on a designated
    subset of "MAX channels".

    Multi-channel-aware fix for the silent-fallback bug in
    `_PeakHuberTopKLoss` / `_PairwiseRankHuberLoss` / `_ListMLELoss` /
    `_MSEWithStdMatch` (each has `if p.numel() != batch.numel():
    return huber_loss(...)` which fires on every multi-channel head
    call). This class loops over the designated channels and applies
    rank / top-K / stdmatch per-channel.

        L = (1 - lam_rank - lam_topk) * Huber_all_channels
            + lam_rank * mean_k mean_g pairwise_rank_pen(p[g, k], t[g, k])
            + lam_topk * mean_k mean_g topk_huber_g(p[g, k], t[g, k])
            + lam_stdmatch * mean_k mean_g |std_g(p[k]) - std_g(t[k])|

    The base Huber down-weight only subtracts lam_rank+lam_topk (NOT
    lam_stdmatch) so the v5-paper convention is preserved: stdmatch was
    an *additive* L1 term on top of full-weight MSE. Rank/topk are blends.

    Designed to PRESS on within-SVE rank + tail (top-1 hit, top-K recall)
    AND within-SVE dispersion of the per-grain MAX channels.

    `topk_frac` = fraction of true-largest grains per graph that get
    upweighted. `topk_w` = upweight factor on those grains.
    """

    def __init__(
        self,
        lam_rank: float = 0.3,
        lam_topk: float = 0.0,
        lam_stdmatch: float = 0.0,
        rank_channel_indices: Optional[List[int]] = None,
        margin: float = LOSS_RANK_MARGIN,
        subsample: int = LOSS_RANK_SUBSAMPLE,
        rank_top_frac: float = 0.10,
        topk_frac: float = TOPK_FRACTION,
        topk_w: float = TOPK_WEIGHT,
        delta: float = 1.0,
        min_grains_per_graph: int = 3,
    ):
        # rank_top_frac: only include pairs where the higher-target endpoint
        # is in the top quantile (default 0.10 = top decile). The rest of
        # the within-graph rank is unconstrained — fatigue cares about the
        # top of the distribution, not the bottom-90% ordering.
        self.lam_rank = lam_rank
        self.lam_topk = lam_topk
        self.lam_stdmatch = lam_stdmatch
        self.rank_idx = (
            list(rank_channel_indices) if rank_channel_indices is not None else [0]
        )
        self.margin = margin
        self.subsample = subsample
        self.rank_top_frac = rank_top_frac
        self.topk_frac = topk_frac
        self.topk_w = topk_w
        self.delta = delta
        self.min_g = min_grains_per_graph

    def __call__(self, pred, target):
        huber = torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        if self.lam_rank <= 0 and self.lam_topk <= 0 and self.lam_stdmatch <= 0:
            return huber
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return huber
        # Single-channel: apply directly.
        if pred.dim() < 2:
            extras = []
            if self.lam_rank > 0:
                r = self._rank_per_graph(pred, target, batch)
                if r is not None:
                    extras.append(self.lam_rank * r)
            if self.lam_topk > 0:
                k = self._topk_per_graph(pred, target, batch)
                if k is not None:
                    extras.append(self.lam_topk * k)
            if self.lam_stdmatch > 0:
                s = self._stdmatch_per_graph(pred, target, batch)
                if s is not None:
                    extras.append(self.lam_stdmatch * s)
            if not extras:
                return huber
            base_w = max(0.0, 1.0 - self.lam_rank - self.lam_topk)
            return base_w * huber + sum(extras)
        # Multi-channel: rank/topk/stdmatch only on designated MAX channels.
        rank_terms: List[torch.Tensor] = []
        topk_terms: List[torch.Tensor] = []
        std_terms: List[torch.Tensor] = []
        for k in self.rank_idx:
            if k >= pred.shape[-1]:
                continue
            if self.lam_rank > 0:
                r = self._rank_per_graph(pred[:, k], target[:, k], batch)
                if r is not None:
                    rank_terms.append(r)
            if self.lam_topk > 0:
                t = self._topk_per_graph(pred[:, k], target[:, k], batch)
                if t is not None:
                    topk_terms.append(t)
            if self.lam_stdmatch > 0:
                s = self._stdmatch_per_graph(pred[:, k], target[:, k], batch)
                if s is not None:
                    std_terms.append(s)
        extras: List[torch.Tensor] = []
        if rank_terms:
            extras.append(self.lam_rank * torch.stack(rank_terms).mean())
        if topk_terms:
            extras.append(self.lam_topk * torch.stack(topk_terms).mean())
        if std_terms:
            extras.append(self.lam_stdmatch * torch.stack(std_terms).mean())
        if not extras:
            return huber
        base_w = max(0.0, 1.0 - self.lam_rank - self.lam_topk)
        return base_w * huber + sum(extras)

    def _rank_per_graph(self, p_flat, t_flat, batch) -> Optional[torch.Tensor]:
        device = p_flat.device
        if batch.device != device:
            batch = batch.to(device)
        # Loss is invoked for BOTH per-grain (length N_grains) and per-graph
        # (length n_graphs) heads; only the per-grain call aligns with
        # batch.numel(). Short-circuit on the per-graph call.
        if p_flat.numel() != batch.numel():
            return None
        terms: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            idxs = mask.nonzero(as_tuple=True)[0]
            n = idxs.numel()
            if n < self.min_g:
                continue
            if n > self.subsample:
                perm = torch.randperm(n, device=device)[: self.subsample]
                idxs = idxs[perm]
                n = self.subsample
            pg = p_flat[idxs]
            tg = t_flat[idxs]
            # Top-decile gate: only count pairs where j is in top
            # `rank_top_frac` of the graph's targets. Fatigue tail is the
            # only physically-meaningful ordering; bottom-90% rank is noise.
            tau = torch.quantile(tg, 1.0 - self.rank_top_frac)
            top_mask = tg >= tau  # (n,) bool, marks "j" candidates
            diff_t = tg.unsqueeze(0) - tg.unsqueeze(1)
            diff_p = pg.unsqueeze(0) - pg.unsqueeze(1)
            # Allow pair (i, j) iff j is in top decile AND target_j > target_i.
            pair_mask = (diff_t > 1e-6) & top_mask.unsqueeze(1)
            if not pair_mask.any():
                continue
            pen = torch.clamp(self.margin - diff_p, min=0.0)
            terms.append(pen[pair_mask].mean())
        if not terms:
            return None
        return torch.stack(terms).mean()

    def _stdmatch_per_graph(self, p_flat, t_flat, batch) -> Optional[torch.Tensor]:
        """Per-graph |std(pred_g) - std(target_g)| L1 (v5-style stdmatch)."""
        device = p_flat.device
        if batch.device != device:
            batch = batch.to(device)
        if p_flat.numel() != batch.numel():
            return None  # short-circuit when called on per-graph head
        diffs: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            n = int(mask.sum().item())
            if n < self.min_g:
                continue
            pg = p_flat[mask]
            tg = t_flat[mask]
            diffs.append((pg.std(unbiased=False) - tg.std(unbiased=False)).abs())
        if not diffs:
            return None
        return torch.stack(diffs).mean()

    def _topk_per_graph(self, p_flat, t_flat, batch) -> Optional[torch.Tensor]:
        device = p_flat.device
        if batch.device != device:
            batch = batch.to(device)
        if p_flat.numel() != batch.numel():
            return None  # short-circuit when called on per-graph head
        diff = p_flat - t_flat
        absd = diff.abs()
        elem = torch.where(
            absd < self.delta,
            0.5 * diff * diff,
            self.delta * (absd - 0.5 * self.delta),
        )
        terms: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            n = int(mask.sum().item())
            if n < 10:
                terms.append(elem[mask].mean())
                continue
            t_g = t_flat[mask]
            e_g = elem[mask]
            tau = torch.quantile(t_g, 1.0 - self.topk_frac)
            above = (t_g >= tau).to(e_g.dtype)
            w_g = 1.0 + above * (self.topk_w - 1.0)
            terms.append((w_g * e_g).sum() / w_g.sum())
        if not terms:
            return None
        return torch.stack(terms).mean()


class _WassersteinHuber:
    """Huber on every channel + Gaussian-W₂² (mean+std matching) per-SVE
    on a designated subset of "MAX channels" of a multi-channel head.

    The W₂² term is the Gaussian-closed-form 2-Wasserstein squared:

        W₂²_g(channel k) = (μ_g(pred_k) - μ_g(target_k))²
                          + (σ_g(pred_k) - σ_g(target_k))²

    averaged across SVEs g and across designated MAX channels k.
    Multiplied by λ and added to base Huber loss.

    Designed for multi-channel heads (v9-onwards). Unlike
    _PeakHuberTopKLoss whose rank/topk terms silently fall back to plain
    Huber on multi-channel input, this loss explicitly handles the
    (N_nodes, K) shape and only fires the W₂² term on the channels
    where within-SVE distribution matching is the physical target
    (per-grain MAX of FIP per variant).

    Restores the per-SVE rank+scale pressure that v5's stdmatch had
    (per `mse_stdmatch_is_wasserstein_cousin_2026-05-05.md` memo) but
    in proper Gaussian-W₂ form. Cite Givens & Shortt (1984) for the
    closed form in the paper.
    """

    def __init__(
        self,
        lam: float = 0.5,
        max_channel_indices: Optional[List[int]] = None,
        delta: float = 1.0,
        min_grains_per_graph: int = 5,
    ):
        self.lam = lam
        self.max_idx = (
            list(max_channel_indices) if max_channel_indices is not None else [0]
        )
        self.delta = delta
        self.min_g = min_grains_per_graph

    def __call__(self, pred, target):
        huber = torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        if self.lam <= 0:
            return huber
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return huber
        if pred.dim() < 2:
            # Single-channel head — slot 0 is the only channel.
            return huber + self.lam * self._w2_per_graph(pred, target, batch)
        # Multi-channel head — apply W₂² only on designated MAX channels.
        terms: List[torch.Tensor] = []
        for k in self.max_idx:
            if k >= pred.shape[-1]:
                continue
            terms.append(self._w2_per_graph(pred[:, k], target[:, k], batch))
        if not terms:
            return huber
        return huber + self.lam * torch.stack(terms).mean()

    def _w2_per_graph(self, p_flat, t_flat, batch) -> torch.Tensor:
        """Gaussian-W₂² on a single channel, averaged across SVEs."""
        device = p_flat.device
        if batch.device != device:
            batch = batch.to(device)
        # Vector p_flat / t_flat must align with batch (one entry per node).
        if p_flat.numel() != batch.numel():
            return torch.tensor(0.0, device=device)
        diffs: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            if int(mask.sum().item()) < self.min_g:
                continue
            pg = p_flat[mask]
            tg = t_flat[mask]
            mu_diff = pg.mean() - tg.mean()
            sd_diff = pg.std(unbiased=False) - tg.std(unbiased=False)
            diffs.append(mu_diff * mu_diff + sd_diff * sd_diff)
        if not diffs:
            return torch.tensor(0.0, device=device)
        return torch.stack(diffs).mean()


class _MSEWithStdMatch:
    """MSE plus a per-graph within-SVE standard-deviation matching term.

    The intent is to give the optimiser a direct signal on the *spread*
    of per-grain predictions inside each \\sve. By the
    method-of-moments mapping for a Gumbel
    ($\\beta = \\sigma \\sqrt{6}/\\pi$), matching within-SVE std is
    equivalent to matching the Gumbel scale parameter. We compute it
    in transformed-target space (where MSE is computed); since the
    bucketed $z$-score is affine within a bucket, the within-SVE std
    of $\\tilde y$ equals the within-SVE std of the raw $y$ divided
    by the bucket scale, and the matching loss therefore retains the
    same dimensionless meaning.

    Loss = MSE(pred, target)
         + lambda * mean_g | std_g(pred) - std_g(target) |
    """

    def __init__(self, lam: float = 0.5, min_n_per_graph: int = 5):
        self.lam = lam
        self.min_n_per_graph = min_n_per_graph

    def __call__(self, pred, target):
        mse = torch.nn.functional.mse_loss(pred, target)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None or self.lam <= 0.0:
            return mse
        device = pred.device
        if batch.device != device:
            batch = batch.to(device)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        # The std-matching term only makes sense for the per-grain head, where
        # `pred` has one entry per grain (length == batch tensor length). The
        # per-graph head has one entry per SVE; for it, fall back to plain MSE.
        if p.numel() != batch.numel():
            return mse
        diffs: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() < self.min_n_per_graph:
                continue
            std_p = p[idxs].std(unbiased=False)
            std_t = t[idxs].std(unbiased=False)
            diffs.append(torch.abs(std_p - std_t))
        if not diffs:
            return mse
        match_term = torch.stack(diffs).mean()
        return mse + self.lam * match_term


class _MSEGumbelMomentMatch:
    """MSE plus per-graph Gumbel(mu, beta) moment matching.

    For each SVE g, fit a Gumbel by method-of-moments to the per-grain
    MAX values inside g (population-std variant):

        beta_g = sigma_g * sqrt(6) / pi
        mu_g   = ybar_g - beta_g * gamma_E    (gamma_E = 0.5772 Euler)

    The matching loss penalises the L1 deviation of the predicted
    Gumbel parameters (computed the same way from the per-grain
    predictions inside g) from the truth's Gumbel parameters:

        Loss = MSE(pred, target)
             + lam_b * mean_g | beta_g - beta_g_hat |
             + lam_m * mean_g |   mu_g -   mu_g_hat |

    This generalises mse_stdmatch (which is the beta term alone, up
    to a constant sqrt(6)/pi ~ 0.78) by adding mu matching. The mu
    matching is correlated with mean matching (already implicit in
    MSE) but adds an explicit signal in the Gumbel-physics frame.
    """

    def __init__(
        self, lam_b: float = 0.5, lam_m: float = 0.5, min_n_per_graph: int = 5
    ):
        self.lam_b = lam_b
        self.lam_m = lam_m
        self.min_n = min_n_per_graph

    def __call__(self, pred, target):
        mse = torch.nn.functional.mse_loss(pred, target)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None or (self.lam_b <= 0.0 and self.lam_m <= 0.0):
            return mse
        if batch.device != pred.device:
            batch = batch.to(pred.device)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        if p.numel() != batch.numel():
            return mse
        b_diffs: List[torch.Tensor] = []
        m_diffs: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() < self.min_n:
                continue
            p_g = p[idxs]
            t_g = t[idxs]
            sig_p = p_g.std(unbiased=False)
            sig_t = t_g.std(unbiased=False)
            beta_p = sig_p * SQRT6_OVER_PI
            beta_t = sig_t * SQRT6_OVER_PI
            mu_p = p_g.mean() - beta_p * GAMMA_EULER
            mu_t = t_g.mean() - beta_t * GAMMA_EULER
            b_diffs.append(torch.abs(beta_p - beta_t))
            m_diffs.append(torch.abs(mu_p - mu_t))
        loss = mse
        if b_diffs and self.lam_b > 0.0:
            loss = loss + self.lam_b * torch.stack(b_diffs).mean()
        if m_diffs and self.lam_m > 0.0:
            loss = loss + self.lam_m * torch.stack(m_diffs).mean()
        return loss


class _MSEStdMatchMaxMargin:
    """MSE + lam_std * within-SVE std-match + lam_mar * within-SVE max-margin.

    The max-margin term directly optimises the within-SVE top-1 hit rate.
    For each SVE g with i* = argmax over the true per-grain target inside g,
    we penalise

        relu(margin - (pred[i*] - max_{j != i*} pred[j])).

    The penalty is zero whenever the prediction at the true argmax exceeds
    every other prediction in the SVE by at least `margin`; otherwise it
    grows linearly with the deficit. Because i* is a function of the truth
    only, the gradient of this term flows through both the prediction at
    the true argmax (positive direction) and the prediction at the
    currently-arg-max-of-pred (negative direction), pushing them apart.

    Loss = MSE(pred, target)
         + lam_std * mean_g | std_g(pred) - std_g(target) |
         + lam_mar * mean_g relu(margin - (pred[i*_g] - max_{j != i*_g} pred[j]))

    Recipe: "mse_stdmm_<LS>_<LM>" with LS = lam_std * 100, LM = lam_mar * 100.
    Default margin = 0.5 (in transformed-target units; targets are bucketed
    z-scores, so 0.5 = half a within-bucket standard deviation).
    """

    def __init__(
        self,
        lam_std: float = 0.5,
        lam_mar: float = 0.5,
        margin: float = 0.5,
        min_n_per_graph: int = 5,
    ):
        self.lam_std = lam_std
        self.lam_mar = lam_mar
        self.margin = margin
        self.min_n = min_n_per_graph

    def __call__(self, pred, target):
        mse = torch.nn.functional.mse_loss(pred, target)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None or (self.lam_std <= 0.0 and self.lam_mar <= 0.0):
            return mse
        if batch.device != pred.device:
            batch = batch.to(pred.device)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        if p.numel() != batch.numel():
            return mse
        std_diffs: List[torch.Tensor] = []
        margin_terms: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() < self.min_n:
                continue
            p_g = p[idxs]
            t_g = t[idxs]
            std_diffs.append(
                torch.abs(p_g.std(unbiased=False) - t_g.std(unbiased=False))
            )
            i_star = t_g.argmax()
            p_at_max = p_g[i_star]
            mask_not_max = torch.ones_like(p_g, dtype=torch.bool)
            mask_not_max[i_star] = False
            max_others = p_g[mask_not_max].max()
            margin_terms.append(
                torch.clamp(self.margin - (p_at_max - max_others), min=0.0)
            )
        loss = mse
        if std_diffs and self.lam_std > 0.0:
            loss = loss + self.lam_std * torch.stack(std_diffs).mean()
        if margin_terms and self.lam_mar > 0.0:
            loss = loss + self.lam_mar * torch.stack(margin_terms).mean()
        return loss


class _MSEGumbelTailWeighted:
    """Per-grain MSE re-weighted by within-SVE Gumbel-CDF percentile.

    For each SVE g:
      * fit Gumbel(mu_g, beta_g) by MoM to TRUE within-SVE per-grain MAX values
      * for each grain i: u_i = exp(-exp(-(t_i - mu_g)/beta_g))   in [0, 1]
        (the Gumbel CDF percentile)
      * weight w_i = (1 - alpha) + alpha * u_i ** k

    Per-grain MSE residuals are re-weighted by w_i, normalised per SVE,
    averaged over SVEs.

    Compared with peak_huber_topk's hard top-decile threshold, this
    smoothly upweights the upper tail under the Gumbel CDF, with a
    steepness exponent k. Larger alpha and k push more weight to the
    extreme tail; alpha = 0 reduces to plain MSE.
    """

    def __init__(self, alpha: float = 0.5, k: float = 3.0, min_n_per_graph: int = 5):
        self.alpha = alpha
        self.k = k
        self.min_n = min_n_per_graph

    def __call__(self, pred, target):
        mse = torch.nn.functional.mse_loss(pred, target)
        batch = _CURRENT_BATCH.get("batch")
        if batch is None:
            return mse
        if batch.device != pred.device:
            batch = batch.to(pred.device)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        if p.numel() != batch.numel():
            return mse
        terms: List[torch.Tensor] = []
        for g in torch.unique(batch):
            mask = batch == g
            idxs = mask.nonzero(as_tuple=True)[0]
            n = idxs.numel()
            if n < self.min_n:
                continue
            p_g = p[idxs]
            t_g = t[idxs]
            sigma_t = t_g.std(unbiased=False).clamp(min=1e-8)
            beta_t = sigma_t * SQRT6_OVER_PI
            mu_t = t_g.mean() - beta_t * GAMMA_EULER
            z = (t_g - mu_t) / beta_t
            u = torch.exp(-torch.exp(-z))
            w = (1.0 - self.alpha) + self.alpha * u.pow(self.k)
            res = (p_g - t_g) ** 2
            terms.append((w * res).sum() / w.sum().clamp(min=1e-8))
        if not terms:
            return mse
        return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Phase L3 cheap tail-targeted losses (≤ 1.1× Huber speed, multi-channel safe)
# ---------------------------------------------------------------------------
# Diagnosis. v9 Phase L1/L6 parity plots show systematic upper-tail under-
# prediction even when std_ratio ≈ 1.0 (L6_stdmatch_75 HCF/FS: bulk R²=0.94,
# std_ratio=1.0, but predictions cap at p99_ratio=0.88 — visible as a kink
# in the parity plot). std-match alone stretches the bulk without lifting
# the tail. The mc_huber_*_*_* family is 5–25× slower (per-channel × per-
# component × per-graph aggregations); not viable for a 100-trial HPO.
# These four classes target the upper tail with a single forward pass and
# at most one per-batch sort or quantile — each runs ≤ 1.1× plain Huber.


class _AsymmetricHuberLoss:
    """Huber with γ× stronger penalty for under-prediction (pred < target).

    Mechanism. The standard Huber gradient is symmetric in (pred − target),
    so a model that under-predicts the tail and over-predicts the bulk by
    similar amounts has zero net gradient pull toward the tail. With γ > 1,
    the negative-residual side gets γ× the gradient, breaking the symmetry
    and pulling predictions UP. γ=2 is mild; γ=4 is aggressive.

    Multi-channel safe: torch.where + element-wise multiply broadcast.
    """

    def __init__(self, gamma: float = 2.0, delta: float = 1.0):
        self.gamma = float(gamma)
        self.delta = float(delta)

    def __call__(self, pred, target):
        diff = pred - target
        absd = diff.abs()
        base = torch.where(
            absd < self.delta,
            0.5 * diff * diff,
            self.delta * (absd - 0.5 * self.delta),
        )
        weight = torch.where(
            diff < 0, torch.full_like(base, self.gamma), torch.ones_like(base)
        )
        return (weight * base).mean()


class _TailWeightedHuberLoss:
    """Huber weighted by (per-batch, per-channel rank of target)^α.

    Mechanism. After argsort-twice, each grain gets a rank ∈ [0, 1] within
    its batch (by true FIP per channel). Weight = (rank)^α tilts the loss
    toward higher-ranked grains. α=0 → plain Huber; α=1 → linear up-tilt;
    α=2 → quadratic; α=3 → strong tail focus. Weights are renormalised so
    mean weight = 1, preserving loss scale.

    Cost. One argsort.argsort per channel — O(N log N), tiny vs forward pass.
    """

    def __init__(self, alpha: float = 1.0, delta: float = 1.0):
        self.alpha = float(alpha)
        self.delta = float(delta)

    def __call__(self, pred, target):
        if target.dim() == 1:
            n = max(1, target.numel() - 1)
            ranks = target.argsort(0).argsort(0).float() / n
        else:
            n = max(1, target.shape[0] - 1)
            ranks = target.argsort(0).argsort(0).float() / n
        weights = (ranks + 1e-6).pow(self.alpha)
        # Normalise so mean weight per channel = 1 (preserves loss scale).
        weights = weights / weights.mean(dim=0, keepdim=True).clamp(min=1e-9)
        diff = pred - target
        absd = diff.abs()
        base = torch.where(
            absd < self.delta,
            0.5 * diff * diff,
            self.delta * (absd - 0.5 * self.delta),
        )
        return (weights * base).mean()


class _FocalTopDecileHuberLoss:
    """Plain Huber + λ × MSE on the per-batch, per-channel top-decile of true.

    Mechanism. Adds an explicit MSE term that activates only on grains in
    the upper top_frac of true values per channel. The squared error on
    these tail grains creates much larger gradients than Huber would (since
    the residuals on tail grains are often > δ, where Huber clips to L1).
    λ=0 → plain Huber; λ=0.5 → tail term contributes ~half the gradient;
    λ=1 → equal weight.

    Cost. One torch.quantile per channel — O(N) — plus a masked mean. Small.
    """

    def __init__(self, lam: float = 0.5, top_frac: float = 0.10, delta: float = 1.0):
        self.lam = float(lam)
        self.top_frac = float(top_frac)
        self.delta = float(delta)

    def __call__(self, pred, target):
        diff = pred - target
        absd = diff.abs()
        base = torch.where(
            absd < self.delta,
            0.5 * diff * diff,
            self.delta * (absd - 0.5 * self.delta),
        ).mean()

        if target.dim() == 1:
            thr = torch.quantile(target, 1.0 - self.top_frac)
            mask = target >= thr
        else:
            thr = torch.quantile(target, 1.0 - self.top_frac, dim=0, keepdim=True)
            mask = target >= thr
        if mask.any():
            tail_mse = ((pred - target) ** 2)[mask].mean()
            return base + self.lam * tail_mse
        return base


class _HeteroscedasticTopdecLoss:
    """Phase L5 — heteroscedastic Gaussian NLL on per-grain MAX channels +
    plain Huber on per-grain mean channels + optional topdec aux on μ_max.

    Architecture contract. Per-grain head emits N_grains × 12 channels:
      [..., 0:4]   = μ predictions for max channels (4 cells: lcf_fs, lcf_wp, hcf_fs, hcf_wp)
      [..., 4:8]   = μ predictions for mean channels (4 cells; within-grain mean)
      [..., 8:12]  = log_σ predictions for max channels (paired with 0:4)

    Target is N_grains × 12 with dummy zeros in [..., 8:12] (the σ slots
    are not supervised — they're nuisance parameters). The loss class
    ignores the σ target slots.

    NLL on max channels (per cell):
        L_NLL = 0.5 · log(σ²) + 0.5 · (μ_pred − target)² / σ²
              = log_σ + 0.5 · residual² · exp(-2·log_σ)

    Plain Huber on mean channels.

    Optional topdec aux on μ_max: 0.5·MSE on per-channel top-decile of
    true MAX (provides tail magnitude pressure independent of σ output).
    Set λ_topdec=0 to use pure NLL.

    σ floor: clamp log_σ ≥ log_sigma_floor (default −3, i.e. σ ≥ 0.05) to
    prevent the NLL from collapsing to σ→0 early in training.

    Why this is different from the failed v7 σ-aux approach. v7 P7 added
    σ_target as a SUPERVISED regression target (Huber on σ_pred vs
    σ_target); the σ R² hit ceiling 0.35 and the σ outputs drained
    encoder capacity. Here σ is a NUISANCE PARAMETER inferred to make
    NLL smaller — it gets gradient signal from how well it scales the
    residual variance per-grain, not from matching a stationary target.
    The model's freedom to predict large σ on hard grains and small σ on
    easy grains is what should break the tail-plateau pattern.
    """

    def __init__(
        self,
        n_max: int = 4,
        n_mean: int = 4,
        n_extra: int = 0,
        lam_topdec: float = 0.0,
        top_frac: float = 0.10,
        delta: float = 1.0,
        log_sigma_floor: float = -3.0,
    ):
        self.n_max = int(n_max)
        self.n_mean = int(n_mean)
        self.n_extra = int(n_extra)
        self.lam_topdec = float(lam_topdec)
        self.top_frac = float(top_frac)
        self.delta = float(delta)
        self.log_sigma_floor = float(log_sigma_floor)

    def __call__(self, pred, target):
        nm = self.n_max
        nn = self.n_mean
        ne = self.n_extra  # extra supervised distribution-shape targets (Huber)
        n_super = nm + nn + ne
        n_total = n_super + nm  # σ_pn block paired with first nm channels

        # Single-channel fall-back (called on per-graph head): plain Huber.
        if pred.dim() < 2:
            return torch.nn.functional.huber_loss(pred, target, delta=self.delta)

        # If the per-grain head doesn't have the σ extension, fall back to
        # plain Huber (defensive: cell may be miswired).
        if pred.shape[-1] != n_total:
            target_eff = target[..., : pred.shape[-1]] if target.dim() >= 1 else target
            return torch.nn.functional.huber_loss(pred, target_eff, delta=self.delta)

        mu_max = pred[..., :nm]
        mu_mean = pred[..., nm : nm + nn]
        # n_extra supervised distribution-shape channels in the middle
        # (e.g. lognormal_μ, lognormal_σ, p90, p99 — depends on recipe).
        # σ block remains paired with the first nm channels (max).
        if ne > 0:
            mu_extra = pred[..., nm + nn : nm + nn + ne]
            target_extra = target[..., nm + nn : nm + nn + ne]
        log_sigma = pred[..., nm + nn + ne :]
        target_max = target[..., :nm]
        target_mean = target[..., nm : nm + nn]

        log_sigma_eff = torch.clamp(log_sigma, min=self.log_sigma_floor)
        log_var = 2.0 * log_sigma_eff
        inv_var = torch.exp(-log_var)

        diff = mu_max - target_max
        nll = 0.5 * (log_var + diff * diff * inv_var)
        valid_max = torch.isfinite(target_max)
        nll = torch.where(valid_max, nll, torch.zeros_like(nll))
        n_v_max = valid_max.sum().clamp(min=1).float()
        nll_loss = nll.sum() / n_v_max

        diff_mean = mu_mean - target_mean
        absd = diff_mean.abs()
        huber_mean = torch.where(
            absd < self.delta,
            0.5 * diff_mean * diff_mean,
            self.delta * (absd - 0.5 * self.delta),
        )
        valid_mean = torch.isfinite(target_mean)
        huber_mean = torch.where(valid_mean, huber_mean, torch.zeros_like(huber_mean))
        n_v_mean = valid_mean.sum().clamp(min=1).float()
        huber_mean_loss = huber_mean.sum() / n_v_mean

        loss = nll_loss + huber_mean_loss

        # Huber on extra supervised distribution-shape channels (NEW for
        # node-level distribution targets — Proposals A/B/C 2026-05-06).
        # n_extra=0 ⇒ this branch is a no-op; recipe sets n_extra via
        # `_distA` / `_distB` / `_distC` suffix.
        if ne > 0:
            diff_extra = mu_extra - target_extra
            absd_e = diff_extra.abs()
            huber_extra = torch.where(
                absd_e < self.delta,
                0.5 * diff_extra * diff_extra,
                self.delta * (absd_e - 0.5 * self.delta),
            )
            valid_extra = torch.isfinite(target_extra)
            huber_extra = torch.where(
                valid_extra, huber_extra, torch.zeros_like(huber_extra)
            )
            n_v_extra = valid_extra.sum().clamp(min=1).float()
            huber_extra_loss = huber_extra.sum() / n_v_extra
            loss = loss + huber_extra_loss

        # Optional topdec aux on μ_max — adds tail magnitude pressure even
        # when σ output absorbs residual variance.
        if self.lam_topdec > 0:
            tail_terms = []
            for k in range(nm):
                tk = target_max[..., k]
                vk = valid_max[..., k]
                if not vk.any():
                    continue
                col = tk[vk]
                thr = torch.quantile(col, 1.0 - self.top_frac)
                mask = (tk >= thr) & vk
                if mask.any():
                    rk = (mu_max[..., k] - tk)[mask]
                    tail_terms.append((rk * rk).mean())
            if tail_terms:
                loss = loss + self.lam_topdec * torch.stack(tail_terms).mean()

        return loss


class _HeteroscedasticPergraphLoss:
    """Phase L6 — heteroscedastic Gaussian NLL on the per-graph head's 4
    SVE-MAX channels. Sister of `_HeteroscedasticTopdecLoss` at the SVE level.

    Architecture contract. Per-graph head emits N_graphs × 17 channels:
      [..., 0:4]   = μ predictions for the 4 SVE-MAX cells (lcf_fs, lcf_wp,
                       hcf_fs, hcf_wp — order set by `_V9_PERGRAPH_TARGETS_13`)
      [..., 4:8]   = μ predictions for 4 within-SVE μ cells (ve_mu)
      [..., 8:13]  = μ predictions for 5 cyclic-plasticity mech QoIs
      [..., 13:17] = log_σ predictions paired with the 4 SVE-MAX cells

    Target shape: N_graphs × 17 with zeros in [..., 13:17] (σ slots are
    unsupervised nuisance parameters).

    Loss = NLL(μ_max, σ_max, target_max)              averaged over 4 cells
         + Huber(μ_ve_mu, target_ve_mu)               averaged over 4 cells
         + Huber(μ_mech, target_mech)                 averaged over 5 mech QoIs

    No topdec auxiliary at the per-graph level — per-batch top-decile of
    ~16 SVEs is 1–2 SVEs, not enough gradient signal (user decision 2026-05-06).

    σ floor: log_σ ≥ log_sigma_floor (default −3 → σ ≥ 0.05).

    Why this is different from v8's failed per-SVE β attempt: that treated β
    as a SUPERVISED regression target where β was sampling-noise-dominated
    (β R²≈0.067; ~30 grains per SVE → high MoM variance). Here σ is a
    nuisance parameter inferred to make NLL smaller per SVE.
    """

    def __init__(
        self,
        n_max: int = 4,
        n_ve_mu: int = 4,
        n_mech: int = 6,
        delta: float = 1.0,
        log_sigma_floor: float = -3.0,
    ):
        # n_mech default updated 2026-05-06 (was 5) per the mech-QoI redesign:
        # yield_stress, hardening_modulus, Wp_lcf, Wp_hcf, sigma_amp_lcf, sigma_amp_hcf.
        self.n_max = int(n_max)
        self.n_ve_mu = int(n_ve_mu)
        self.n_mech = int(n_mech)
        self.delta = float(delta)
        self.log_sigma_floor = float(log_sigma_floor)

    def __call__(self, pred, target):
        nm, nv, nq = self.n_max, self.n_ve_mu, self.n_mech
        n_super = nm + nv + nq
        n_total = n_super + nm

        if pred.dim() < 2:
            return torch.nn.functional.huber_loss(pred, target, delta=self.delta)
        if pred.shape[-1] != n_total:
            target_eff = target[..., : pred.shape[-1]] if target.dim() >= 1 else target
            return torch.nn.functional.huber_loss(pred, target_eff, delta=self.delta)

        mu_max = pred[..., :nm]
        mu_ve_mu = pred[..., nm : nm + nv]
        mu_mech = pred[..., nm + nv : n_super]
        log_sigma = pred[..., n_super:]
        target_max = target[..., :nm]
        target_ve_mu = target[..., nm : nm + nv]
        target_mech = target[..., nm + nv : n_super]

        log_sigma_eff = torch.clamp(log_sigma, min=self.log_sigma_floor)
        log_var = 2.0 * log_sigma_eff
        inv_var = torch.exp(-log_var)

        diff_max = mu_max - target_max
        nll = 0.5 * (log_var + diff_max * diff_max * inv_var)
        valid_max = torch.isfinite(target_max)
        nll = torch.where(valid_max, nll, torch.zeros_like(nll))
        nll_loss = nll.sum() / valid_max.sum().clamp(min=1).float()

        def _huber_block(diff, target_block):
            absd = diff.abs()
            base = torch.where(
                absd < self.delta,
                0.5 * diff * diff,
                self.delta * (absd - 0.5 * self.delta),
            )
            valid = torch.isfinite(target_block)
            base = torch.where(valid, base, torch.zeros_like(base))
            return base.sum() / valid.sum().clamp(min=1).float()

        huber_ve_mu = _huber_block(mu_ve_mu - target_ve_mu, target_ve_mu)
        huber_mech = _huber_block(mu_mech - target_mech, target_mech)
        return nll_loss + huber_ve_mu + huber_mech


class _HetMultiHead:
    """Phase L6 dispatcher — routes per-grain-shape pred to `pn_loss` and
    per-graph-shape pred to `pg_loss`. HydraGNN calls model.loss_function
    once per output head; we detect head by pred.shape[-1].

    Layouts (when corresponding use_heteroscedastic_*=True):
      per-grain head: shape (N_grains, pn_total)  — default 12 (8 supervised + 4 σ)
      per-graph head: shape (N_graphs, pg_total)  — default 17 (13 supervised + 4 σ)

    Unrecognised shapes fall back to plain Huber on the supervised slice.
    """

    def __init__(
        self,
        pn_loss=None,
        pg_loss=None,
        pn_total: int = 12,
        pg_total: int = 18,
        delta: float = 1.0,
    ):
        # pg_total default updated 2026-05-06 (was 17). After the mech-QoI
        # redesign the per-graph head emits 4 SVE-MAX + 4 ve_mu + 6 mech = 14
        # supervised + 4 σ outputs = 18 total channels when L6 het is on.
        self.pn_loss = pn_loss
        self.pg_loss = pg_loss
        self.pn_total = int(pn_total)
        self.pg_total = int(pg_total)
        self.delta = float(delta)

    def __call__(self, pred, target):
        if pred.dim() >= 2:
            shape = pred.shape[-1]
            if shape == self.pn_total and self.pn_loss is not None:
                return self.pn_loss(pred, target)
            if shape == self.pg_total and self.pg_loss is not None:
                return self.pg_loss(pred, target)
        if target.dim() >= 1 and pred.shape[-1] != target.shape[-1]:
            target = target[..., : pred.shape[-1]]
        return torch.nn.functional.huber_loss(pred, target, delta=self.delta)


class _ExpectileHuberLoss:
    """Expectile regression at τ > 0.5 with Huber smoothing.

    Mechanism. Standard MSE (τ=0.5) predicts the conditional mean. At τ=0.85,
    the optimiser predicts a value such that the prob-mass weighted residual
    is balanced with τ on the under-prediction side and (1−τ) on over-pred.
    The result is a "conservative high estimate" that naturally bias toward
    the upper tail. Like quantile regression but with squared-error bowl.
    Huber form: replace `err²` with Huber for robustness against outliers.

    Reference. Newey & Powell (1987), "Asymmetric Least Squares Estimation."
    Cost. One torch.where + one Huber-style branch — no sort, no quantile.
    """

    def __init__(self, tau: float = 0.85, delta: float = 1.0):
        self.tau = float(tau)
        self.delta = float(delta)

    def __call__(self, pred, target):
        err = target - pred  # positive when under-predicting
        absd = err.abs()
        huber_base = torch.where(
            absd <= self.delta,
            0.5 * err * err,
            self.delta * (absd - 0.5 * self.delta),
        )
        weight = torch.where(
            err > 0,
            torch.full_like(err, self.tau),
            torch.full_like(err, 1.0 - self.tau),
        )
        return (weight * huber_base).mean()


# ---------------------------------------------------------------------------
# Builder + model integration
# ---------------------------------------------------------------------------


def build_custom_loss(name: str):
    """Return a (pred, target) -> scalar callable for the given loss name."""
    if name in (None, "huber", ""):
        return torch.nn.HuberLoss()
    if name == "mse":
        return torch.nn.functional.mse_loss
    if name == "mae":
        return torch.nn.functional.l1_loss
    if name == "weighted_huber":
        return _WeightedHuberLoss(alpha=LOSS_ALPHA_WEIGHTED)
    # Phase L3 cheap tail-targeted losses.
    if name.startswith("asym_huber_"):
        # asym_huber_GG → γ = GG/10 (e.g. asym_huber_20 → 2.0×).
        suffix = name[len("asym_huber_") :]
        gamma = float(suffix) / 10.0
        return _AsymmetricHuberLoss(gamma=gamma)
    if name.startswith("tw_huber_"):
        # tw_huber_AA → α = AA/10 (e.g. tw_huber_10 → α=1.0, tw_huber_20 → 2.0).
        suffix = name[len("tw_huber_") :]
        alpha = float(suffix) / 10.0
        return _TailWeightedHuberLoss(alpha=alpha)
    if name.startswith("topdec_huber_"):
        # topdec_huber_LL → λ = LL/100 (e.g. topdec_huber_50 → λ=0.5).
        # Optional second token for top fraction: topdec_huber_50_05 → top 5 %.
        rest = name[len("topdec_huber_") :]
        parts = rest.split("_")
        lam = float(parts[0]) / 100.0
        top_frac = float(parts[1]) / 100.0 if len(parts) > 1 else 0.10
        return _FocalTopDecileHuberLoss(lam=lam, top_frac=top_frac)
    if name.startswith("expect_huber_"):
        # expect_huber_TT → τ = TT/100 (e.g. expect_huber_85 → τ=0.85).
        suffix = name[len("expect_huber_") :]
        tau = float(suffix) / 100.0
        return _ExpectileHuberLoss(tau=tau)
    if name.startswith("het_nll_pg"):
        # Phase L6 — pure heteroscedastic NLL on per-graph SVE-MAX channels
        # (no topdec aux; per-graph head has too-few SVEs for top-decile
        # gradient signal). Recipe always wrapped in a `_HetMultiHead`
        # dispatcher so it can also be invoked on the per-grain head (where
        # it falls through to plain Huber if no per-grain σ is configured).
        # Format: "het_nll_pg" → standalone pg-only mode
        #         "het_nll_both_LL" → both pn (λ_topdec=LL/100) AND pg
        #         "het_nll_both_LL_TT" → top_frac=TT/100 on the pn side
        return _HetMultiHead(pn_loss=None, pg_loss=_HeteroscedasticPergraphLoss())
    if name.startswith("het_nll_both"):
        rest = name[len("het_nll_both") :].lstrip("_")
        # Distribution-shape target extension (Proposals A/B/C 2026-05-06).
        # Recipe suffix `_distA` ⇒ +4 supervised channels (σ_log×4 cells)
        #                `_distB` ⇒ +8 (μ_log×4 + σ_log×4)
        #                `_distC` ⇒ +8 (p90×4 + p99×4)
        n_extra = 0
        for tag, ne in (("_distA", 4), ("_distB", 8), ("_distC", 8)):
            if rest.endswith(tag):
                n_extra = ne
                rest = rest[: -len(tag)]
                break
        lam_topdec = 0.0
        top_frac = 0.10
        if rest:
            parts = rest.split("_")
            lam_topdec = float(parts[0]) / 100.0
            if len(parts) > 1:
                top_frac = float(parts[1]) / 100.0
        pn_loss = _HeteroscedasticTopdecLoss(
            lam_topdec=lam_topdec,
            top_frac=top_frac,
            n_extra=n_extra,
        )
        pg_loss = _HeteroscedasticPergraphLoss()
        # pn_total = 8 supervised + n_extra + 4 σ_pn
        return _HetMultiHead(pn_loss=pn_loss, pg_loss=pg_loss, pn_total=12 + n_extra)
    if name.startswith("het_nll"):
        # Phase L5 — Heteroscedastic Gaussian NLL on per-grain MAX channels.
        # Format: "het_nll" → λ_topdec=0
        #         "het_nll_topdec_LL" → λ_topdec=LL/100, top_frac=0.10
        #         "het_nll_topdec_LL_TT" → top_frac=TT/100
        # Wrapped in `_HetMultiHead` so the same callable can be invoked on
        # the per-graph head (falls through to plain Huber when pg σ is off).
        rest = name[len("het_nll") :].lstrip("_")
        lam_topdec = 0.0
        top_frac = 0.10
        if rest.startswith("topdec_"):
            parts = rest[len("topdec_") :].split("_")
            lam_topdec = float(parts[0]) / 100.0
            if len(parts) > 1:
                top_frac = float(parts[1]) / 100.0
        pn_loss = _HeteroscedasticTopdecLoss(
            lam_topdec=lam_topdec,
            top_frac=top_frac,
        )
        return _HetMultiHead(pn_loss=pn_loss, pg_loss=None)
    if name.startswith("pinball_"):
        suffix = name[len("pinball_") :]
        tau = float(suffix) / 100.0
        return _PinballLoss(tau=tau)
    if name.startswith("rank_huber_"):
        suffix = name.rsplit("_", 1)[1]
        alpha = float(suffix) / 10.0
        return _PairwiseRankHuberLoss(alpha=alpha)
    if name.startswith("listmle_"):
        suffix = name.rsplit("_", 1)[1]
        alpha = float(suffix) / 10.0
        return _ListMLELoss(alpha=alpha)
    if name.startswith("mse_stdmatch"):
        # Format: "mse_stdmatch_LL" where LL is lambda × 100, e.g.
        # mse_stdmatch_05 → lambda = 0.05, mse_stdmatch_50 → lambda = 0.50.
        # Bare "mse_stdmatch" defaults to lambda = 0.5.
        suffix = name[len("mse_stdmatch") :].lstrip("_")
        lam = float(suffix) / 100.0 if suffix else 0.5
        return _MSEWithStdMatch(lam=lam)
    if name.startswith("mse_stdmm_"):
        # Format: "mse_stdmm_LS_LM" with LS = lam_std × 100, LM = lam_mar × 100.
        # Example: mse_stdmm_50_50 → lam_std = 0.50, lam_mar = 0.50, margin = 0.5.
        parts = name[len("mse_stdmm_") :].split("_")
        if len(parts) != 2:
            raise ValueError(f"Bad mse_stdmm name: {name}")
        lam_std = float(parts[0]) / 100.0
        lam_mar = float(parts[1]) / 100.0
        return _MSEStdMatchMaxMargin(lam_std=lam_std, lam_mar=lam_mar)
    if name.startswith("mse_gumbel_tw_"):
        # Format: "mse_gumbel_tw_AA_KK" with AA = alpha × 10, KK = k × 10.
        # Example: mse_gumbel_tw_05_30 → alpha = 0.5, k = 3.0.
        parts = name[len("mse_gumbel_tw_") :].split("_")
        if len(parts) != 2:
            raise ValueError(f"Bad mse_gumbel_tw name: {name}")
        alpha = float(parts[0]) / 10.0
        k = float(parts[1]) / 10.0
        return _MSEGumbelTailWeighted(alpha=alpha, k=k)
    if name.startswith("mse_gumbel_"):
        # Format: "mse_gumbel_LB_LM" with LB = lam_b × 100, LM = lam_m × 100.
        # Example: mse_gumbel_50_50 → lam_b = 0.50, lam_m = 0.50.
        parts = name[len("mse_gumbel_") :].split("_")
        if len(parts) != 2:
            raise ValueError(f"Bad mse_gumbel name: {name}")
        lam_b = float(parts[0]) / 100.0
        lam_m = float(parts[1]) / 100.0
        return _MSEGumbelMomentMatch(lam_b=lam_b, lam_m=lam_m)
    if name.startswith("peak_huber_topk_"):
        parts = name[len("peak_huber_topk_") :].split("_")
        if len(parts) == 1:
            alpha = float(parts[0]) / 10.0
            beta = (1.0 - alpha) / 2.0
            gamma = (1.0 - alpha) / 2.0
        elif len(parts) == 2:
            alpha = float(parts[0]) / 10.0
            gamma = float(parts[1]) / 100.0
            beta = 1.0 - alpha - gamma
        else:
            raise ValueError(f"Bad peak_huber_topk name: {name}")
        return _PeakHuberTopKLoss(alpha=alpha, beta=beta, gamma=gamma)
    if name.startswith("mc_huber_") or name.startswith("rank_topk_mc_"):
        # Multi-channel-aware Huber + per-graph rank/topk/stdmatch on
        # designated MAX channels.
        #
        # Format A (3-axis):  "mc_huber_RR_TT_SS[_chK1,K2,...]"
        #   RR = rank weight × 100        (rank focuses on top decile of
        #                                   each graph's true FIP)
        #   TT = topk weight × 100        (top-K hot-spot Huber upweight)
        #   SS = stdmatch weight × 100    (v5-style L1 |Δσ| per-graph)
        #   ch = optional comma-separated channel list (default [0])
        #
        # Format B (legacy 2-axis): "rank_topk_mc_RR_TT[_chK1,K2,...]"
        #   stdmatch implied 0.
        #
        # Examples:
        #   "mc_huber_30_00_00_ch0,1,2,3"    → λ_rank=0.3, ch=4 MAX
        #   "mc_huber_00_30_00_ch0,1,2,3"    → λ_topk=0.3, ch=4 MAX
        #   "mc_huber_00_00_75_ch0,1,2,3"    → v5-style stdmatch λ=0.75, 4 MAX
        #   "mc_huber_30_00_75_ch0,1,2,3"    → rank+stdmatch combo
        #   "rank_topk_mc_15_15_ch0,1,2,3"   → λ_rank=0.15 + λ_topk=0.15
        if name.startswith("mc_huber_"):
            rest = name[len("mc_huber_") :]
            n_axes = 3
        else:
            rest = name[len("rank_topk_mc_") :]
            n_axes = 2
        if "_ch" in rest:
            head, ch_str = rest.split("_ch", 1)
            ch_idx = [int(x) for x in ch_str.split(",") if x]
        else:
            head = rest
            ch_idx = [0]
        parts = head.split("_")
        if len(parts) != n_axes:
            raise ValueError(f"Bad {name.split('_')[0]} name: {name}")
        lam_rank = float(parts[0]) / 100.0
        lam_topk = float(parts[1]) / 100.0
        lam_stdmatch = float(parts[2]) / 100.0 if n_axes == 3 else 0.0
        return _RankTopKHuberMC(
            lam_rank=lam_rank,
            lam_topk=lam_topk,
            lam_stdmatch=lam_stdmatch,
            rank_channel_indices=ch_idx,
        )
    if name.startswith("w2_huber_"):
        # Format: "w2_huber_LL[_chK1,K2,...]" — λ = LL/100; optional channel
        # list overrides the default of slot-0 only. Examples:
        #   "w2_huber_05"            → λ=0.5, MAX channels=[0]
        #   "w2_huber_05_ch0,1,2,3"  → λ=0.5, MAX channels=[0,1,2,3]
        # MAX channels parsed at recipe-build time and passed into the
        # _WassersteinHuber constructor.
        rest = name[len("w2_huber_") :]
        if "_ch" in rest:
            lam_str, ch_str = rest.split("_ch", 1)
            max_idx = [int(x) for x in ch_str.split(",") if x]
        else:
            lam_str = rest
            max_idx = [0]
        lam = float(lam_str) / 100.0
        return _WassersteinHuber(lam=lam, max_channel_indices=max_idx)
    raise ValueError(f"Unknown loss recipe: {name}")


def install_batch_stash_hook(model: torch.nn.Module) -> None:
    """Forward pre-hook that stashes data.batch into _CURRENT_BATCH for the
    per-graph losses (rank, topk, peak)."""

    def hook(_module, args, _kwargs=None):
        if args and hasattr(args[0], "batch"):
            _CURRENT_BATCH["batch"] = args[0].batch
        return None

    model.register_forward_pre_hook(hook, with_kwargs=False)


def override_model_loss(model: torch.nn.Module, loss_recipe: Optional[str]) -> None:
    """Replace model.loss_function with our custom callable AFTER model build.

    Bypasses nn.Module.__setattr__ via object.__setattr__ + _modules cleanup,
    same trick anik used.
    """
    if not loss_recipe or loss_recipe in ("default", "huber"):
        return
    inner = model.module if hasattr(model, "module") else model
    loss_obj = build_custom_loss(loss_recipe)
    if "loss_function" in inner._modules:
        del inner._modules["loss_function"]
    object.__setattr__(inner, "loss_function", loss_obj)
    inner.loss_function_type = loss_recipe
    print(
        f"[loss override] model.loss_function -> {type(loss_obj).__name__}  (recipe={loss_recipe})",
        flush=True,
    )
