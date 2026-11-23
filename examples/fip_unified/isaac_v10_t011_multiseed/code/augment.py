"""FCC cubic-symmetry augmentation wrapper.

Each grain has a quaternion in `data.x[:, 3:7]`. Under FCC cubic symmetry,
24 unit quaternions represent the same physical orientation (the proper
rotation group O of order 24). Multiplying each grain's quaternion by a
randomly-sampled cubic operator gives a different representation of the
same physical microstructure — the per-grain orientation features change,
but the per-grain Schmid factor / Taylor factor / slip-transfer m' /
elastic mismatch are mathematically invariant.

Edge features that DO change (because they're computed from a single
quaternion-pair representative, not the full disorientation):
  - misorientation_angle   (col 0)
  - relpos_crystal_x/y/z   (cols 2-4)

These get recomputed on-the-fly. Slip-transfer m' (col 5), elastic_mismatch
(col 6) are invariant under cubic rotation of either grain → unchanged.
edge_length (col 1), is_periodic (col 7) are unaffected.

Wraps a list[Data] with a torch Dataset that applies augmentation
on __getitem__; only used at training time.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# 24-element FCC proper cubic rotation group (point group O).
# Adapted from cp3d/train_pre.py:_build_fcc_cubic_group_quats.
def _build_fcc_24_quats() -> torch.Tensor:
    s2 = 1.0 / math.sqrt(2.0)
    h = 0.5
    quats: List[List[float]] = []
    quats.append([1.0, 0.0, 0.0, 0.0])
    for axis in range(3):
        for sign in (+1.0, -1.0):
            q = [s2, 0.0, 0.0, 0.0]
            q[axis + 1] = sign * s2
            quats.append(q)
        q = [0.0, 0.0, 0.0, 0.0]
        q[axis + 1] = 1.0
        quats.append(q)
    for sx in (+1.0, -1.0):
        for sy in (+1.0, -1.0):
            for sz in (+1.0, -1.0):
                quats.append([h, sx * h, sy * h, sz * h])
    face_axes = [
        (s2, s2, 0.0),
        (s2, -s2, 0.0),
        (s2, 0.0, s2),
        (s2, 0.0, -s2),
        (0.0, s2, s2),
        (0.0, s2, -s2),
    ]
    for (x, y, z) in face_axes:
        quats.append([0.0, x, y, z])
    Q = torch.tensor(quats, dtype=torch.float32)
    Q = Q / Q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return Q


FCC_24_QUAT = _build_fcc_24_quats()  # (24, 4)


def _random_so3_quat() -> torch.Tensor:
    """Sample a unit quaternion uniformly from SO(3) via normalized Gaussian."""
    q = torch.randn(4)
    return q / q.norm().clamp_min(1e-12)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product. Both shape (..., 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1
    )
    row1 = torch.stack(
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1
    )
    row2 = torch.stack(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def _edge_miso(q: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    row, col = edge_index
    dot = (q[row] * q[col]).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def _edge_relpos_crystal(
    q: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    row, col = edge_index
    d = pos[col] - pos[row]
    d_norm = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    R = _quat_to_rotmat(q[row])
    return torch.einsum("nij,nj->ni", R, d_norm)


class SO3AugmentDataset(Dataset):
    """Train-time SO(3) sample-frame augmentation.

    Samples ONE random unit quaternion `q_global` per __getitem__ call and
    rotates the whole microstructure by it: q_new = q_global ∘ q_old (PRE-mul,
    same group action as a sample-frame rotation). Per-grain orientations
    move; the relative orientations between grains are PRESERVED — so
    misorientation, m', elastic-mismatch are invariant. pos rotates with
    the same quaternion to keep crystal-frame relpos consistent.

    Anik tested this on corrupted orientations (where every quat was
    near-identity) and val ρ collapsed 0.55 → 0.29. On v2 with proper
    orientations, augmentation might give the model genuine rotation
    invariance training signal.
    """

    def __init__(self, base, p: float = 1.0, seed: Optional[int] = None):
        self.base = base
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        d = self.base[idx]
        if self.rng.random() >= self.p:
            return d
        # ONE quaternion for the whole VE (sample-frame rotation).
        q_glob = torch.tensor(self.rng.normal(size=4), dtype=d.x.dtype)
        q_glob = q_glob / q_glob.norm().clamp_min(1e-12)
        N = int(d.x.shape[0])
        g = q_glob.unsqueeze(0).expand(N, 4)

        q_old = d.x[:, 3:7]
        q_new = _quat_mul(g, q_old)
        q_new = q_new / q_new.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Rotate pos by R(q_glob) so crystal-relpos stays consistent.
        # Pos is in sample frame → rotate by R(q_glob) gives the new sample frame.
        R = _quat_to_rotmat(q_glob)
        pos_new = torch.einsum("ij,nj->ni", R, d.pos)

        d2 = d.clone()
        d2.x = d.x.clone()
        d2.x[:, 3:7] = q_new
        d2.pos = pos_new
        # edge_attr: misorientation, m', emis are invariant. crystal_relpos uses
        # row's quat & pos; both rotated consistently → invariant. Don't touch.
        return d2


class FCCSymAugmentDataset(Dataset):
    """Wraps a list[Data]; on __getitem__ applies a random cubic-symmetry
    rotation per grain and recomputes the affected edge_attr columns.

    Edge_attr layout (8 cols) preserved:
      col 0: misorientation_angle  (RECOMPUTED)
      col 1: edge_length           (unchanged)
      col 2-4: relpos_crystal_xyz  (RECOMPUTED)
      col 5: slip_transfer_mprime  (invariant under cubic sym → unchanged)
      col 6: elastic_mismatch      (invariant → unchanged)
      col 7: is_periodic           (unchanged)

    x[:, 0:3] (Euler angles) are NOT recomputed — they're an alternative
    representation that the trainer typically doesn't use after quaternions.
    """

    def __init__(self, base: List, p: float = 1.0, seed: Optional[int] = None):
        """`base` is a list[Data] (already split + scaled by FIPDataset).
        `p` is the augmentation probability per epoch (per call to __getitem__).
        """
        self.base = base
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        d = self.base[idx]
        if self.rng.random() >= self.p:
            return d  # no augment this call

        # Sample one cubic op per grain.
        N = int(d.x.shape[0])
        picks = torch.from_numpy(self.rng.integers(0, 24, size=N).astype(np.int64))
        g = FCC_24_QUAT[picks].to(dtype=d.x.dtype)  # (N, 4)

        # Old + new quaternion (PRE-multiply: q_new = g ∘ q_old, crystal-frame symmetry).
        q_old = d.x[:, 3:7]
        q_new = _quat_mul(g, q_old)
        q_new = q_new / q_new.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Build a NEW Data with mutated x and recomputed edge_attr columns.
        # NOTE: we clone so the underlying base list isn't modified.
        d2 = d.clone()
        d2.x = d.x.clone()
        d2.x[:, 3:7] = q_new

        # Recompute miso + crystal_relpos.
        if d.edge_attr is not None and d.edge_attr.numel() > 0:
            ea = d.edge_attr.clone()
            ea[:, 0] = _edge_miso(q_new, d.edge_index)
            ea[:, 2:5] = _edge_relpos_crystal(q_new, d.pos, d.edge_index)
            d2.edge_attr = ea
        return d2
