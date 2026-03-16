import torch
import math
from torch_geometric.data import Data

def _quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions in (w, x, y, z) convention."""
    qw, qx, qy, qz = q.unbind(dim=-1)
    rw, rx, ry, rz = r.unbind(dim=-1)
    return torch.stack(
        [
            qw * rw - qx * rx - qy * ry - qz * rz,
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
        ],
        dim=-1,
    )

def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for (w, x, y, z)."""
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def euler_bunge_zxz_to_quat(euler: torch.Tensor, degrees: bool = True) -> torch.Tensor:
    """
    Convert Bunge Euler angles (phi1, Phi, phi2) with intrinsic ZXZ rotations to unit quaternions.

    Returns quaternion in (w, x, y, z) convention.
    """
    if degrees:
        euler = euler * (math.pi / 180.0)
    phi1, Phi, phi2 = euler.unbind(dim=-1)
    half = 0.5
    z1 = torch.stack([torch.cos(half * phi1), torch.zeros_like(phi1), torch.zeros_like(phi1), torch.sin(half * phi1)], dim=-1)
    x = torch.stack([torch.cos(half * Phi), torch.sin(half * Phi), torch.zeros_like(Phi), torch.zeros_like(Phi)], dim=-1)
    z2 = torch.stack([torch.cos(half * phi2), torch.zeros_like(phi2), torch.zeros_like(phi2), torch.sin(half * phi2)], dim=-1)
    q = _quat_mul(_quat_mul(z1, x), z2)
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    return q


class EulerAngleMismatch:
    """
    Add an edge feature based on mismatch between Euler angles of the two endpoint grains.

    This is a simple circular difference in angle space (handles 0/360 wrap if degrees=True).
    If `data.euler` exists it will be used; otherwise uses `data.x[:, euler_cols]`.
    """

    def __init__(
        self,
        euler_cols=(0, 1, 2),
        degrees: bool = True,
        reduce: str = "l2",  # "l2", "sum", or "none"
        keep_abs_deltas: bool = False,  # if True, include |dphi1|, |dPhi|, |dphi2| in edge_attr
    ):
        self.euler_cols = tuple(euler_cols)
        self.degrees = degrees
        self.reduce = reduce
        self.keep_abs_deltas = keep_abs_deltas

    def __call__(self, data: Data) -> Data:
        if getattr(data, "edge_index", None) is None:
            return data

        if hasattr(data, "euler") and data.euler is not None:
            euler = data.euler
        else:
            if getattr(data, "x", None) is None:
                return data
            euler = data.x[:, list(self.euler_cols)]

        row, col = data.edge_index
        diff = euler[col] - euler[row]

        period = 360.0 if self.degrees else (2.0 * math.pi)
        half = period / 2.0
        # wrap to [-period/2, period/2)
        diff = (diff + half) % period - half
        absdiff = diff.abs()

        if self.reduce == "l2":
            mismatch = torch.linalg.norm(absdiff, dim=1, keepdim=True)
        elif self.reduce == "sum":
            mismatch = absdiff.sum(dim=1, keepdim=True)
        elif self.reduce == "none":
            mismatch = absdiff
        else:
            raise ValueError(f"Unknown reduce={self.reduce} (expected 'l2', 'sum', or 'none')")

        feat = torch.cat([absdiff, mismatch], dim=1) if self.keep_abs_deltas and mismatch.shape[1] == 1 else mismatch

        if getattr(data, "edge_attr", None) is None:
            data.edge_attr = feat
        else:
            data.edge_attr = torch.cat([data.edge_attr, feat], dim=1)
        return data


class QuaternionAngleMismatch:
    """
    Add an edge feature based on angular distance between unit quaternions of endpoint grains.

    Angle = 2*acos(|dot(q_i, q_j)|)  in radians, in [0, pi].
    Uses `data.quat` if present; otherwise uses `data.x[:, quat_cols]`.
    """

    def __init__(self, quat_cols=(0, 1, 2, 3)):
        self.quat_cols = tuple(quat_cols)

    def __call__(self, data: Data) -> Data:
        if getattr(data, "edge_index", None) is None:
            return data

        if hasattr(data, "quat") and data.quat is not None:
            quat = data.quat
        else:
            if getattr(data, "x", None) is None:
                return data
            quat = data.x[:, list(self.quat_cols)]

        quat = quat / (quat.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        row, col = data.edge_index
        dot = (quat[row] * quat[col]).sum(dim=-1).abs().clamp(-1.0, 1.0)
        angle = 2.0 * torch.acos(dot)
        feat = angle.view(-1, 1)

        if getattr(data, "edge_attr", None) is None:
            data.edge_attr = feat
        else:
            data.edge_attr = torch.cat([data.edge_attr, feat], dim=1)
        return data


class QuaternionRelativeRotation:
    """
    Add edge features representing the full relative rotation between endpoint unit quaternions.

    For edge (i -> j), computes the relative quaternion:

        q_rel = q_j ⊗ q_i^{-1}

    where inverse is conjugate for unit quaternions, and ⊗ is the Hamilton product
    in (w, x, y, z) convention.

    The result is canonicalized to remove the q vs -q ambiguity by enforcing w >= 0.
    Uses `data.quat` if present; otherwise uses `data.x[:, quat_cols]`.
    """

    def __init__(self, quat_cols=(0, 1, 2, 3), canonicalize: bool = True):
        self.quat_cols = tuple(quat_cols)
        self.canonicalize = canonicalize

    def __call__(self, data: Data) -> Data:
        if getattr(data, "edge_index", None) is None:
            return data

        if hasattr(data, "quat") and data.quat is not None:
            quat = data.quat
        else:
            if getattr(data, "x", None) is None:
                return data
            quat = data.x[:, list(self.quat_cols)]

        quat = quat / (quat.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        row, col = data.edge_index
        q_i = quat[row]
        q_j = quat[col]

        q_rel = _quat_mul(q_j, _quat_conj(q_i))
        q_rel = q_rel / (q_rel.norm(dim=-1, keepdim=True).clamp_min(1e-12))

        if self.canonicalize:
            # Remove q vs -q ambiguity by forcing the first non-zero component (w,x,y,z) to be positive.
            # This is stricter than enforcing w>=0, which does not disambiguate 180° rotations (w≈0).
            eps = 1e-12
            sign = torch.ones((q_rel.shape[0], 1), device=q_rel.device, dtype=q_rel.dtype)
            mask0 = q_rel[:, 0].abs() <= eps
            mask1 = mask0 & (q_rel[:, 1].abs() <= eps)
            mask2 = mask1 & (q_rel[:, 2].abs() <= eps)

            sign[q_rel[:, 0] < -eps] = -1.0
            sign[mask0 & (q_rel[:, 1] < -eps)] = -1.0
            sign[mask1 & (q_rel[:, 2] < -eps)] = -1.0
            sign[mask2 & (q_rel[:, 3] < -eps)] = -1.0
            q_rel = q_rel * sign

        feat = q_rel  # (E, 4)

        if getattr(data, "edge_attr", None) is None:
            data.edge_attr = feat
        else:
            data.edge_attr = torch.cat([data.edge_attr, feat], dim=1)
        return data


class PhasePairCode:
    """
    Add a categorical-ish edge feature based on whether endpoint node phases match.

    For an edge (i -> j), with integer phases p_i, p_j, produces a single scalar:
    - if p_i != p_j: feature = 0
    - if p_i == p_j == phase_A: feature = -1
    - if p_i == p_j == phase_B: feature = +1

    Phase IDs are auto-detected to handle common encodings:
    - if max(phase) <= 1: treat phases as {0,1} with A=0, B=1
    - elif min(phase) >= 1 and max(phase) >= 2: treat phases as {1,2} with A=1, B=2
    - else: fall back to A=min(unique_phases), B=max(unique_phases)

    Uses `data.phase` if present; otherwise uses `data.x[:, phase_col]`.
    """

    def __init__(self, phase_col: int = 3):
        self.phase_col = int(phase_col)

    def __call__(self, data: Data) -> Data:
        if getattr(data, "edge_index", None) is None:
            return data

        if hasattr(data, "phase") and data.phase is not None:
            phase = data.phase
        else:
            if getattr(data, "x", None) is None:
                return data
            phase = data.x[:, self.phase_col]

        # Be robust to floats representing integer phases.
        phase = phase.view(-1).to(torch.float32).round().to(torch.long)
        # Auto-detect the two phase IDs (most common cases: {0,1} or {1,2}).
        pmin = int(phase.min().item())
        pmax = int(phase.max().item())
        if pmax <= 1:
            phase_A, phase_B = 0, 1
        elif pmin >= 1 and pmax >= 2:
            phase_A, phase_B = 1, 2
        else:
            uniq = torch.unique(phase)
            phase_A = int(uniq.min().item())
            phase_B = int(uniq.max().item())

        row, col = data.edge_index
        pi = phase[row]
        pj = phase[col]

        feat = torch.zeros((pi.shape[0],), dtype=torch.float32, device=pi.device)
        same = pi == pj
        feat[same & (pi == phase_A)] = -1.0
        if phase_B != phase_A:
            feat[same & (pi == phase_B)] = 1.0
        feat = feat.view(-1, 1)

        if getattr(data, "edge_attr", None) is None:
            data.edge_attr = feat
        else:
            data.edge_attr = torch.cat([data.edge_attr, feat], dim=1)
        return data