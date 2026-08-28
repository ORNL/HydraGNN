"""Spectral positional data for heterogeneous OPF models.

The GridOpt/PyG OPF feature layout used here is:

* AC line: ``[angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]``
* Transformer: ``[angmin, angmax, r, x, rate_a, rate_b, rate_c,
  tap, shift, g_sh, b_sh]``
* Shunt node: ``[b_sh, g_sh]``

All quantities are already in per-unit/radians.  The resulting Ybus is kept
complex and non-symmetric; in particular, phase-shifting transformers produce
different from-to and to-from entries.

In addition to the existing Ybus SVD relative positional encoding, this module
provides bus-level input positional encodings and direct pairwise attention
biases:

* eigenpairs of the unweighted topological bus Laplacian; and
* diagonal-free effective-resistance statistics;
* diagonal-free effective-impedance statistics over the real and imaginary
  components separately; and
* effective-resistance and effective-impedance matrices consumed by a learned
  attention-bias MLP.

The expensive graph-level calculations can be cached on disk.  This matters
for the OPF datasets because every operating-point sample in a fixed-topology
case repeats the same network matrices.
"""

import copy
import fcntl
import hashlib
import json
import os
import re

import torch


_BUS_BRANCH_TYPES = (
    (("bus", "ac_line", "bus"), False),
    (("bus", "transformer", "bus"), True),
)
_PE_SOURCE_ALIASES = {
    "laplacian": "laplacian",
    "lpe": "laplacian",
    "topological_laplacian": "laplacian",
    "effective_resistance": "effective_resistance",
    "resistance": "effective_resistance",
    "er": "effective_resistance",
    "dc_summary": "effective_resistance",
    "effective_impedance": "effective_impedance",
    "impedance": "effective_impedance",
    "ac_summary": "effective_impedance",
    "effective_resistance_rpe": "effective_resistance_rpe",
    "resistance_rpe": "effective_resistance_rpe",
    "er_rpe": "effective_resistance_rpe",
    "effective_resistance_qk": "effective_resistance_qk",
    "resistance_qk": "effective_resistance_qk",
    "er_qk": "effective_resistance_qk",
    "effective_impedance_rpe": "effective_impedance_rpe",
    "impedance_rpe": "effective_impedance_rpe",
    "ei_rpe": "effective_impedance_rpe",
    "ybus_svd": "ybus_svd",
    "svd_ybus": "ybus_svd",
}
_CACHE_VERSION = "opf-spectral-pe-v1"


def _canonical_sources(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    result = []
    for value in values:
        key = str(value).lower()
        if key not in _PE_SOURCE_ALIASES:
            raise ValueError(
                f"Unknown OPF positional encoding source '{value}'. Expected one "
                "of: laplacian, effective_resistance, effective_impedance, "
                "effective_resistance_rpe, effective_resistance_qk, "
                "effective_impedance_rpe, ybus_svd."
            )
        canonical = _PE_SOURCE_ALIASES[key]
        if canonical not in result:
            result.append(canonical)
    return result


def resolve_opf_positional_encoding_config(architecture_config):
    """Return a validated, backward-compatible OPF PE configuration."""

    arch = architecture_config or {}
    nested = arch.get("positional_encodings")
    if nested is None:
        nested = {}
    elif not isinstance(nested, dict):
        raise TypeError("Architecture.positional_encodings must be a dictionary.")
    config = copy.deepcopy(nested)

    # Preserve the original SVD-RPE configuration format.
    legacy_encoder = str(arch.get("pe_encoder", "none")).lower()
    legacy_svd = legacy_encoder in {"svd_ybus", "ybus_svd"}
    if not config and legacy_svd and int(arch.get("pe_dim", 0)) > 0:
        config = {
            "precompute": ["ybus_svd"],
            "use": ["ybus_svd"],
            "ybus_svd": {
                "dim": int(arch["pe_dim"]),
                "relative_tolerance": arch.get("svd_rpe_tolerance"),
            },
        }

    use = _canonical_sources(config.get("use", []))
    precompute = _canonical_sources(config.get("precompute", use))
    missing = set(use) - set(precompute)
    if missing:
        raise ValueError(
            "Every active positional encoding must also be precomputed; missing "
            f"from positional_encodings.precompute: {sorted(missing)}"
        )

    laplacian = copy.deepcopy(config.get("laplacian", {}))
    laplacian.setdefault("dim", 8)
    laplacian.setdefault("eigenvector_selection", "smallest_nonzero")
    laplacian.setdefault("relative_tolerance", None)
    laplacian.setdefault("random_sign_flip", False)
    if int(laplacian["dim"]) <= 0 and "laplacian" in precompute:
        raise ValueError("positional_encodings.laplacian.dim must be positive.")
    if str(laplacian["eigenvector_selection"]).lower() != "smallest_nonzero":
        raise ValueError(
            "Only laplacian.eigenvector_selection='smallest_nonzero' is supported."
        )

    resistance = copy.deepcopy(config.get("effective_resistance", {}))
    resistance.setdefault("statistics", ["min", "max", "std", "median", "mean"])
    resistance.setdefault("std_correction", 0)
    resistance.setdefault("exclude_diagonal", True)
    expected_stats = ["min", "max", "std", "median", "mean"]
    if list(resistance["statistics"]) != expected_stats:
        raise ValueError(
            "The paper-style effective-resistance approximation requires statistics "
            f"in this order: {expected_stats}."
        )
    if not bool(resistance["exclude_diagonal"]):
        raise ValueError(
            "effective_resistance.exclude_diagonal must be true for the "
            "diagonal-free OPF summary."
        )

    impedance = copy.deepcopy(config.get("effective_impedance", {}))
    impedance.setdefault("statistics", expected_stats)
    impedance.setdefault("components", ["real", "imag"])
    impedance.setdefault("std_correction", 0)
    impedance.setdefault("exclude_diagonal", True)
    impedance.setdefault("relative_tolerance", None)
    if list(impedance["statistics"]) != expected_stats:
        raise ValueError(
            "The effective-impedance summary requires statistics in this "
            f"order: {expected_stats}."
        )
    if list(impedance["components"]) != ["real", "imag"]:
        raise ValueError(
            "effective_impedance.components must be ['real', 'imag']."
        )
    if not bool(impedance["exclude_diagonal"]):
        raise ValueError(
            "effective_impedance.exclude_diagonal must be true for the "
            "diagonal-free OPF summary."
        )

    resistance_rpe = copy.deepcopy(config.get("effective_resistance_rpe", {}))
    resistance_rpe.setdefault("feature_dim", 1)
    resistance_rpe.setdefault("mlp_hidden_dim", 8)
    resistance_rpe.setdefault("zero_diagonal_bias", True)
    if int(resistance_rpe["feature_dim"]) != 1:
        raise ValueError("effective_resistance_rpe.feature_dim must be 1.")
    if int(resistance_rpe["mlp_hidden_dim"]) <= 0:
        raise ValueError("effective_resistance_rpe.mlp_hidden_dim must be positive.")

    resistance_qk = copy.deepcopy(config.get("effective_resistance_qk", {}))
    resistance_qk.setdefault("dim", 8)
    resistance_qk.setdefault("relative_tolerance", None)
    resistance_qk.setdefault("coefficient_init", 0.0)
    resistance_qk.setdefault("placement", "qk")
    if int(resistance_qk["dim"]) <= 0 and "effective_resistance_qk" in precompute:
        raise ValueError("effective_resistance_qk.dim must be positive.")
    relative_tolerance = resistance_qk.get("relative_tolerance")
    if relative_tolerance is not None and float(relative_tolerance) < 0.0:
        raise ValueError(
            "effective_resistance_qk.relative_tolerance must be non-negative."
        )
    placement = str(resistance_qk["placement"]).lower()
    if placement not in {"input", "qk", "both"}:
        raise ValueError(
            "effective_resistance_qk.placement must be 'input', 'qk', or 'both'."
        )
    resistance_qk["placement"] = placement

    impedance_rpe = copy.deepcopy(config.get("effective_impedance_rpe", {}))
    impedance_rpe.setdefault("feature_dim", 2)
    impedance_rpe.setdefault("mlp_hidden_dim", 8)
    impedance_rpe.setdefault("zero_diagonal_bias", True)
    impedance_rpe.setdefault("relative_tolerance", None)
    if int(impedance_rpe["feature_dim"]) != 2:
        raise ValueError("effective_impedance_rpe.feature_dim must be 2.")
    if int(impedance_rpe["mlp_hidden_dim"]) <= 0:
        raise ValueError("effective_impedance_rpe.mlp_hidden_dim must be positive.")

    ybus_svd = copy.deepcopy(config.get("ybus_svd", {}))
    ybus_svd.setdefault("dim", int(arch.get("pe_dim", 8) or 8))
    ybus_svd.setdefault("relative_tolerance", arch.get("svd_rpe_tolerance"))
    if int(ybus_svd["dim"]) <= 0 and "ybus_svd" in precompute:
        raise ValueError("positional_encodings.ybus_svd.dim must be positive.")

    active_rpe = set(use) & {
        "ybus_svd",
        "effective_resistance_rpe",
        "effective_resistance_qk",
        "effective_impedance_rpe",
    }
    if len(active_rpe) > 1:
        raise ValueError(
            "Only one attention RPE can be active at a time; got "
            f"{sorted(active_rpe)}."
        )

    return {
        "precompute": precompute,
        "use": use,
        "cache_by_case": bool(config.get("cache_by_case", False)),
        "compute_device": str(config.get("compute_device", "auto")).lower(),
        "laplacian": laplacian,
        "effective_resistance": resistance,
        "effective_impedance": impedance,
        "effective_resistance_rpe": resistance_rpe,
        "effective_resistance_qk": resistance_qk,
        "effective_impedance_rpe": impedance_rpe,
        "ybus_svd": ybus_svd,
    }


def _edge_store(data, edge_type):
    if edge_type not in data.edge_types:
        return None
    return data[edge_type]


def _resolve_compute_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Spectral PE compute_device requests CUDA, but it is unavailable.")
    return device


def _add_laplacian_branches(matrix, data, weighted):
    """Accumulate symmetric branch contributions into a bus Laplacian."""

    for edge_type, transformer in _BUS_BRANCH_TYPES:
        store = _edge_store(data, edge_type)
        if store is None or store.edge_index.numel() == 0:
            continue
        edge_index = store.edge_index.to(device=matrix.device, dtype=torch.long)
        src, dst = edge_index[0], edge_index[1]
        if weighted:
            attr = store.edge_attr.to(device=matrix.device, dtype=matrix.dtype)
            x_index = 3 if transformer else 5
            reactance = attr[:, x_index].abs()
            eps = torch.finfo(matrix.dtype).eps
            if torch.any(reactance <= eps):
                raise ValueError(f"Zero branch reactance encountered for {edge_type}.")
            weight = reactance.reciprocal()
            if transformer:
                tap = attr[:, 7].abs()
                tap = torch.where(tap > eps, tap, torch.ones_like(tap))
                weight = weight / tap
        else:
            weight = torch.ones(src.numel(), dtype=matrix.dtype, device=matrix.device)

        matrix.index_put_((src, src), weight, accumulate=True)
        matrix.index_put_((dst, dst), weight, accumulate=True)
        matrix.index_put_((src, dst), -weight, accumulate=True)
        matrix.index_put_((dst, src), -weight, accumulate=True)


def build_topological_laplacian(data, dtype=torch.float64, device=None):
    """Build the unweighted combinatorial bus Laplacian ``D - A``."""

    num_bus = int(data["bus"].x.size(0))
    laplacian = torch.zeros((num_bus, num_bus), dtype=dtype, device=device)
    _add_laplacian_branches(laplacian, data, weighted=False)
    return laplacian


def build_susceptance_laplacian(data, dtype=torch.float64, device=None):
    """Build the paper's DC susceptance-weighted bus Laplacian.

    AC lines use ``1 / |x|`` and transformers use ``1 / (|x| |tap|)``.
    The matrix is deliberately symmetric and has zero row sums, matching the
    weighted-Laplacian definition in the paper rather than the full AC Ybus.
    """

    num_bus = int(data["bus"].x.size(0))
    laplacian = torch.zeros((num_bus, num_bus), dtype=dtype, device=device)
    _add_laplacian_branches(laplacian, data, weighted=True)
    return laplacian


def _eigenvalue_tolerance(values, matrix_size, relative_tolerance=None):
    if values.numel() == 0:
        return 0.0
    scale = values.abs().max()
    if relative_tolerance is not None:
        return float(scale) * float(relative_tolerance)
    return float(scale) * matrix_size * torch.finfo(values.dtype).eps


def compute_topological_laplacian_pe(
    data,
    k,
    relative_tolerance=None,
    compute_device="auto",
):
    """Compute the ``k`` smallest nonzero topological Laplacian eigenpairs."""

    k = int(k)
    num_bus = int(data["bus"].x.size(0))
    if k <= 0:
        raise ValueError("Laplacian PE dimension must be positive.")
    if num_bus == 0:
        return {
            "lap_eigvec": torch.empty((0, k), dtype=torch.float32),
            "lap_eigval": torch.zeros((1, k), dtype=torch.float32),
        }

    device = _resolve_compute_device(compute_device)
    laplacian = build_topological_laplacian(data, device=device)
    values, vectors = torch.linalg.eigh(laplacian)
    tolerance = _eigenvalue_tolerance(values, num_bus, relative_tolerance)
    selected = torch.nonzero(values > tolerance, as_tuple=False).flatten()[:k]

    eigvec = torch.zeros((num_bus, k), dtype=torch.float64, device=device)
    eigval = torch.zeros((k,), dtype=torch.float64, device=device)
    count = int(selected.numel())
    if count:
        eigvec[:, :count] = vectors[:, selected]
        eigval[:count] = values[selected]
    return {
        "lap_eigvec": eigvec.float().cpu(),
        "lap_eigval": eigval.float().view(1, k).cpu(),
    }


def _num_bus_components(data):
    num_bus = int(data["bus"].x.size(0))
    parent = list(range(num_bus))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge_type, _ in _BUS_BRANCH_TYPES:
        store = _edge_store(data, edge_type)
        if store is None:
            continue
        for left, right in store.edge_index.detach().cpu().t().tolist():
            union(int(left), int(right))
    return len({find(node) for node in range(num_bus)}) if num_bus else 0


def compute_effective_resistance_qk(
    data,
    k,
    relative_tolerance=None,
    compute_device="auto",
):
    """Return truncated coordinates whose squared distances approximate resistance.

    For the susceptance-weighted admittance Laplacian ``L = U diag(lambda) U^T``,
    the coordinates are ``X = U[:, 1:k+1] / sqrt(lambda[1:k+1])``. Therefore,
    ``||X[i] - X[j]||^2`` is the rank-limited effective resistance. The dense
    pairwise resistance matrix is never formed.
    """

    k = int(k)
    num_bus = int(data["bus"].x.size(0))
    if k <= 0:
        raise ValueError("Effective-resistance Q/K dimension must be positive.")
    if num_bus == 0:
        return {
            "effective_resistance_qk": torch.empty((0, k), dtype=torch.float32)
        }
    components = _num_bus_components(data)
    if components != 1:
        raise ValueError(
            "Effective resistance is undefined between disconnected components; "
            f"the bus graph has {components} components."
        )

    device = _resolve_compute_device(compute_device)
    laplacian = build_susceptance_laplacian(data, device=device)
    values, vectors = torch.linalg.eigh(laplacian)
    tolerance = _eigenvalue_tolerance(values, num_bus, relative_tolerance)
    selected = torch.nonzero(values > tolerance, as_tuple=False).flatten()[:k]

    coordinates = torch.zeros(
        (num_bus, k), dtype=laplacian.dtype, device=device
    )
    count = int(selected.numel())
    if count:
        selected_values = values[selected]
        coordinates[:, :count] = vectors[:, selected] * selected_values.rsqrt()
    return {"effective_resistance_qk": coordinates.float().cpu()}


def compute_effective_resistance_matrix(data, compute_device="auto"):
    """Return the dense DC effective-resistance matrix for one grid."""

    num_bus = int(data["bus"].x.size(0))
    if num_bus == 0:
        return torch.empty((0, 0), dtype=torch.float64)
    components = _num_bus_components(data)
    if components != 1:
        raise ValueError(
            "Effective resistance is undefined between disconnected components; "
            f"the bus graph has {components} components."
        )

    device = _resolve_compute_device(compute_device)
    weighted_laplacian = build_susceptance_laplacian(data, device=device)
    centering = torch.full(
        (num_bus, num_bus),
        1.0 / float(num_bus),
        dtype=weighted_laplacian.dtype,
        device=device,
    )
    # For a connected Laplacian: Q+ = (Q + 11^T/n)^-1 - 11^T/n.
    q_pinv = torch.linalg.inv(weighted_laplacian + centering) - centering
    diagonal = torch.diagonal(q_pinv)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * q_pinv
    resistance.clamp_min_(0.0)
    resistance.fill_diagonal_(0.0)
    return resistance


def _offdiagonal_rows(matrix):
    """Return each row with its diagonal entry removed."""

    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError(
            "Expected a square pairwise matrix, got "
            f"shape {tuple(matrix.shape)}."
        )
    num_bus = matrix.size(0)
    if num_bus < 2:
        raise ValueError("Diagonal-free statistics require at least two buses.")
    mask = ~torch.eye(num_bus, dtype=torch.bool, device=matrix.device)
    return matrix.masked_select(mask).view(num_bus, num_bus - 1)


def _five_statistics(values, std_correction=0):
    """Compute ``[min, max, std, median, mean]`` across each row."""

    correction = int(std_correction)
    if correction < 0 or correction >= values.size(1):
        raise ValueError(
            "std_correction must be non-negative and smaller than the number "
            f"of summarized values ({values.size(1)}), got {correction}."
        )

    return torch.stack(
        (
            values.min(dim=1).values,
            values.max(dim=1).values,
            values.std(dim=1, correction=correction),
            torch.quantile(values, 0.5, dim=1),
            values.mean(dim=1),
        ),
        dim=1,
    )


def compute_effective_resistance_pe(
    data,
    std_correction=0,
    compute_device="auto",
    resistance=None,
):
    """Compute diagonal-free DC resistance statistics for every bus."""

    num_bus = int(data["bus"].x.size(0))
    if num_bus == 0:
        return {"effective_resistance_pe": torch.empty((0, 5), dtype=torch.float32)}
    if resistance is None:
        resistance = compute_effective_resistance_matrix(data, compute_device)
    stats = _five_statistics(
        _offdiagonal_rows(resistance), std_correction=std_correction
    )
    return {"effective_resistance_pe": stats.float().cpu()}


def compute_effective_resistance_rpe(
    data,
    compute_device="auto",
    resistance=None,
):
    """Return one raw effective-resistance feature for every bus pair."""

    if resistance is None:
        resistance = compute_effective_resistance_matrix(data, compute_device)
    return {"pairwise_rpe": resistance.float().unsqueeze(-1).cpu()}


def build_ybus(data, dtype=torch.complex128, device=None):
    """Build the raw complex bus-admittance matrix for one OPF sample."""

    bus = data["bus"]
    num_bus = int(bus.x.size(0))
    device = bus.x.device if device is None else torch.device(device)
    ybus = torch.zeros((num_bus, num_bus), dtype=dtype, device=device)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    def _add_branch(edge_type, transformer=False):
        store = data[edge_type]
        if store.edge_index.numel() == 0:
            return
        edge_index = store.edge_index.to(device=device, dtype=torch.long)
        attr = store.edge_attr.to(device=device, dtype=real_dtype)
        src, dst = edge_index[0], edge_index[1]

        if transformer:
            r, x = attr[:, 2], attr[:, 3]
            tap = attr[:, 7]
            tap = torch.where(tap.abs() > 0.0, tap, torch.ones_like(tap))
            shift = attr[:, 8]
            shunt = torch.complex(attr[:, 9], attr[:, 10]).to(dtype)
            complex_tap = tap.to(dtype) * torch.exp(1j * shift.to(dtype))
            series = (1.0 / torch.complex(r, x)).to(dtype)
            y_ff = (series + shunt) / (complex_tap * complex_tap.conj())
            y_ft = -series / complex_tap.conj()
            y_tf = -series / complex_tap
            y_tt = series
        else:
            r, x = attr[:, 4], attr[:, 5]
            series = (1.0 / torch.complex(r, x)).to(dtype)
            y_ff = series + 1j * attr[:, 2].to(dtype)
            y_tt = series + 1j * attr[:, 3].to(dtype)
            y_ft = -series
            y_tf = -series

        ybus.index_put_((src, src), y_ff, accumulate=True)
        ybus.index_put_((src, dst), y_ft, accumulate=True)
        ybus.index_put_((dst, src), y_tf, accumulate=True)
        ybus.index_put_((dst, dst), y_tt, accumulate=True)

    _add_branch(("bus", "ac_line", "bus"), transformer=False)
    _add_branch(("bus", "transformer", "bus"), transformer=True)

    shunt_type = ("shunt", "shunt_link", "bus")
    if "shunt" in data.node_types and shunt_type in data.edge_types:
        shunt_x = data["shunt"].x.to(device=device, dtype=real_dtype)
        bus_index = data[shunt_type].edge_index[1].to(device=device, dtype=torch.long)
        shunt_y = torch.complex(shunt_x[:, 1], shunt_x[:, 0]).to(dtype)
        ybus.index_put_((bus_index, bus_index), shunt_y, accumulate=True)

    return ybus


def compute_effective_impedance_matrix(
    data,
    relative_tolerance=None,
    compute_device="auto",
):
    """Return the dense complex effective-impedance matrix.

    ``Zbus = pinv(Ybus)`` maps bus-current injections to bus voltages.  For a
    unit transfer from bus ``i`` to bus ``j``, the corresponding voltage
    difference is ``Zii + Zjj - Zij - Zji``.  A transpose, rather than a
    conjugate transpose, is required by that injection/withdrawal experiment.
    """

    num_bus = int(data["bus"].x.size(0))
    if num_bus == 0:
        return torch.empty((0, 0), dtype=torch.complex128)
    device = _resolve_compute_device(compute_device)
    ybus = build_ybus(data, device=device)
    if relative_tolerance is None:
        zbus = torch.linalg.pinv(ybus)
    else:
        zbus = torch.linalg.pinv(ybus, rtol=float(relative_tolerance))
    diagonal = torch.diagonal(zbus)
    impedance = (
        diagonal[:, None]
        + diagonal[None, :]
        - zbus
        - zbus.transpose(0, 1)
    )
    impedance.fill_diagonal_(0.0)
    return impedance


def compute_effective_impedance_pe(
    data,
    std_correction=0,
    relative_tolerance=None,
    compute_device="auto",
    impedance=None,
):
    """Compute ten diagonal-free AC statistics: five each for real/imag."""

    num_bus = int(data["bus"].x.size(0))
    if num_bus == 0:
        return {"effective_impedance_pe": torch.empty((0, 10), dtype=torch.float32)}
    if impedance is None:
        impedance = compute_effective_impedance_matrix(
            data,
            relative_tolerance=relative_tolerance,
            compute_device=compute_device,
        )
    offdiagonal = _offdiagonal_rows(impedance)
    real_stats = _five_statistics(
        offdiagonal.real, std_correction=std_correction
    )
    imag_stats = _five_statistics(
        offdiagonal.imag, std_correction=std_correction
    )
    return {
        "effective_impedance_pe": torch.cat(
            (real_stats, imag_stats), dim=1
        ).float().cpu()
    }


def compute_effective_impedance_rpe(
    data,
    relative_tolerance=None,
    compute_device="auto",
    impedance=None,
):
    """Return raw real/imaginary effective impedance for every bus pair."""

    if impedance is None:
        impedance = compute_effective_impedance_matrix(
            data,
            relative_tolerance=relative_tolerance,
            compute_device=compute_device,
        )
    pairwise = torch.stack((impedance.real, impedance.imag), dim=-1)
    return {"pairwise_rpe": pairwise.float().cpu()}


def compute_ybus_svd_rpe(
    data,
    k,
    relative_tolerance=None,
    compute_device="auto",
):
    """Compute the compact factors needed to form dense bus-pair SVD-RPE.

    The ``k`` smallest nonzero singular components are selected.  Missing
    components are zero padded, and singular values are normalized by the
    largest singular value of the full Ybus spectrum.
    """

    k = int(k)
    if k <= 0:
        raise ValueError("Ybus SVD-RPE dimension must be positive.")

    device = _resolve_compute_device(compute_device)
    ybus = build_ybus(data, device=device)
    u, singular_values, vh = torch.linalg.svd(ybus, full_matrices=False)
    v = vh.conj().transpose(-2, -1)

    if singular_values.numel() == 0:
        selected = torch.empty(0, dtype=torch.long, device=ybus.device)
        scale = torch.tensor(1.0, dtype=singular_values.dtype, device=ybus.device)
    else:
        if relative_tolerance is None:
            eps = torch.finfo(singular_values.dtype).eps
            tolerance = max(ybus.shape) * singular_values.max() * eps
        else:
            tolerance = singular_values.max() * float(relative_tolerance)
        nonzero = torch.nonzero(singular_values > tolerance, as_tuple=False).view(-1)
        selected = nonzero[-k:]  # torch.linalg.svd returns descending values.
        selected = torch.flip(selected, dims=[0])  # smallest nonzero first.
        scale = singular_values.max().clamp_min(torch.finfo(singular_values.dtype).eps)

    num_bus = ybus.size(0)
    u_selected = torch.zeros((num_bus, k), dtype=ybus.dtype, device=ybus.device)
    v_selected = torch.zeros_like(u_selected)
    s_selected = torch.zeros((k,), dtype=singular_values.dtype, device=ybus.device)
    count = min(k, int(selected.numel()))
    if count:
        u_selected[:, :count] = u[:, selected[:count]]
        v_selected[:, :count] = v[:, selected[:count]]
        s_selected[:count] = singular_values[selected[:count]] / scale

    return {
        "svd_u_real": u_selected.real.float().cpu(),
        "svd_u_imag": u_selected.imag.float().cpu(),
        "svd_v_real": v_selected.real.float().cpu(),
        "svd_v_imag": v_selected.imag.float().cpu(),
        # Repeat per bus so the attention-bias path remains backward compatible.
        "svd_s": s_selected.float()
        .view(1, k)
        .expand(num_bus, k)
        .contiguous()
        .cpu(),
    }


_PAIRWISE_RPE_ATTRIBUTES = {
    "effective_resistance_rpe": (
        "effective_resistance_rpe",
        "effective_resistance_rpe_path",
    ),
    "effective_impedance_rpe": (
        "effective_impedance_rpe",
        "effective_impedance_rpe_path",
    ),
}


def _attach_artifact(data, source, artifact, artifact_path=None):
    store = data["bus"]
    if source == "laplacian":
        store.lap_eigvec = artifact["lap_eigvec"]
        # One row per graph avoids repeating eigenvalues for every bus.  PyG
        # batches this into [num_graphs, k], and the model expands by `batch`.
        store.lap_eigval = artifact["lap_eigval"]
    elif source == "effective_resistance":
        store.effective_resistance_pe = artifact["effective_resistance_pe"]
    elif source == "effective_resistance_qk":
        store.effective_resistance_qk = artifact["effective_resistance_qk"]
    elif source == "effective_impedance":
        store.effective_impedance_pe = artifact["effective_impedance_pe"]
    elif source in _PAIRWISE_RPE_ATTRIBUTES:
        tensor_attr, path_attr = _PAIRWISE_RPE_ATTRIBUTES[source]
        if artifact_path is None:
            setattr(store, tensor_attr, artifact["pairwise_rpe"])
        else:
            # Dense pair tensors are topology-level artifacts.  Serializing
            # only their path avoids duplicating an O(N^2) matrix in every
            # operating-point sample stored in HDF5.
            setattr(store, path_attr, os.path.abspath(artifact_path))
    elif source == "ybus_svd":
        for name, value in artifact.items():
            setattr(store, name, value)
    else:
        raise ValueError(f"Cannot attach unknown PE source '{source}'.")
    return data


def add_ybus_svd_rpe(
    data,
    k,
    relative_tolerance=None,
    compute_device="auto",
):
    """Attach Ybus SVD-RPE factors (legacy public entry point)."""

    artifact = compute_ybus_svd_rpe(
        data,
        k,
        relative_tolerance=relative_tolerance,
        compute_device=compute_device,
    )
    return _attach_artifact(data, "ybus_svd", artifact)


def _update_hash_with_tensor(digest, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())


def _source_fingerprint(data, source, source_config):
    digest = hashlib.sha256()
    digest.update(_CACHE_VERSION.encode("utf-8"))
    digest.update(source.encode("utf-8"))
    cache_config = copy.deepcopy(source_config)
    # Sign flipping is a model-time augmentation and does not change artifacts.
    cache_config.pop("random_sign_flip", None)
    # The learnable model coefficient does not change resistance coordinates.
    cache_config.pop("coefficient_init", None)
    cache_config.pop("placement", None)
    digest.update(json.dumps(cache_config, sort_keys=True).encode("utf-8"))
    digest.update(str(int(data["bus"].x.size(0))).encode("utf-8"))

    for edge_type, _ in _BUS_BRANCH_TYPES:
        store = _edge_store(data, edge_type)
        if store is None:
            digest.update(str(edge_type).encode("utf-8"))
            digest.update(b"missing")
            continue
        _update_hash_with_tensor(digest, store.edge_index)
        if source != "laplacian":
            _update_hash_with_tensor(digest, store.edge_attr)

    if source in {
        "ybus_svd",
        "effective_impedance",
        "effective_impedance_rpe",
        "effective_impedance_matrix",
    }:
        shunt_type = ("shunt", "shunt_link", "bus")
        if "shunt" in data.node_types and shunt_type in data.edge_types:
            _update_hash_with_tensor(digest, data["shunt"].x)
            _update_hash_with_tensor(digest, data[shunt_type].edge_index)
    return digest.hexdigest()


def _safe_case_name(case_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name or "graph"))


class OPFSpectralPEPreprocessor:
    """Compute, cache, and attach configured OPF positional encodings."""

    def __init__(self, architecture_config, cache_dir=None):
        self.config = resolve_opf_positional_encoding_config(architecture_config)
        self.cache_dir = cache_dir
        self._memory_cache = {}
        self._matrix_cache = {}
        self._case_keys = {}
        if cache_dir is not None and self.config["precompute"]:
            os.makedirs(cache_dir, exist_ok=True)

    @property
    def enabled(self):
        return bool(self.config["precompute"])

    def _settings(self, source):
        return self.config[source]

    def _shared_pairwise_matrix(self, data, kind, relative_tolerance=None):
        settings = {"relative_tolerance": relative_tolerance}
        fingerprint = _source_fingerprint(data, f"{kind}_matrix", settings)
        key = (kind, fingerprint)
        if key not in self._matrix_cache:
            if kind == "effective_resistance":
                matrix = compute_effective_resistance_matrix(
                    data, compute_device=self.config["compute_device"]
                )
            elif kind == "effective_impedance":
                matrix = compute_effective_impedance_matrix(
                    data,
                    relative_tolerance=relative_tolerance,
                    compute_device=self.config["compute_device"],
                )
            else:
                raise ValueError(f"Unknown pairwise matrix kind '{kind}'.")
            self._matrix_cache[key] = matrix
        return self._matrix_cache[key]

    def _compute(self, data, source):
        settings = self._settings(source)
        compute_device = self.config["compute_device"]
        if source == "laplacian":
            return compute_topological_laplacian_pe(
                data,
                int(settings["dim"]),
                relative_tolerance=settings.get("relative_tolerance"),
                compute_device=compute_device,
            )
        if source == "effective_resistance":
            resistance = self._shared_pairwise_matrix(
                data, "effective_resistance"
            )
            return compute_effective_resistance_pe(
                data,
                std_correction=int(settings.get("std_correction", 0)),
                compute_device=compute_device,
                resistance=resistance,
            )
        if source == "effective_resistance_qk":
            return compute_effective_resistance_qk(
                data,
                int(settings["dim"]),
                relative_tolerance=settings.get("relative_tolerance"),
                compute_device=compute_device,
            )
        if source == "effective_impedance":
            impedance = self._shared_pairwise_matrix(
                data,
                "effective_impedance",
                relative_tolerance=settings.get("relative_tolerance"),
            )
            return compute_effective_impedance_pe(
                data,
                std_correction=int(settings.get("std_correction", 0)),
                relative_tolerance=settings.get("relative_tolerance"),
                compute_device=compute_device,
                impedance=impedance,
            )
        if source == "effective_resistance_rpe":
            resistance = self._shared_pairwise_matrix(
                data, "effective_resistance"
            )
            return compute_effective_resistance_rpe(
                data,
                compute_device=compute_device,
                resistance=resistance,
            )
        if source == "effective_impedance_rpe":
            impedance = self._shared_pairwise_matrix(
                data,
                "effective_impedance",
                relative_tolerance=settings.get("relative_tolerance"),
            )
            return compute_effective_impedance_rpe(
                data,
                relative_tolerance=settings.get("relative_tolerance"),
                compute_device=compute_device,
                impedance=impedance,
            )
        if source == "ybus_svd":
            return compute_ybus_svd_rpe(
                data,
                int(settings["dim"]),
                relative_tolerance=settings.get("relative_tolerance"),
                compute_device=compute_device,
            )
        raise ValueError(f"Unknown PE source '{source}'.")

    @staticmethod
    def _load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_or_compute(self, data, source, case_name):
        case_key = (source, str(case_name))
        if self.config["cache_by_case"] and case_key in self._case_keys:
            fingerprint = self._case_keys[case_key]
        else:
            fingerprint = _source_fingerprint(data, source, self._settings(source))
            if self.config["cache_by_case"]:
                self._case_keys[case_key] = fingerprint

        memory_key = (source, fingerprint)
        if memory_key in self._memory_cache:
            artifact, path = self._memory_cache[memory_key]
            return artifact, path

        path = None
        if self.cache_dir is None:
            artifact = self._compute(data, source)
        else:
            filename = f"{_safe_case_name(case_name)}-{source}-{fingerprint[:20]}.pt"
            path = os.path.join(self.cache_dir, filename)
            lock_path = path + ".lock"
            with open(lock_path, "a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                if os.path.isfile(path):
                    artifact = self._load(path)
                else:
                    artifact = self._compute(data, source)
                    temporary = f"{path}.tmp-{os.getpid()}"
                    torch.save(artifact, temporary)
                    os.replace(temporary, path)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        self._memory_cache[memory_key] = (artifact, path)
        return artifact, path

    def __call__(self, data, case_name=None):
        for source in self.config["precompute"]:
            artifact, path = self._load_or_compute(data, source, case_name)
            _attach_artifact(data, source, artifact, artifact_path=path)
        return data
