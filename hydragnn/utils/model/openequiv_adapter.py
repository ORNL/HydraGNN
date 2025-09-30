"""
Adapter for converting an e3nn TensorProduct specification (irreps + instructions)
into an OpenEquivariance TPProblem and TensorProduct instance.

Used only when the OpenEquivariance backend is selected and available. If the
library is not installed, helper functions gracefully return (None, None).
"""
from __future__ import annotations
from typing import List, Sequence, Tuple, Any

try:
    import openequivariance as oeq  # type: ignore

    _OEQ_AVAILABLE = True
except Exception:  # pragma: no cover - guarded import
    _OEQ_AVAILABLE = False

from e3nn import o3

InstructionLikeWithScale = Tuple[int, int, int, str, bool, float]


def _normalize_instructions(
    instructions: Sequence[Tuple[Any, ...]]
) -> List[InstructionLikeWithScale]:
    """Normalize instruction tuples.

    e3nn style: (i_in1, i_in2, i_out, mode, trainable)
    OEQ superset: (i_in1, i_in2, i_out, mode, trainable, weight_scale)

    Adds weight_scale = 1.0 if missing.
    """
    norm: List[InstructionLikeWithScale] = []
    for inst in instructions:
        if len(inst) == 5:
            i1, i2, iout, mode, train = inst
            norm.append((int(i1), int(i2), int(iout), str(mode), bool(train), 1.0))
        elif len(inst) == 6:
            i1, i2, iout, mode, train, scale = inst
            norm.append(
                (int(i1), int(i2), int(iout), str(mode), bool(train), float(scale))
            )
        else:
            raise ValueError(
                f"Instruction {inst} has unsupported length {len(inst)}. Expected 5 or 6 entries."
            )
    return norm


def build_tp_problem_from_e3nn(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: Sequence[Tuple[Any, ...]],
    shared_weights: bool,
    internal_weights: bool,
):
    """Construct an OpenEquivariance TPProblem from e3nn tensor product inputs.
    Returns (problem, weight_numel) or (None, None) if OEQ not available.
    """
    if not _OEQ_AVAILABLE:
        return None, None

    if internal_weights:
        raise ValueError(
            "OpenEquivariance requires internal_weights=False (external weight tensor)."
        )

    norm_instr = _normalize_instructions(instructions)
    problem = oeq.TPProblem(
        irreps_in1,
        irreps_in2,
        irreps_out,
        norm_instr,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    return problem, problem.weight_numel


def build_oeq_tensor_product_module(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: Sequence[Tuple[Any, ...]],
    shared_weights: bool,
    internal_weights: bool,
    torch_op: bool = True,
    use_opaque: bool = False,
):
    """Return (OpenEquivariance TensorProduct module, weight_numel) or (None, None)."""
    if not _OEQ_AVAILABLE:
        return None, None
    problem, _ = build_tp_problem_from_e3nn(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights,
        internal_weights,
    )
    if problem is None:
        return None, None
    tp = oeq.TensorProduct(problem, torch_op=torch_op, use_opaque=use_opaque)
    return tp, problem.weight_numel
