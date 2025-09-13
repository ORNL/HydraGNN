###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
# Taken From:
# GitHub: https://github.com/ACEsuit/mace
# ArXiV: https://arxiv.org/pdf/2206.07697
# Date: August 27, 2024  |  12:37 (EST)
###########################################################################################

import collections
from typing import List, Union
import logging

import torch
from e3nn import o3

# Try to import OpenEquivariance for faster Clebsch-Gordon calculations
try:
    import openequivariance as oeq
    _OPENEQUIVARIANCE_AVAILABLE = True
    logging.debug("OpenEquivariance is available for accelerated Clebsch-Gordon tensor products")
except ImportError:
    _OPENEQUIVARIANCE_AVAILABLE = False
    logging.debug("OpenEquivariance not available, using e3nn for Clebsch-Gordon tensor products")

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim ** 0.5
                if normalization == "norm":
                    C *= ir_left.dim ** 0.5 * ir.dim ** 0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    """
    Compute U matrix for real Clebsch-Gordon coefficients.
    
    This function will use OpenEquivariance for acceleration when available,
    falling back to the original e3nn-based implementation otherwise.
    """
    # Try to use OpenEquivariance for faster computation
    if _OPENEQUIVARIANCE_AVAILABLE and correlation <= 4:
        try:
            return _U_matrix_real_openequivariance(
                irreps_in, irreps_out, correlation, normalization, filter_ir_mid, dtype
            )
        except Exception as e:
            logging.debug(f"OpenEquivariance U_matrix computation failed, falling back to e3nn: {e}")
    
    # Fallback to original e3nn implementation
    return _U_matrix_real_e3nn(
        irreps_in, irreps_out, correlation, normalization, filter_ir_mid, dtype
    )


def _U_matrix_real_openequivariance(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps], 
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    """
    OpenEquivariance-accelerated version of U_matrix_real.
    
    This leverages OpenEquivariance's optimized Clebsch-Gordon tensor products
    for better performance.
    """
    irreps_out = o3.Irreps(irreps_out)
    irreps_in = o3.Irreps(irreps_in)
    
    # For higher-order correlations, we can use OpenEquivariance's tensor products
    # to compute the symmetric contractions more efficiently
    if correlation == 2:
        # Direct tensor product for correlation 2
        instructions = []
        i_out = 0
        for mul_out, ir_out in irreps_out:
            for i1, (mul1, ir1) in enumerate(irreps_in):
                for i2, (mul2, ir2) in enumerate(irreps_in):
                    if ir_out in ir1 * ir2:
                        instructions.append((i1, i2, i_out, "uvu", True))
            i_out += 1
        
        # Create OpenEquivariance tensor product
        tp_problem = oeq.TPProblem(
            irreps_in, irreps_in, irreps_out, instructions,
            shared_weights=False, internal_weights=False
        )
        tp = oeq.TensorProduct(tp_problem, torch_op=True)
        
        # Generate basis matrices efficiently
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 1
        x1 = torch.eye(irreps_in.dim, dtype=dtype, device=device).unsqueeze(0)
        x2 = torch.eye(irreps_in.dim, dtype=dtype, device=device).unsqueeze(0)
        
        if tp.weight_numel > 0:
            weight = torch.ones(batch_size, tp.weight_numel, dtype=dtype, device=device)
            result = tp(x1, x2, weight)
        else:
            result = tp(x1, x2)
            
        return result.squeeze(0)
    
    # For other cases, fall back to original implementation
    return _U_matrix_real_e3nn(irreps_in, irreps_out, correlation, normalization, filter_ir_mid, dtype)


def _U_matrix_real_e3nn(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component", 
    filter_ir_mid=None,
    dtype=None,
):
    """
    Original e3nn-based implementation of U_matrix_real.
    """
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out
