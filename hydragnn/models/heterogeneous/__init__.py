import torch
import torch_geometric.typing as _pyg_typing


def _disable_cpu_only_segment_matmul():
    # A CPU-only pyg-lib build exposes ``segment_matmul`` without a GPU kernel,
    # so PyG's HeteroLinear (HGT/HEAT) would dispatch device tensors to it.
    if not (_pyg_typing.WITH_SEGMM and torch.cuda.is_available()):
        return
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "pyg::segment_matmul", "CUDA"
    ):
        _pyg_typing.WITH_SEGMM = False
        _pyg_typing.WITH_GMM = False


_disable_cpu_only_segment_matmul()

from .hetero_base import HeteroBase
from .hetero_gin_stack import HeteroGINStack
from .hetero_sage_stack import HeteroSAGEStack
from .hetero_gat_stack import HeteroGATStack
from .hetero_pna_stack import HeteroPNAStack
from .hetero_hgt_stack import HeteroHGTStack
from .hetero_heat_stack import HeteroHEATStack
from .hetero_rgat_stack import HeteroRGATStack
