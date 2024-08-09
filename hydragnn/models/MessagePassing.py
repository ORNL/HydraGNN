import os.path as osp
import warnings
from abc import abstractmethod
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from torch_geometric import EdgeIndex, is_compiling
from torch_geometric.index import ptr2index
from torch_geometric.inspector import Inspector, Signature
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.template import module_from_template
from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
)

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}
HookDict = OrderedDict[int, Callable]


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers.

    Message passing layers follow the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\bigoplus` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"add"`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
        flow (str, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            This method can accelerate GNN execution on CPU-based platforms
            (*e.g.*, 2-3x speedup on the
            :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
            models such as :class:`~torch_geometric.nn.models.GCN`,
            :class:`~torch_geometric.nn.models.GraphSAGE`,
            :class:`~torch_geometric.nn.models.GIN`, etc.
            However, this method is not applicable to all GNN operators
            available, in particular for operators in which message computation
            can not easily be decomposed, *e.g.* in attention-based GNNs.
            The selection of the optimal value of :obj:`decomposed_layers`
            depends both on the specific graph dataset and available hardware
            resources.
            A value of :obj:`2` is suitable in most cases.
            Although the peak memory usage is directly associated with the
            granularity of feature decomposition, the same is not necessarily
            true for execution speedups. (default: :obj:`1`)
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    # Supports `message_and_aggregate` via `EdgeIndex`.
    # TODO Remove once migration is finished.
    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = False

    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'sum',
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
    ) -> None:
        super().__init__()

        if flow not in ['source_to_target', 'target_to_source']:
            raise ValueError(f"Expected 'flow' to be either 'source_to_target'"
                             f" or 'target_to_source' (got '{flow}')")

        # Cast `aggr` into a string representation for backward compatibility:
        self.aggr: Optional[Union[str, List[str]]]
        if aggr is None:
            self.aggr = None
        elif isinstance(aggr, (str, Aggregation)):
            self.aggr = str(aggr)
        elif isinstance(aggr, (tuple, list)):
            self.aggr = [str(x) for x in aggr]

        self.aggr_module = aggr_resolver(aggr, **(aggr_kwargs or {}))
        self.flow = flow
        self.node_dim = node_dim

        # Collect attribute names requested in message passing hooks:
        self.inspector = Inspector(self.__class__)
        self.inspector.inspect_signature(self.message)
        self.inspector.inspect_signature(self.aggregate, exclude=[0, 'aggr'])
        self.inspector.inspect_signature(self.message_and_aggregate, [0])
        self.inspector.inspect_signature(self.update, exclude=[0])
        self.inspector.inspect_signature(self.edge_update)

        self._user_args: List[str] = self.inspector.get_flat_param_names(
            ['message', 'aggregate', 'update'], exclude=self.special_args)
        self._fused_user_args: List[str] = self.inspector.get_flat_param_names(
            ['message_and_aggregate', 'update'], exclude=self.special_args)
        self._edge_user_args: List[str] = self.inspector.get_param_names(
            'edge_update', exclude=self.special_args)

        # Support for "fused" message passing:
        self.fuse = self.inspector.implements('message_and_aggregate')
        if self.aggr is not None:
            self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS

        # Hooks:
        self._propagate_forward_pre_hooks: HookDict = OrderedDict()
        self._propagate_forward_hooks: HookDict = OrderedDict()
        self._message_forward_pre_hooks: HookDict = OrderedDict()
        self._message_forward_hooks: HookDict = OrderedDict()
        self._aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._aggregate_forward_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_hooks: HookDict = OrderedDict()
        self._edge_update_forward_pre_hooks: HookDict = OrderedDict()
        self._edge_update_forward_hooks: HookDict = OrderedDict()

        # Set jittable `propagate` and `edge_updater` function templates:
        self._set_jittable_templates()

        # Explainability:
        self._explain: Optional[bool] = None
        self._edge_mask: Optional[Tensor] = None
        self._loop_mask: Optional[Tensor] = None
        self._apply_sigmoid: bool = True

        # Inference Decomposition:
        self._decomposed_layers = 1
        self.decomposed_layers = decomposed_layers

def propagate(
    self,
    edge_index: Adj,
    size: Size = None,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    decomposed_layers = 1 if self.explain else self.decomposed_layers

    for hook in self._propagate_forward_pre_hooks.values():
        res = hook(self, (edge_index, size, kwargs))
        if res is not None:
            edge_index, size, kwargs = res

    mutable_size = self._check_input(edge_index, size)

    if fuse:
        coll_dict = self._collect(self._fused_user_args, edge_index, mutable_size, kwargs)
        msg_aggr_kwargs = self.inspector.collect_param_data('message_and_aggregate', coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        out = self.update(out, **update_kwargs)
    else:
        if decomposed_layers > 1:
            user_args = self._user_args
            decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
            decomp_kwargs = {a: kwargs[a].chunk(decomposed_layers, -1) for a in decomp_args}
            decomp_out = []

        for i in range(decomposed_layers):
            if decomposed_layers > 1:
                for arg in decomp_args:
                    kwargs[arg] = decomp_kwargs[arg][i]

            coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

            # Call message function
            msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
            for hook in self._message_forward_pre_hooks.values():
                res = hook(self, (msg_kwargs, ))
                if res is not None:
                    msg_kwargs = res[0] if isinstance(res, tuple) else res

            # Change: Receive two outputs from message function
            message_scalar, message_vector = self.message(**msg_kwargs)

            for hook in self._message_forward_hooks.values():
                # Change: Pass two outputs to hooks
                res = hook(self, (msg_kwargs, ), (message_scalar, message_vector))
                if res is not None:
                    message_scalar, message_vector = res

            if self.explain:
                explain_msg_kwargs = self.inspector.collect_param_data('explain_message', coll_dict)
                message_scalar = self.explain_message(message_scalar, **explain_msg_kwargs)
                message_vector = self.explain_message(message_vector, **explain_msg_kwargs)

            # Aggregate function
            aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
            for hook in self._aggregate_forward_pre_hooks.values():
                res = hook(self, (aggr_kwargs, ))
                if res is not None:
                    aggr_kwargs = res[0] if isinstance(res, tuple) else res

            # Change: Perform aggregation separately for each output
            aggregated_scalar = self.aggr_module_scalar(message_scalar, **aggr_kwargs)
            aggregated_vector = self.aggr_module_vector(message_vector, **aggr_kwargs)

            for hook in self._aggregate_forward_hooks.values():
                # Change: Pass aggregated outputs to hooks
                res = hook(self, (aggr_kwargs, ), (aggregated_scalar, aggregated_vector))
                if res is not None:
                    aggregated_scalar, aggregated_vector = res

            # Update function
            update_kwargs = self.inspector.collect_param_data('update', coll_dict)
            # Change: Update with both aggregated outputs
            out = self.update((aggregated_scalar, aggregated_vector), **update_kwargs)

            if decomposed_layers > 1:
                decomp_out.append(out)

        if decomposed_layers > 1:
            out = torch.cat(decomp_out, dim=-1)

    for hook in self._propagate_forward_hooks.values():
        res = hook(self, (edge_index, mutable_size, kwargs), out)
        if res is not None:
            out = res

    return out

def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)
