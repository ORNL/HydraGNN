import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
import hydragnn.utils.profiling_and_tracing.tracer as tr
import os
from contextlib import contextmanager

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.optim import Optimizer


def average_gradients(model, group):
    """Averages gradients across all processes using all_reduce."""
    group_size = dist.get_world_size(group=group)

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=group, op=dist.ReduceOp.SUM)
            param.grad /= group_size  # Normalize by the number of processes


class EncoderModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.device = base_model.device
        self.is_mace = hasattr(base_model, "multihead_decoders")
        self._embedding = base_model._embedding  # Use existing embedding function
        self.graph_convs = base_model.graph_convs
        self.feature_layers = base_model.feature_layers
        self.activation_function = base_model.activation_function
        self.conv_checkpointing = base_model.conv_checkpointing
        self._apply_graph_conditioning = getattr(
            base_model, "_apply_graph_conditioning"
        )

    def forward(self, data):
        if self.is_mace:
            inv_node_feat, equiv_node_feat, conv_args = self._embedding(data)

            batch_for_cond = (
                data.batch
                if hasattr(data, "batch") and data.batch is not None
                else None
            )
            inv_node_feat = self._apply_graph_conditioning(
                inv_node_feat, batch_for_cond, data
            )

            node_features_per_layer = []
            for conv in self.graph_convs:
                if not self.conv_checkpointing:
                    inv_node_feat, equiv_node_feat = conv(
                        inv_node_feat=inv_node_feat,
                        equiv_node_feat=equiv_node_feat,
                        **conv_args,
                    )
                else:
                    inv_node_feat, equiv_node_feat = checkpoint(
                        conv,
                        use_reentrant=False,
                        inv_node_feat=inv_node_feat,
                        equiv_node_feat=equiv_node_feat,
                        **conv_args,
                    )

                inv_node_feat = self._apply_graph_conditioning(
                    inv_node_feat, batch_for_cond, data
                )
                node_features_per_layer.append(
                    torch.cat([inv_node_feat, equiv_node_feat], dim=1)
                )

            return {
                "node_attributes": data.node_attributes,
                "node_features_per_layer": node_features_per_layer,
            }

        ### encoder part ####
        inv_node_feat, equiv_node_feat, conv_args = self._embedding(data)

        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                inv_node_feat, equiv_node_feat = conv(
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
            else:
                inv_node_feat, equiv_node_feat = checkpoint(
                    conv,
                    use_reentrant=False,
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
            inv_node_feat = self.activation_function(feat_layer(inv_node_feat))

        return inv_node_feat, equiv_node_feat, conv_args


class DecoderModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.device = base_model.device
        self.is_mace = hasattr(base_model, "multihead_decoders")

        if self.is_mace:
            self.multihead_decoders = base_model.multihead_decoders
        else:
            self.graph_shared = base_model.graph_shared
            self.heads_NN = base_model.heads_NN
            self.head_dims = base_model.head_dims
            self.head_type = base_model.head_type
            self.config_heads = base_model.config_heads
            self.var_output = base_model.var_output
            self.activation_function = base_model.activation_function
            self.num_branches = base_model.num_branches
            self.graph_pool_fn = base_model.graph_pool_fn
            self.graph_pool_reduction = base_model.graph_pool_reduction
            self.graph_pooling = base_model.graph_pooling

    def _pool_graph_features(self, x_tensor, batch_tensor):
        if batch_tensor is None:
            if self.graph_pool_reduction == "mean":
                return x_tensor.mean(dim=0, keepdim=True)
            if self.graph_pool_reduction == "max":
                return x_tensor.max(dim=0, keepdim=True).values
            return x_tensor.sum(dim=0, keepdim=True)
        return self.graph_pool_fn(x_tensor, batch_tensor.to(x_tensor.device))

    def forward(self, data, encoded_feats):
        if self.is_mace:
            outputs = self.multihead_decoders[0](data, encoded_feats["node_attributes"])
            for readout, node_features in zip(
                self.multihead_decoders[1:], encoded_feats["node_features_per_layer"]
            ):
                output = readout(data, node_features)
                for idx, prediction in enumerate(output):
                    outputs[idx] = outputs[idx] + prediction
            return outputs

        ## Take encoded features as input
        inv_node_feat, equiv_node_feat, conv_args = encoded_feats
        x = inv_node_feat

        #### multi-head decoder part####
        # shared dense layers for graph level output
        if data.batch is None:
            x_graph = self._pool_graph_features(x, None)
            data.batch = data.x * 0
        else:
            x_graph = self._pool_graph_features(x, data.batch)

        outputs = []
        outputs_var = []

        datasetIDs = data.dataset_name.unique()
        unique, node_counts = torch.unique_consecutive(data.batch, return_counts=True)
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                head = torch.zeros(
                    (len(data.dataset_name), head_dim), device=x.device, dtype=x.dtype
                )
                headvar = torch.zeros(
                    (len(data.dataset_name), head_dim * self.var_output),
                    device=x.device,
                    dtype=x.dtype,
                )
                if self.num_branches == 1:
                    x_graph_head = self.graph_shared["branch-0"](x_graph)
                    output_head = headloc["branch-0"](x_graph_head)
                    head = output_head[:, :head_dim]
                    headvar = (output_head[:, head_dim:] ** 2).to(dtype=x.dtype)
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask = mask[:, 0]
                        branchtype = f"branch-{ID.item()}"
                        # print("Pei debugging:", branchtype, data.dataset_name, mask, data.dataset_name[mask])
                        x_graph_head = self.graph_shared[branchtype](x_graph[mask, :])
                        output_head = headloc[branchtype](x_graph_head)
                        head[mask] = output_head[:, :head_dim]
                        headvar[mask] = (output_head[:, head_dim:] ** 2).to(
                            dtype=x.dtype
                        )
                outputs.append(head)
                outputs_var.append(headvar)
            else:
                # assuming all node types are the same
                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                head = torch.zeros(
                    (x.shape[0], head_dim), device=x.device, dtype=x.dtype
                )
                headvar = torch.zeros(
                    (x.shape[0], head_dim * self.var_output),
                    device=x.device,
                    dtype=x.dtype,
                )
                if self.num_branches == 1:
                    branchtype = "branch-0"
                    if node_NN_type == "conv":
                        inv_node_feat = x
                        equiv_node_feat_ = equiv_node_feat
                        for conv, batch_norm in zip(
                            headloc[branchtype][0::2], headloc[branchtype][1::2]
                        ):
                            inv_node_feat, equiv_node_feat_ = conv(
                                inv_node_feat=inv_node_feat,
                                equiv_node_feat=equiv_node_feat_,
                                **conv_args,
                            )
                            inv_node_feat = batch_norm(inv_node_feat)
                            inv_node_feat = self.activation_function(inv_node_feat)
                        x_node = inv_node_feat
                    else:
                        x_node = headloc[branchtype](x=x, batch=data.batch)
                    head = x_node[:, :head_dim]
                    headvar = (x_node[:, head_dim:] ** 2).to(dtype=x.dtype)
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask_nodes = torch.repeat_interleave(mask, node_counts)
                        branchtype = f"branch-{ID.item()}"
                        # print("Pei debugging:", branchtype, data.dataset_name, mask, data.dataset_name[mask])
                        if node_NN_type == "conv":
                            inv_node_feat = x[mask_nodes, :]
                            equiv_node_feat_ = equiv_node_feat[mask_nodes, :]
                            for conv, batch_norm in zip(
                                headloc[branchtype][0::2], headloc[branchtype][1::2]
                            ):
                                inv_node_feat, equiv_node_feat_ = conv(
                                    inv_node_feat=inv_node_feat,
                                    equiv_node_feat=equiv_node_feat_,
                                    **conv_args,
                                )
                                inv_node_feat = batch_norm(inv_node_feat)
                                inv_node_feat = self.activation_function(inv_node_feat)
                            x_node = inv_node_feat
                        else:
                            x_node = headloc[branchtype](
                                x=x[mask_nodes, :], batch=data.batch[mask_nodes]
                            )
                        head[mask_nodes] = x_node[:, :head_dim]
                        headvar[mask_nodes] = (x_node[:, head_dim:] ** 2).to(
                            dtype=x.dtype
                        )
                outputs.append(head)
                outputs_var.append(headvar)
        if self.var_output:
            return outputs, outputs_var
        return outputs


class MultiTaskModelMP(nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        group_color: int,
        head_pg: dist.ProcessGroup,
    ):
        super().__init__()

        self.shared_pg = dist.group.WORLD
        self.head_pg = head_pg
        self.shared_pg_size = dist.get_world_size(group=self.shared_pg)
        self.shared_pg_rank = dist.get_rank(group=self.shared_pg)
        self.head_pg_size = dist.get_world_size(group=self.head_pg)
        self.head_pg_rank = dist.get_rank(group=self.head_pg)
        print(
            self.shared_pg_rank,
            "shared, head:",
            (self.shared_pg_size, self.shared_pg_rank),
            (self.head_pg_size, self.head_pg_rank),
            group_color,
        )

        # assert self.shared_pg_size % self.head_pg_size == 0
        # self.total_num_heads = self.shared_pg_size // self.head_pg_size
        self.branch_id = group_color
        print(self.shared_pg_rank, "branch_id:", self.branch_id)

        self.encoder = EncoderModel(base_model)
        self.decoder = DecoderModel(base_model)

        if hasattr(self.decoder, "multihead_decoders"):
            for decoder_block in self.decoder.multihead_decoders:
                if hasattr(decoder_block, "graph_shared"):
                    delete_list = []
                    for name in decoder_block.graph_shared.keys():
                        if name != f"branch-{self.branch_id}":
                            delete_list.append(name)
                    for key in delete_list:
                        del decoder_block.graph_shared[key]

                if hasattr(decoder_block, "heads_NN"):
                    for layer in decoder_block.heads_NN:
                        delete_list = []
                        for key in layer.keys():
                            if key != f"branch-{self.branch_id}":
                                delete_list.append(key)
                        for key in delete_list:
                            del layer[key]
        else:
            delete_list = list()
            for name, layer in self.decoder.graph_shared.named_children():
                if name != f"branch-{self.branch_id}":
                    delete_list.append(name)

            for k in delete_list:
                del self.decoder.graph_shared[k]

            for name, layer in self.decoder.heads_NN.named_children():
                delete_list = list()
                for k in layer.keys():
                    if k != f"branch-{self.branch_id}":
                        delete_list.append(k)
                for k in delete_list:
                    del layer[k]

        ## check if FSDP is to be used
        use_fsdp = bool(int(os.getenv("HYDRAGNN_USE_FSDP", "0")))
        fsdp_version = int(os.getenv("HYDRAGNN_FSDP_VERSION", "1"))
        if fsdp_version not in [1, 2]:
            raise ValueError(
                f"Unsupported HYDRAGNN_FSDP_VERSION={fsdp_version}. Supported values are 1 or 2."
            )
        ## List of ShardingStrategy: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD, HYBRID_SHARD_ZERO2
        fsdp_strategy = os.getenv("HYDRAGNN_FSDP_STRATEGY", "FULL_SHARD")
        sharding_strategy = eval(f"ShardingStrategy.{fsdp_strategy}")
        print(
            "MultiTaskModelMP FSDP:",
            use_fsdp,
            "Version:",
            fsdp_version,
            "Sharding:",
            sharding_strategy,
        )

        if use_fsdp:
            if fsdp_version != 1:
                raise NotImplementedError(
                    "MultiTaskModelMP currently supports only HYDRAGNN_FSDP_VERSION=1. "
                    "FSDP2/composable wrapping does not yet support this separate-process-group "
                    "encoder/decoder setup in HydraGNN."
                )
            self.encoder = FSDP(
                self.encoder,
                process_group=self.shared_pg,
                sharding_strategy=sharding_strategy,
            )
            self.decoder = FSDP(
                self.decoder,
                process_group=self.head_pg,
                sharding_strategy=sharding_strategy,
            )
        else:
            self.encoder = DDP(self.encoder, process_group=self.shared_pg)
            self.decoder = DDP(self.decoder, process_group=self.head_pg)
        self.module = base_model

    def forward(self, data):
        tr.start("enc_forward")
        encoded_feats = self.encoder(data)  # First call (encoder)
        tr.stop("enc_forward")
        tr.start(f"branch{self.branch_id}_forward")
        out = self.decoder(data, encoded_feats)  # Second call (decoder)
        tr.stop(f"branch{self.branch_id}_forward")
        return out

    def parameters(self):
        for x in self.encoder.parameters():
            yield x
        for x in self.decoder.parameters():
            yield x

    def named_parameters(self):
        for name, param in self.encoder.named_parameters():
            yield name, param
        for name, param in self.decoder.named_parameters():
            yield name, param

    def state_dict(self):
        return OrderedDict(
            list(self.encoder.state_dict().items())
            + list(self.decoder.state_dict().items())
        )

    def load_state_dict(self, state_dict):
        enc_state_dict = OrderedDict()
        dec_state_dict = OrderedDict()
        enc_keys = self.encoder.state_dict().keys()
        dec_keys = self.decoder.state_dict().keys()
        for k in state_dict.keys():
            if k in enc_keys:
                enc_state_dict[k] = state_dict[k]
            elif k in dec_keys:
                dec_state_dict[k] = state_dict[k]
            else:
                print("Warning: key not found in either encoder or decoder:", k)
        self.encoder.load_state_dict(enc_state_dict)
        self.decoder.load_state_dict(dec_state_dict)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def gradient_all_reduce(self):
        average_gradients(self.encoder, self.shared_pg)
        average_gradients(self.decoder, self.head_pg)

    @contextmanager
    def no_sync(self):
        old_encoder_require_backward_grad_sync = self.encoder.require_backward_grad_sync
        old_decoder_require_backward_grad_sync = self.decoder.require_backward_grad_sync
        self.encoder.require_backward_grad_sync = False
        self.decoder.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.encoder.require_backward_grad_sync = (
                old_encoder_require_backward_grad_sync
            )
            self.decoder.require_backward_grad_sync = (
                old_decoder_require_backward_grad_sync
            )


class DualOptimizer(Optimizer):
    """
    A wrapper optimizer that combines two optimizers.
    """

    def __init__(self, optimizer1: Optimizer, optimizer2: Optimizer):
        # Optimizer base class requires param_groups,
        # but we delegate everything to the wrapped optimizers.
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2

        # Fake param_groups to satisfy base Optimizer API
        param_groups = optimizer1.param_groups + optimizer2.param_groups
        defaults = {}
        super().__init__(param_groups, defaults)

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer1.zero_grad(set_to_none=set_to_none)
        self.optimizer2.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.optimizer1.step()
        self.optimizer2.step()
        return loss

    def state_dict(self):
        return {
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer1.load_state_dict(state_dict["optimizer1"])
        self.optimizer2.load_state_dict(state_dict["optimizer2"])
