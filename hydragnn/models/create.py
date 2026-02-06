##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import torch
from torch_geometric.data import Data
from typing import List, Union

import torch_scatter

from hydragnn.models.Base import Base
from hydragnn.models.GINStack import GINStack
from hydragnn.models.PNAStack import PNAStack
from hydragnn.models.PNAPlusStack import PNAPlusStack
from hydragnn.models.GATStack import GATStack
from hydragnn.models.MFCStack import MFCStack
from hydragnn.models.CGCNNStack import CGCNNStack
from hydragnn.models.SAGEStack import SAGEStack
from hydragnn.models.SCFStack import SCFStack
from hydragnn.models.DIMEStack import DIMEStack
from hydragnn.models.EGCLStack import EGCLStack
from hydragnn.models.PNAEqStack import PNAEqStack
from hydragnn.models.PAINNStack import PAINNStack
from hydragnn.models.MACEStack import MACEStack
from hydragnn.models.heterogeneous import (
    HeteroGINStack,
    HeteroSAGEStack,
    HeteroGATStack,
    HeteroPNAStack,
)

# InteratomicPotential functionality is now implemented via wrapper composition

from hydragnn.utils.distributed import get_device
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.train.train_validate_test import resolve_precision


def create_model_config(
    config: dict,
    verbosity: int = 0,
    use_gpu: bool = True,
    metadata=None,
):
    model = create_model(
        mpnn_type=config["Architecture"]["mpnn_type"],
        input_dim=config["Architecture"]["input_dim"],
        hidden_dim=config["Architecture"]["hidden_dim"],
        output_dim=config["Architecture"]["output_dim"],
        pe_dim=config["Architecture"]["pe_dim"],
        global_attn_engine=config["Architecture"]["global_attn_engine"],
        global_attn_type=config["Architecture"]["global_attn_type"],
        global_attn_heads=config["Architecture"]["global_attn_heads"],
        output_type=config["Architecture"]["output_type"],
        output_heads=config["Architecture"]["output_heads"],
        activation_function=config["Architecture"]["activation_function"],
        loss_function_type=config["Training"]["loss_function_type"],
        task_weights=config["Architecture"]["task_weights"],
        num_conv_layers=config["Architecture"]["num_conv_layers"],
        freeze_conv=config["Architecture"]["freeze_conv_layers"],
        initial_bias=config["Architecture"]["initial_bias"],
        num_nodes=config["Architecture"]["num_nodes"],
        max_neighbours=config["Architecture"]["max_neighbours"],
        edge_dim=config["Architecture"]["edge_dim"],
        pna_deg=config["Architecture"]["pna_deg"],
        num_before_skip=config["Architecture"]["num_before_skip"],
        num_after_skip=config["Architecture"]["num_after_skip"],
        num_radial=config["Architecture"]["num_radial"],
        radial_type=config["Architecture"]["radial_type"],
        distance_transform=config["Architecture"]["distance_transform"],
        basis_emb_size=config["Architecture"]["basis_emb_size"],
        int_emb_size=config["Architecture"]["int_emb_size"],
        out_emb_size=config["Architecture"]["out_emb_size"],
        envelope_exponent=config["Architecture"]["envelope_exponent"],
        num_spherical=config["Architecture"]["num_spherical"],
        num_gaussians=config["Architecture"]["num_gaussians"],
        num_filters=config["Architecture"]["num_filters"],
        radius=config["Architecture"]["radius"],
        equivariance=config["Architecture"]["equivariance"],
        correlation=config["Architecture"]["correlation"],
        max_ell=config["Architecture"]["max_ell"],
        node_max_ell=config["Architecture"]["node_max_ell"],
        avg_num_neighbors=config["Architecture"]["avg_num_neighbors"],
        conv_checkpointing=config["Training"]["conv_checkpointing"],
        enable_interatomic_potential=config["Architecture"].get(
            "enable_interatomic_potential", False
        ),
        energy_weight=config["Architecture"].get("energy_weight", 0.0),
        energy_peratom_weight=config["Architecture"].get("energy_peratom_weight", 0.0),
        force_weight=config["Architecture"].get("force_weight", 0.0),
        use_graph_attr_conditioning=config["Architecture"].get(
            "use_graph_attr_conditioning", False
        ),
        graph_attr_conditioning_mode=config["Architecture"].get(
            "graph_attr_conditioning_mode", "concat_node"
        ),
        graph_pooling=config["Architecture"].get("graph_pooling", "mean"),
        hetero_pooling_mode=config["Architecture"].get("hetero_pooling_mode", "sum"),
        node_target_type=config["Architecture"].get("node_target_type", None),
        share_relation_weights=config["Architecture"].get(
            "share_relation_weights", False
        ),
        metadata=metadata,
        verbosity=verbosity,
        use_gpu=use_gpu,
    )

    ## Set precision: bf16, fp32, fp64
    training_cfg = config["Training"]
    precision, param_dtype, _ = resolve_precision(training_cfg.get("precision", "fp32"))
    torch.set_default_dtype(param_dtype)

    return model.to(dtype=param_dtype)


# FIXME: interface does not include ilossweights_hyperp, ilossweights_nll, dropout
def create_model(
    mpnn_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: list,
    pe_dim: int,
    global_attn_engine: str,
    global_attn_type: str,
    global_attn_heads: int,
    output_type: list,
    output_heads: dict,
    activation_function: str,
    loss_function_type: str,
    task_weights: list,
    num_conv_layers: int,
    freeze_conv: bool = False,
    initial_bias: float = None,
    num_nodes: int = None,
    max_neighbours: int = None,
    edge_dim: int = None,
    pna_deg: torch.tensor = None,
    num_before_skip: int = None,
    num_after_skip: int = None,
    num_radial: int = None,
    radial_type: str = None,
    distance_transform: str = None,
    basis_emb_size: int = None,
    int_emb_size: int = None,
    out_emb_size: int = None,
    envelope_exponent: int = None,
    num_spherical: int = None,
    num_gaussians: int = None,
    num_filters: int = None,
    radius: float = None,
    equivariance: bool = False,
    correlation: Union[int, List[int]] = None,
    max_ell: int = None,
    node_max_ell: int = None,
    avg_num_neighbors: int = None,
    conv_checkpointing: bool = False,
    enable_interatomic_potential: bool = False,
    energy_weight: float = 0.0,
    energy_peratom_weight: float = 0.0,
    force_weight: float = 0.0,
    use_graph_attr_conditioning: bool = False,
    graph_attr_conditioning_mode: str = "fuse_pool",
    graph_pooling: str = "mean",
    hetero_pooling_mode: str = "sum",
    node_target_type: str = None,
    share_relation_weights: bool = False,
    metadata=None,
    verbosity: int = 0,
    use_gpu: bool = True,
):
    timer = Timer("create_model")
    timer.start()
    torch.manual_seed(0)

    device = get_device(use_gpu, verbosity_level=verbosity)

    # Note: model-specific inputs must come first.
    if mpnn_type == "GIN":
        model = GINStack(
            "inv_node_feat, equiv_node_feat, edge_index",
            "inv_node_feat, edge_index",
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "PNA":
        assert pna_deg is not None, "PNA requires degree input."
        model = PNAStack(
            "inv_node_feat, equiv_node_feat, edge_index",
            "inv_node_feat, edge_index",
            pna_deg,
            edge_dim,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "PNAPlus":
        assert pna_deg is not None, "PNAPlus requires degree input."
        assert (
            envelope_exponent is not None
        ), "PNAPlus requires envelope_exponent input."
        assert num_radial is not None, "PNAPlus requires num_radial input."
        assert radius is not None, "PNAPlus requires radius input."
        model = PNAPlusStack(
            "inv_node_feat, equiv_node_feat, edge_index, rbf",
            "inv_node_feat, edge_index, rbf",
            pna_deg,
            edge_dim,
            envelope_exponent,
            num_radial,
            radius,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "GAT":
        # FIXME: expose options to users
        heads = 6
        negative_slope = 0.05
        model = GATStack(
            "inv_node_feat, equiv_node_feat, edge_index",
            "inv_node_feat, edge_index",
            heads,
            negative_slope,
            edge_dim,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "MFC":
        assert max_neighbours is not None, "MFC requires max_neighbours input."
        model = MFCStack(
            "inv_node_feat, equiv_node_feat, edge_index",
            "inv_node_feat, edge_index",
            max_neighbours,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "CGCNN":
        model = CGCNNStack(
            "inv_node_feat, equiv_node_feat, edge_index",  # input_args
            "inv_node_feat, edge_index",  # conv_args
            edge_dim,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "SAGE":
        model = SAGEStack(
            "inv_node_feat, equiv_node_feat, edge_index",  # input_args
            "inv_node_feat, edge_index",  # conv_args
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "SchNet":
        assert num_gaussians is not None, "SchNet requires num_guassians input."
        assert num_filters is not None, "SchNet requires num_filters input."
        assert radius is not None, "SchNet requires radius input."
        model = SCFStack(
            "",  # Depends on SchNet usage of edge_features
            "inv_node_feat, equiv_node_feat, edge_index, edge_weight, edge_rbf",
            num_filters,
            edge_dim,
            num_gaussians,
            radius,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            max_neighbours=max_neighbours,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "DimeNet":
        assert basis_emb_size is not None, "DimeNet requires basis_emb_size input."
        assert (
            envelope_exponent is not None
        ), "DimeNet requires envelope_exponent input."
        assert int_emb_size is not None, "DimeNet requires int_emb_size input."
        assert out_emb_size is not None, "DimeNet requires out_emb_size input."
        assert num_after_skip is not None, "DimeNet requires num_after_skip input."
        assert num_before_skip is not None, "DimeNet requires num_before_skip input."
        assert num_radial is not None, "DimeNet requires num_radial input."
        assert num_spherical is not None, "DimeNet requires num_spherical input."
        assert radius is not None, "DimeNet requires radius input."
        model = DIMEStack(
            "inv_node_feat, equiv_node_feat, rbf, sbf, i, j, idx_kj, idx_ji",  # input_args
            "",  # conv_args
            basis_emb_size,
            envelope_exponent,
            int_emb_size,
            out_emb_size,
            num_after_skip,
            num_before_skip,
            num_radial,
            num_spherical,
            edge_dim,
            radius,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            max_neighbours=max_neighbours,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "EGNN":
        model = EGCLStack(
            "inv_node_feat, equiv_node_feat, edge_index, edge_attr, edge_shifts",  # input_args
            "",  # conv_args
            edge_dim,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            max_neighbours=max_neighbours,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "PAINN":
        model = PAINNStack(
            # edge_dim,   # To-do add edge_features
            "inv_node_feat, equiv_node_feat, edge_index, diff, dist",
            "inv_node_feat, equiv_node_feat, edge_index, diff, dist",
            edge_dim,
            num_radial,
            radius,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "PNAEq":
        assert pna_deg is not None, "PNAEq requires degree input."
        model = PNAEqStack(
            "inv_node_feat, equiv_node_feat, edge_index, edge_rbf, edge_vec",
            "inv_node_feat, equiv_node_feat, edge_index, edge_rbf, edge_vec",
            pna_deg,
            edge_dim,
            num_radial,
            radius,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "MACE":
        assert radius is not None, "MACE requires radius input."
        assert num_radial is not None, "MACE requires num_radial input."
        assert max_ell is not None, "MACE requires max_ell input."
        assert node_max_ell is not None, "MACE requires node_max_ell input."
        assert max_ell >= 1, "MACE requires max_ell >= 1."
        assert node_max_ell >= 1, "MACE requires node_max_ell >= 1."
        model = MACEStack(
            "node_attributes, equiv_node_feat, inv_node_feat, edge_attributes, edge_features, edge_index",
            "node_attributes, edge_attributes, edge_features, edge_index",
            radius,
            radial_type,
            distance_transform,
            num_radial,
            edge_dim,
            max_ell,
            node_max_ell,
            avg_num_neighbors,
            envelope_exponent,
            correlation,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
        )

    elif mpnn_type == "HeteroGIN":
        model = HeteroGINStack(
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
            hetero_pooling_mode=hetero_pooling_mode,
            node_target_type=node_target_type,
            share_relation_weights=share_relation_weights,
            metadata=metadata,
        )

    elif mpnn_type == "HeteroSAGE":
        model = HeteroSAGEStack(
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
            hetero_pooling_mode=hetero_pooling_mode,
            node_target_type=node_target_type,
            share_relation_weights=share_relation_weights,
            metadata=metadata,
        )

    elif mpnn_type == "HeteroGAT":
        heads = 6
        negative_slope = 0.05
        model = HeteroGATStack(
            heads,
            negative_slope,
            edge_dim,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
            hetero_pooling_mode=hetero_pooling_mode,
            node_target_type=node_target_type,
            share_relation_weights=share_relation_weights,
            metadata=metadata,
        )

    elif mpnn_type == "HeteroPNA":
        assert pna_deg is not None, "HeteroPNA requires degree input."
        model = HeteroPNAStack(
            pna_deg,
            edge_dim,
            input_dim,
            hidden_dim,
            output_dim,
            pe_dim,
            global_attn_engine,
            global_attn_type,
            global_attn_heads,
            output_type,
            output_heads,
            activation_function,
            loss_function_type,
            equivariance,
            loss_weights=task_weights,
            freeze_conv=freeze_conv,
            initial_bias=initial_bias,
            num_conv_layers=num_conv_layers,
            num_nodes=num_nodes,
            graph_pooling=graph_pooling,
            use_graph_attr_conditioning=use_graph_attr_conditioning,
            graph_attr_conditioning_mode=graph_attr_conditioning_mode,
            hetero_pooling_mode=hetero_pooling_mode,
            node_target_type=node_target_type,
            share_relation_weights=share_relation_weights,
            metadata=metadata,
        )
    else:
        raise ValueError("Unknown mpnn_type: {0}".format(mpnn_type))

    # Apply interatomic potential enhancement if requested
    if enable_interatomic_potential:
        # Instead of complex inheritance, use composition with delegation
        # This avoids MRO issues and __init__ complications
        class EnhancedModelWrapper(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model
                self.energy_weight = energy_weight
                self.energy_peratom_weight = energy_peratom_weight
                self.force_weight = force_weight

            def __getattr__(self, name):
                # First try to get from the wrapper itself
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    pass

                # Then try to get from the wrapped model
                try:
                    return getattr(self.model, name)
                except AttributeError:
                    # Handle specific method names that may be expected for interatomic potentials
                    if name in [
                        "_compute_enhanced_geometric_features",
                        "_compute_three_body_interactions",
                        "_apply_atomic_environment_descriptors",
                    ]:
                        # Return placeholder methods that don't interfere with existing architectures
                        return lambda *args, **kwargs: None
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )

            # ---------- forward ----------
            def forward(self, data):

                return self.model(data)

            def energy_force_loss(self, pred, data):
                """
                Compute energy and force loss for MLIP training.

                This method is specific to interatomic potentials and computes:
                1. Energy loss between predicted and true total energies
                2. Force loss between predicted and true forces (via autograd on positions)

                Forces are computed as negative gradients of total energy with respect to positions.
                """
                # Asserts
                assert (
                    data.pos is not None
                    and data.energy is not None
                    and data.forces is not None
                ), "data.pos, data.energy, data.forces must be provided for energy-force loss. Check your dataset creation and naming."
                assert (
                    data.pos.requires_grad
                ), "data.pos does not have grad, so force predictions cannot be computed. Check that data.pos has grad set to true before prediction."

                assert (
                    self.num_heads == 1
                ), "Force predictions require exactly one head."

                # Support both node and graph heads; enforce sum pooling for graph heads
                if self.head_type[0] == "node":
                    node_energy_pred = pred[0]
                    graph_energy_pred = (
                        torch_scatter.scatter_add(node_energy_pred, data.batch, dim=0)
                        .squeeze()
                        .float()
                    )
                elif self.head_type[0] == "graph":
                    if getattr(self.model, "graph_pooling", "mean") not in ["add"]:
                        raise ValueError(
                            "Graph head force loss requires sum pooling (graph_pooling='add')."
                        )
                    if isinstance(pred, dict) and "graph" in pred:
                        graph_energy_pred = pred["graph"][0].squeeze().float()
                    elif isinstance(pred, (list, tuple)):
                        graph_energy_pred = pred[0].squeeze().float()
                    else:
                        graph_energy_pred = pred.squeeze().float()
                else:
                    raise ValueError(
                        "Force predictions are only supported for node or graph energy heads."
                    )

                graph_energy_true = data.energy.squeeze().float()
                tasks_loss = [self.loss_function(graph_energy_pred, graph_energy_true)]

                energy_loss_weight = self.energy_weight
                energy_peratom_loss_weight = self.energy_peratom_weight
                force_loss_weight = self.force_weight

                # Interatomic potential training requires at least one active loss term
                if (
                    energy_loss_weight <= 0
                    and energy_peratom_loss_weight <= 0
                    and force_loss_weight <= 0
                ):
                    raise ValueError(
                        "All interatomic potential loss weights are zero; set at least one of energy_weight, energy_peratom_weight, or force_weight to a positive value."
                    )

                tot_loss = 0
                if energy_loss_weight > 0:
                    tot_loss += (
                        self.loss_function(graph_energy_pred, graph_energy_true)
                        * energy_loss_weight
                    )

                # Energy per atom
                natoms = torch.bincount(data.batch)
                graph_energy_peratom_pred = graph_energy_pred / natoms
                graph_energy_peratom_true = graph_energy_true / natoms
                tasks_loss.append(
                    self.loss_function(
                        graph_energy_peratom_pred, graph_energy_peratom_true
                    )
                )

                if energy_peratom_loss_weight > 0:
                    tot_loss += (
                        self.loss_function(
                            graph_energy_peratom_pred, graph_energy_peratom_true
                        )
                        * energy_peratom_loss_weight
                    )

                # Forces
                forces_true = data.forces.float()
                forces_pred = torch.autograd.grad(
                    graph_energy_pred,
                    data.pos,
                    grad_outputs=torch.ones_like(graph_energy_pred),
                    retain_graph=graph_energy_pred.requires_grad,
                    # Retain graph only if needed (it will be needed during training, but not during validation/testing)
                    create_graph=True,
                )[0].float()
                assert (
                    forces_pred is not None
                ), "No gradients were found for data.pos. Does your model use positions for prediction?"
                forces_pred = -forces_pred
                tasks_loss.append(self.loss_function(forces_pred, forces_true))

                if force_loss_weight > 0:
                    tot_loss += (
                        self.loss_function(forces_pred, forces_true) * force_loss_weight
                    )  # Have force-weight be the complement to energy-weight
                    ## FixMe: current loss functions require the number of heads to be the number of things being predicted
                    ##        so, we need to do loss calculation manually without calling the other functions.

                return tot_loss, tasks_loss

            def _compute_enhanced_geometric_features(self, data):
                """
                Placeholder for enhanced geometric feature computation (disabled by default).
                """
                return data

            def _compute_three_body_interactions(self, data):
                """
                Placeholder for three-body interaction computation (disabled by default).
                """
                return data

            def _apply_atomic_environment_descriptors(self, data):
                """
                Placeholder for atomic environment descriptor application (disabled by default).
                """
                return data

        enhanced_model = EnhancedModelWrapper(model)
        model = enhanced_model

    if conv_checkpointing:
        model.enable_conv_checkpointing()

    timer.stop()

    return model.to(device)
