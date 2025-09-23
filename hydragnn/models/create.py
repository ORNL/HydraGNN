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

# InteratomicPotential functionality is now implemented via wrapper composition

from hydragnn.utils.distributed import get_device
from hydragnn.utils.profiling_and_tracing.time_utils import Timer


def create_model_config(
    config: dict,
    verbosity: int = 0,
    use_gpu: bool = True,
):
    return create_model(
        config["Architecture"]["mpnn_type"],
        config["Architecture"]["input_dim"],
        config["Architecture"]["hidden_dim"],
        config["Architecture"]["output_dim"],
        config["Architecture"]["pe_dim"],
        config["Architecture"]["global_attn_engine"],
        config["Architecture"]["global_attn_type"],
        config["Architecture"]["global_attn_heads"],
        config["Architecture"]["output_type"],
        config["Architecture"]["output_heads"],
        config["Architecture"]["activation_function"],
        config["Training"]["loss_function_type"],
        config["Architecture"]["task_weights"],
        config["Architecture"]["num_conv_layers"],
        config["Architecture"]["freeze_conv_layers"],
        config["Architecture"]["initial_bias"],
        config["Architecture"]["num_nodes"],
        config["Architecture"]["max_neighbours"],
        config["Architecture"]["edge_dim"],
        config["Architecture"]["pna_deg"],
        config["Architecture"]["num_before_skip"],
        config["Architecture"]["num_after_skip"],
        config["Architecture"]["num_radial"],
        config["Architecture"]["radial_type"],
        config["Architecture"]["distance_transform"],
        config["Architecture"]["basis_emb_size"],
        config["Architecture"]["int_emb_size"],
        config["Architecture"]["out_emb_size"],
        config["Architecture"]["envelope_exponent"],
        config["Architecture"]["num_spherical"],
        config["Architecture"]["num_gaussians"],
        config["Architecture"]["num_filters"],
        config["Architecture"]["radius"],
        config["Architecture"]["equivariance"],
        config["Architecture"]["correlation"],
        config["Architecture"]["max_ell"],
        config["Architecture"]["node_max_ell"],
        config["Architecture"]["avg_num_neighbors"],
        config["Training"]["conv_checkpointing"],
        config["Architecture"].get("enable_interatomic_potential", False),
        verbosity,
        use_gpu,
    )


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
        )

    elif mpnn_type == "EGNN":
        model = EGCLStack(
            "inv_node_feat, equiv_node_feat, edge_index, edge_attr",  # input_args
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
                    self.num_heads == 1 and self.head_type[0] == "node"
                ), "Force predictions are only supported for models with one head that predict nodal energy. Check your num_heads and head_types."
                # Initialize loss
                tot_loss = 0
                tasks_loss = []
                # Energies
                node_energy_pred = pred[0]
                graph_energy_pred = (
                    torch_scatter.scatter_add(node_energy_pred, data.batch, dim=0)
                    .squeeze()
                    .float()
                )
                graph_energy_true = data.energy.squeeze().float()
                energy_loss_weight = self.loss_weights[
                    0
                ]  # There should only be one loss-weight for energy
                tot_loss += (
                    self.loss_function(graph_energy_pred, graph_energy_true)
                    * energy_loss_weight
                )
                tasks_loss.append(
                    self.loss_function(graph_energy_pred, graph_energy_true)
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
                force_loss_weight = (
                    energy_loss_weight
                    * torch.mean(torch.abs(graph_energy_true))
                    / (torch.mean(torch.abs(forces_true)) + 1e-8)
                )  # Weight force loss and graph energy equally
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
