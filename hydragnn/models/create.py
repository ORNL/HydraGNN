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
                
                # Add interatomic potential attributes
                self.radius = getattr(original_model, 'radius', 6.0)
                self.max_neighbours = getattr(original_model, 'max_neighbours', 50)
                
                # Enhanced features are disabled by default to avoid interference
                # with native message passing architectures (MACE, DimeNet++, etc.)
                self.use_enhanced_geometry = False
                self.use_three_body_interactions = False
                self.use_atomic_environment_descriptors = False

            def __getattr__(self, name):
                # Delegate all attribute access to the wrapped model
                # This makes the wrapper transparent for most operations
                try:
                    return getattr(self.model, name)
                except AttributeError:
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            def forward(self, data):
                """Enhanced forward method with interatomic potential capabilities."""
                # Import here to avoid circular imports
                from torch_geometric.nn import radius_graph
                
                # For MLIPs, we need to reconstruct the graph from atomic positions
                # to ensure proper gradient flow for force calculations
                if hasattr(data, 'pos') and hasattr(data, 'batch'):
                    # Dynamically construct the graph based on atomic positions and cutoff radius
                    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.size(1) == 0:
                        edge_index = radius_graph(
                            data.pos, 
                            r=self.radius, 
                            batch=data.batch, 
                            max_num_neighbors=self.max_neighbours
                        )
                        data.edge_index = edge_index
                
                # Use the original model's forward method
                return self.model.forward(data)

        enhanced_model = EnhancedModelWrapper(model)
        model = enhanced_model

    if conv_checkpointing:
        model.enable_conv_checkpointing()

    timer.stop()

    return model.to(device)
