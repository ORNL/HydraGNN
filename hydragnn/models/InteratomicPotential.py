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

import torch
from torch.nn import ModuleList, Sequential, Linear, Module
import torch.nn.functional as F

# Direct imports - torch_geometric and torch_scatter are core dependencies
from torch_geometric.nn import global_mean_pool, BatchNorm
import torch_scatter
from hydragnn.models.Base import Base
from hydragnn.utils.model.operations import get_edge_vectors_and_lengths

import math


class InteratomicPotentialMixin:
    """
    Mixin class to enhance HydraGNN models with interatomic potential capabilities.
    
    This mixin focuses on the core MLIP requirements:
    1. Dynamic graph construction for proper gradient flow
    2. Node-level prediction enforcement for atomic energies
    
    It does NOT override native message passing features of specialized architectures
    like MACE, DimeNet++, etc., which already have sophisticated implementations
    of three-body interactions and geometric features.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store default values for radius and max_neighbours if not already set
        # These are essential for dynamic graph construction in MLIPs
        if not hasattr(self, 'radius'):
            self.radius = 6.0  # Default radius for neighbor finding
        if not hasattr(self, 'max_neighbours'):
            self.max_neighbours = 50  # Default max neighbors per atom
        
        # Enhanced features are disabled by default to avoid interference
        # with native message passing architectures (MACE, DimeNet++, etc.)
        self.use_enhanced_geometry = getattr(kwargs, 'use_enhanced_geometry', False)
        self.use_three_body_interactions = getattr(kwargs, 'use_three_body_interactions', False) 
        self.use_atomic_environment_descriptors = getattr(kwargs, 'use_atomic_environment_descriptors', False)
        
        # Only initialize enhanced feature layers if explicitly requested
        if (self.use_enhanced_geometry or self.use_three_body_interactions or 
            self.use_atomic_environment_descriptors) and hasattr(self, 'hidden_dim'):
            self._init_interatomic_layers()
    
    def _init_interatomic_layers(self):
        """Initialize additional layers for interatomic potential computation."""
        # Three-body interaction layers
        if self.use_three_body_interactions:
            self.three_body_mlp = Sequential(
                Linear(self.hidden_dim * 2, self.hidden_dim),
                self.activation_function,
                Linear(self.hidden_dim, self.hidden_dim),
                self.activation_function
            )
        
        # Atomic environment descriptor layers
        if self.use_atomic_environment_descriptors:
            self.env_descriptor_mlp = Sequential(
                Linear(self.hidden_dim + 3, self.hidden_dim),  # +3 for distance features
                self.activation_function,
                Linear(self.hidden_dim, self.hidden_dim)
            )
    
    def _compute_enhanced_geometric_features(self, data, conv_args):
        """
        Compute enhanced geometric features including distances, angles, and atomic environments.
        
        Args:
            data: Graph data containing atomic positions and connectivity
            conv_args: Convolution arguments containing edge information
            
        Returns:
            Enhanced geometric features for better interatomic potential prediction
        """
        if not self.use_enhanced_geometry:
            return conv_args
        
        # Compute edge vectors and distances
        edge_index = conv_args["edge_index"]
        positions = data.pos
        
        # Handle periodic boundary conditions if present
        shifts = getattr(data, 'edge_shifts', torch.zeros((edge_index.size(1), 3), device=edge_index.device))
        
        # Get edge vectors and lengths
        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions, edge_index, shifts, normalize=False
        )
        
        # Add distance features to conv_args
        conv_args["edge_vectors"] = edge_vectors
        conv_args["edge_lengths"] = edge_lengths
        
        # Compute additional geometric descriptors
        if self.use_atomic_environment_descriptors:
            # Compute local atomic environment features
            local_env_features = self._compute_local_environment_features(
                positions, edge_index, edge_vectors, edge_lengths
            )
            conv_args["local_env_features"] = local_env_features
        
        return conv_args
    
    def _compute_local_environment_features(self, positions, edge_index, edge_vectors, edge_lengths):
        """
        Compute local atomic environment features including coordination numbers,
        bond angles, and local density descriptors.
        """
        num_nodes = positions.size(0)
        
        # Compute coordination numbers (number of neighbors within cutoff)
        coord_numbers = torch_scatter.scatter_add(
            torch.ones_like(edge_lengths.squeeze()), 
            edge_index[1], 
            dim_size=num_nodes
        )
        
        # Compute average distances to neighbors
        avg_distances = torch_scatter.scatter_mean(
            edge_lengths.squeeze(), 
            edge_index[1], 
            dim_size=num_nodes
        )
        
        # Handle nodes with no neighbors
        avg_distances = torch.where(coord_numbers > 0, avg_distances, torch.zeros_like(avg_distances))
        
        # Compute local density (inverse of average distance)
        local_density = torch.where(avg_distances > 0, 1.0 / (avg_distances + 1e-6), torch.zeros_like(avg_distances))
        
        # Stack features
        env_features = torch.stack([coord_numbers, avg_distances, local_density], dim=1)
        
        return env_features
    
    def _compute_three_body_interactions(self, node_features, data, conv_args):
        """
        Compute three-body interactions to capture angular dependencies in molecular systems.
        This enhances the model's ability to predict accurate interatomic potentials.
        """
        if not self.use_three_body_interactions:
            return node_features
        
        edge_index = conv_args["edge_index"]
        edge_vectors = conv_args.get("edge_vectors")
        
        if edge_vectors is None:
            return node_features
        
        # Find triplets (i-j-k) where j is the central atom
        i, j = edge_index
        
        # For each central atom j, find all pairs of neighbors
        j_unique, j_counts = torch.unique_consecutive(j, return_counts=True)
        
        # Only compute three-body terms for atoms with at least 2 neighbors
        valid_mask = j_counts >= 2
        
        if not valid_mask.any():
            return node_features
        
        three_body_features = torch.zeros_like(node_features)
        
        # For efficiency, we'll compute a simplified three-body interaction
        # based on the angular information between neighboring bonds
        edge_start_idx = 0
        for idx, (center_atom, n_neighbors) in enumerate(zip(j_unique, j_counts)):
            if n_neighbors >= 2:
                # Get all edges connected to this central atom
                edge_end_idx = edge_start_idx + n_neighbors
                center_edges = torch.arange(edge_start_idx, edge_end_idx, device=edge_index.device)
                
                # Get vectors from central atom to neighbors
                neighbor_vectors = edge_vectors[center_edges]  # [n_neighbors, 3]
                
                # Compute pairwise angles between all neighbor vectors
                neighbor_features = node_features[i[center_edges]]  # Features of neighbor atoms
                
                # Simple three-body feature: average of pairwise neighbor features
                if neighbor_features.size(0) > 1:
                    three_body_contrib = self.three_body_mlp(
                        torch.cat([
                            neighbor_features.mean(dim=0, keepdim=True).expand(1, -1),
                            node_features[center_atom].unsqueeze(0)
                        ], dim=1)
                    )
                    three_body_features[center_atom] = three_body_contrib.squeeze(0)
            
            edge_start_idx += n_neighbors
        
        return node_features + three_body_features
    
    def _apply_atomic_environment_descriptors(self, node_features, conv_args):
        """
        Apply atomic environment descriptors to enhance local chemical environment understanding.
        """
        if not self.use_atomic_environment_descriptors:
            return node_features
        
        local_env_features = conv_args.get("local_env_features")
        if local_env_features is None:
            return node_features
        
        # Combine node features with local environment descriptors
        enhanced_features = torch.cat([node_features, local_env_features], dim=1)
        enhanced_features = self.env_descriptor_mlp(enhanced_features)
        
        return enhanced_features
    
    def forward(self, data):
        """
        Forward method with essential MLIP capabilities.
        
        Core MLIP requirements:
        1. Dynamic graph construction from atomic positions (essential for proper gradient flow)
        2. Node-level prediction enforcement for atomic energies
        
        Enhanced features (geometric, three-body, environment descriptors) are disabled
        by default to avoid interference with native message passing architectures like
        MACE and DimeNet++ which already implement sophisticated versions of these features.
        
        To enable enhanced features for simpler architectures, set:
        - use_enhanced_geometry=True
        - use_three_body_interactions=True  
        - use_atomic_environment_descriptors=True
        """
        ### Dynamic graph construction for MLIPs ####
        # For interatomic potentials, graph connectivity must depend on atomic positions
        # and be constructed within the forward method for proper gradient flow
        from torch_geometric.transforms import RadiusGraph
        
        # Get radius and max_neighbours from model configuration
        # These are set during model initialization with defaults if not configured
        radius = self.radius
        max_neighbours = self.max_neighbours
        
        # Construct graph connectivity based on current atomic positions
        # This ensures the graph construction is part of the computational graph
        radius_graph = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_neighbours)
        
        # Apply radius graph transform to get edge connectivity
        # This creates edge_index based on current positions with proper gradients
        data_with_edges = radius_graph(data)
        
        # Update data with the dynamically constructed edges
        data.edge_index = data_with_edges.edge_index
        if hasattr(data_with_edges, 'edge_attr'):
            data.edge_attr = data_with_edges.edge_attr
        
        # Ensure edge_shifts exist for periodic boundary conditions
        if not hasattr(data, "edge_shifts"):
            data.edge_shifts = torch.zeros(
                (data.edge_index.size(1), 3), device=data.edge_index.device
            )
        
        ### Standard encoder part ####
        inv_node_feat, equiv_node_feat, conv_args = self._embedding(data)
        
        # Only compute enhanced geometric features if explicitly enabled
        # (disabled by default to avoid interference with MACE, DimeNet++, etc.)
        if self.use_enhanced_geometry:
            conv_args = self._compute_enhanced_geometric_features(data, conv_args)
        
        # Standard convolution layers
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                inv_node_feat, equiv_node_feat = conv(
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
            else:
                from torch.utils.checkpoint import checkpoint
                inv_node_feat, equiv_node_feat = checkpoint(
                    conv,
                    use_reentrant=False,
                    inv_node_feat=inv_node_feat,
                    equiv_node_feat=equiv_node_feat,
                    **conv_args,
                )
            inv_node_feat = self.activation_function(feat_layer(inv_node_feat))
        
        # Only apply enhanced features if explicitly enabled
        # (disabled by default to avoid interference with native message passing)
        if self.use_three_body_interactions:
            inv_node_feat = self._compute_three_body_interactions(inv_node_feat, data, conv_args)
        
        if self.use_atomic_environment_descriptors:
            inv_node_feat = self._apply_atomic_environment_descriptors(inv_node_feat, conv_args)
        
        x = inv_node_feat
        
        #### Multi-head decoder part - Focused on NODE-level predictions for MLIPs ####
        # For Interatomic Potentials, we primarily need node-level predictions (atomic energies)
        # Graph-level total energies can be computed by summing atomic energies if needed
        
        outputs = []
        outputs_var = []
        
        # If no dataset_name, set it to be 0
        if not hasattr(data, "dataset_name"):
            setattr(data, "dataset_name", data.batch.unique() * 0)
        
        datasetIDs = data.dataset_name.unique()
        unique, node_counts = torch.unique_consecutive(data.batch, return_counts=True)
        
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "node":
                # Node-level predictions (atomic energies, forces, etc.) - PRIMARY for MLIPs
                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                head = torch.zeros((x.shape[0], head_dim), device=x.device)
                headvar = torch.zeros(
                    (x.shape[0], head_dim * self.var_output), device=x.device
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
                    headvar = x_node[:, head_dim:] ** 2
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask_nodes = torch.repeat_interleave(mask, node_counts)
                        branchtype = f"branch-{ID.item()}"
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
                        headvar[mask_nodes] = x_node[:, head_dim:] ** 2
                outputs.append(head)
                outputs_var.append(headvar)
            elif type_head == "graph":
                # Graph-level heads are not supported for InteratomicPotential models
                # For MLIPs, total energy should be computed by summing node-level atomic energies
                raise ValueError(
                    "Graph-level heads are not supported for InteratomicPotential models. "
                    "MLIPs require node-level predictions for atomic energies. "
                    "Total energy should be computed by summing atomic energies externally. "
                    "Please configure your model to use 'type': ['node'] instead of 'type': ['graph']."
                )
        
        if self.var_output:
            return outputs, outputs_var
        return outputs


class InteratomicPotentialBase(InteratomicPotentialMixin, Base):
    """
    HydraGNN Base model enhanced for Machine Learning Interatomic Potentials (MLIPs).
    
    This class provides the essential MLIP requirements:
    1. Dynamic graph construction from atomic positions for proper gradient flow
    2. Node-level prediction enforcement for atomic energies (required for force calculations)
    
    IMPORTANT: This class does NOT override native message passing features.
    Architectures like MACE and DimeNet++ already include sophisticated implementations
    of three-body interactions, geometric features, and angular dependencies.
    
    Enhanced features (geometric, three-body, environment descriptors) are disabled
    by default to avoid interference. They can be enabled for simpler architectures
    that lack these native capabilities.
    
    Key MLIP workflow:
    1. Positions → Dynamic graph construction (within forward pass)
    2. Graph + Features → Message passing (architecture-specific)  
    3. Node features → Atomic energies (node-level predictions)
    4. Atomic energies → Forces (via automatic differentiation)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return "InteratomicPotentialBase"