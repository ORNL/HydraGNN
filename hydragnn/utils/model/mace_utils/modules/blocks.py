###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
# Taken From:
# GitHub: https://github.com/ACEsuit/mace
# ArXiV: https://arxiv.org/pdf/2206.07697
# Date: August 27, 2024  |  12:37 (EST)
###########################################################################################

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from torch.nn import ModuleList, Sequential, Linear, ModuleDict
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from hydragnn.utils.model.mace_utils.tools.compile import simplify_if_compile
from hydragnn.utils.model.irreps_tools import (
    reshape_irreps,
    tp_out_irreps_with_instructions,
    create_irreps_string,
)

from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)
from .symmetric_contraction import SymmetricContraction


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps("0e"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@simplify_if_compile
@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate: Optional[Callable]
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(
            irreps_in=self.hidden_irreps, acts=[gate]
        )  # Need to adjust this to actually use the gate
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("0e")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class AtomicBlock(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Initialize the atomic energies as a trainable parameter
        self.atomic_energies = torch.nn.Parameter(
            torch.randn(118, output_dim)
        )  # There are 118 known elements

    def forward(self, atomic_numbers):
        # Perform the linear multiplication (no bias)
        return (
            atomic_numbers @ self.atomic_energies
        )  # Output will now have shape [batch_size, output_dim]


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


###########################################################################################
# NOTE: Below is one of the many possible Interaction Blocks in the MACE architecture.
#       Since there are adaptations to the original code in order to be integrated with
#       the HydraGNN framework, and the changes between blocks are relatively minor, we've
#       elected to adapt one general-purpose block here. Users can access the other blocks
#       and adapt similarly from the original MACE code (linked with date at the top).
###########################################################################################
@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.node_feats_down_irreps = o3.Irreps(
            [(o3.Irreps(self.hidden_irreps).count(o3.Irrep(0, 1)), (0, 1))]
        )
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        self.linear_down = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps  # The irreps here should be scalars because they are fed through an activation in self.conv_tp_weights
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        # The following specifies the network architecture for embedding l=0 (scalar) irreps
        ## It is worth double-checking, but I believe this means that type 0 (scalar) irreps
        # are being embedded by 3 layers of size self.hidden_dim (scalar irreps) and the
        # output dim, then activated.
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim]
            + 3 * [o3.Irreps(self.hidden_irreps).count(o3.Irrep(0, 1))]
            + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = (
            irreps_mid.simplify()
        )  # .simplify() essentially combines irreps of the same type so that normalization is done across them all. The site has an in-depth explanation
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.reshape = reshape_irreps(self.irreps_out)

        # Skip connection.
        self.skip_linear = o3.Linear(
            self.node_feats_irreps, self.hidden_irreps
        )  # This will be size (num_nodes, 64*9) when there are 64 channels and irreps, 0, 1, 2 (1+3+5=9)  ## This becomes sc

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter(
            src=mji, index=receiver, dim=0, dim_size=num_nodes, reduce="sum"
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "shift", torch.tensor(shift, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )


###########################################################################################
# The following block are created by the HydraGNN team, mainly for compatibility between
# HydraGNN's Multihead architecture and MACE's Decoding architecture.
###########################################################################################


@compile_mode("script")
class LinearMultiheadDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        input_irreps,
        config_heads,
        head_dims,
        head_type,
        num_heads,
        activation_function,
        num_nodes,
    ):
        # NOTE The readouts of MACE take in irreps of higher order than just scalars. This is fed through o3.Linear
        #      to reduce to scalars. To implement this in HYDRAGNN, the first layer of the node output head
        #      will be such a layer, then all further layers will operate on scalars. Graph-level output heads, on
        #      the other hand, will always operate on the scalar part of the irreps, because pooling may break
        #      equivariance. (To-Do: Check for equivariant pooling methods)

        # NOTE It's a key point of the MACE architecture that all decoders before the last layer are linear. In order
        #      to avoid numerical instability from many stacked linear layers without activation, the MultiheadDecoderBlock
        #      class will be split into linear and nonlinear versions. The nonlinear version stacks layers in the same way
        #      that HYDRAGNN normally would, but the linear version ignores many parameters to have only one layer.

        super(LinearMultiheadDecoderBlock, self).__init__()
        self.input_irreps = input_irreps
        self.config_heads = config_heads
        self.head_dims = head_dims
        self.head_type = head_type
        self.num_heads = num_heads
        self.activation_function = activation_function
        self.num_nodes = num_nodes

        self.graph_shared = ModuleDict({})
        self.heads_NN = ModuleList()

        self.input_scalar_dim = input_irreps.count(o3.Irrep(0, 1))

        self.num_branches = 1
        if "graph" in self.config_heads:
            self.num_branches = len(self.config_heads["graph"])
        elif "node" in self.config_heads:
            self.num_branches = len(self.config_heads["node"])

        for ihead in range(self.num_heads):
            # mlp for each head output
            head_NN = ModuleDict({})
            if self.head_type[ihead] == "graph":
                for branchdict in self.config_heads["graph"]:
                    branchtype = branchdict["type"]
                    denselayers = []
                    denselayers.append(
                        Linear(
                            self.input_scalar_dim,
                            self.head_dims[ihead],
                        )
                    )
                    head_NN[branchtype] = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                for branchdict in self.config_heads["node"]:
                    branchtype = branchdict["type"]
                    brancharct = branchdict["architecture"]
                    node_NN_type = brancharct["type"]
                    if node_NN_type == "mlp" or node_NN_type == "mlp_per_node":
                        self.num_mlp = 1 if node_NN_type == "mlp" else self.num_nodes
                        assert (
                            self.num_nodes is not None
                        ), "num_nodes must be positive integer for MLP"
                        # """if different graphs in the datasets have different size, one MLP is shared across all nodes """
                        head_NN[branchtype] = LinearMLPNode(
                            input_irreps,
                            self.head_dims[ihead],
                            self.num_mlp,
                            node_NN_type,
                            self.activation_function,
                            self.num_nodes,
                        )
                    elif node_NN_type == "conv":
                        raise ValueError(
                            "Node-level convolutional layers are not supported in MACE"
                        )
                    else:
                        raise ValueError(
                            "Unknown head NN structure for node features"
                            + node_NN_type
                            + "; currently only support 'mlp', 'mlp_per_node' or 'conv' (can be set with config['NeuralNetwork']['Architecture']['output_heads']['node']['type'], e.g., ./examples/ci_multihead.json)"
                        )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def forward(self, data, node_features):
        # Take only the type-0 irreps for graph aggregation
        if data.batch is None:
            graph_features = node_features[:, : self.input_scalar_dim].mean(
                dim=0, keepdim=True
            )
            data.batch = data.x * 0
        else:
            graph_features = global_mean_pool(
                node_features[:, : self.input_scalar_dim],
                data.batch.to(node_features.device),
            )
        # if no dataset_name, set it to be 0
        if not hasattr(data, "dataset_name"):
            setattr(data, "dataset_name", data.batch.unique() * 0)
        datasetIDs = data.dataset_name.unique()
        unique, node_counts = torch.unique_consecutive(data.batch, return_counts=True)
        outputs = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                head = torch.zeros(
                    (len(data.dataset_name), head_dim), device=graph_features.device
                )
                if self.num_branches == 1:
                    output_head = headloc["branch-0"](graph_features)
                    head = output_head[:, :head_dim]
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask = mask[:, 0]
                        branchtype = f"branch-{ID.item()}"
                        output_head = headloc[branchtype](graph_features)
                        head[mask] = output_head[:, :head_dim]
                outputs.append(head)
            else:  # Node-level output
                # assuming all node types are the same
                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                head = torch.zeros((data.x.shape[0], head_dim), device=data.x.device)
                if self.num_branches == 1:
                    branchtype = "branch-0"
                    if node_NN_type == "conv":
                        raise ValueError(
                            "Node-level convolutional layers are not supported in MACE"
                        )
                    else:
                        x_node = headloc[branchtype](node_features, data.batch)
                    head = x_node[:, :head_dim]
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask_nodes = torch.repeat_interleave(mask, node_counts)
                        branchtype = f"branch-{ID.item()}"
                        # print("Pei debugging:", branchtype, data.dataset_name, mask, data.dataset_name[mask])
                        if node_NN_type == "conv":
                            raise ValueError(
                                "Node-level convolutional layers are not supported in MACE"
                            )
                        else:
                            x_node = headloc[branchtype](
                                node_features[mask_nodes, :], data.batch[mask_nodes]
                            )
                        head[mask_nodes] = x_node[:, :head_dim]
                outputs.append(head)
        return outputs


@compile_mode("script")
class NonLinearMultiheadDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        input_irreps,
        config_heads,
        head_dims,
        head_type,
        num_heads,
        activation_function,
        num_nodes,
    ):
        # NOTE The readouts of MACE take in irreps of higher order than just scalars. This is fed through o3.Linear
        #      to reduce to scalars. To implement this in HYDRAGNN, the first layer of the node output head
        #      will be such a layer, then all further layers will operate on scalars. Graph-level output heads, on
        #      the other hand, will always operate on the scalar part of the irreps, because pooling may break
        #      equivariance. (To-Do: Check for equivariant pooling methods)

        # NOTE It's a key point of the MACE architecture that all decoders before the last layer are linear. In order
        #      to avoid numerical instability from many stacked linear layers without activation, the MultiheadDecoderBlock
        #      class will be split into linear and nonlinear versions. The nonlinear version stacks layers in the same way
        #      that HYDRAGNN normally would, but the linear version ignores many parameters to have only one layer.

        super(NonLinearMultiheadDecoderBlock, self).__init__()
        self.input_irreps = input_irreps
        self.config_heads = config_heads
        self.head_dims = head_dims
        self.head_type = head_type
        self.num_heads = num_heads
        self.activation_function = activation_function
        self.num_nodes = num_nodes

        self.graph_shared = ModuleDict({})
        self.heads_NN = ModuleList()

        self.input_scalar_dim = input_irreps.count(o3.Irrep(0, 1))

        # Create shared dense layers for graph-level output if applicable
        dim_sharedlayers = 0
        self.num_branches = 1
        if "graph" in self.config_heads:
            self.num_branches = len(self.config_heads["graph"])
            for branchdict in self.config_heads["graph"]:
                denselayers = []
                dim_sharedlayers = branchdict["architecture"]["dim_sharedlayers"]
                denselayers.append(
                    Linear(self.input_scalar_dim, dim_sharedlayers)
                )  # Count scalar irreps for input
                denselayers.append(self.activation_function)
                for ishare in range(branchdict["architecture"]["num_sharedlayers"] - 1):
                    denselayers.append(Linear(dim_sharedlayers, dim_sharedlayers))
                    denselayers.append(self.activation_function)
                self.graph_shared[branchdict["type"]] = Sequential(*denselayers)

        for ihead in range(self.num_heads):
            # mlp for each head output
            head_NN = ModuleDict({})
            if self.head_type[ihead] == "graph":
                for branchdict in self.config_heads["graph"]:
                    branchtype = branchdict["type"]
                    brancharct = branchdict["architecture"]
                    dim_sharedlayers = brancharct["dim_sharedlayers"]
                    num_head_hidden = brancharct["num_headlayers"]
                    dim_head_hidden = brancharct["dim_headlayers"]
                    denselayers = []
                    denselayers.append(Linear(dim_sharedlayers, dim_head_hidden[0]))
                    denselayers.append(self.activation_function)
                    for ilayer in range(num_head_hidden - 1):
                        denselayers.append(
                            Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                        )
                        denselayers.append(self.activation_function)
                    denselayers.append(
                        Linear(
                            dim_head_hidden[-1],
                            self.head_dims[ihead],
                        )
                    )
                    head_NN[branchtype] = Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                for branchdict in self.config_heads["node"]:
                    branchtype = branchdict["type"]
                    brancharct = branchdict["architecture"]
                    hidden_dim_node = brancharct["dim_headlayers"]
                    node_NN_type = brancharct["type"]
                    if node_NN_type == "mlp" or node_NN_type == "mlp_per_node":
                        self.num_mlp = 1 if node_NN_type == "mlp" else self.num_nodes
                        assert (
                            self.num_nodes is not None
                        ), "num_nodes must be positive integer for MLP"
                        # """if different graphs in the datasets have different size, one MLP is shared across all nodes """
                        head_NN[branchtype] = NonLinearMLPNode(
                            input_irreps,
                            self.head_dims[ihead],
                            self.num_mlp,
                            hidden_dim_node,
                            node_NN_type,
                            self.activation_function,
                            self.num_nodes,
                        )
                    elif node_NN_type == "conv":
                        raise ValueError(
                            "Node-level convolutional layers are not supported in MACE"
                        )
                    else:
                        raise ValueError(
                            "Unknown head NN structure for node features"
                            + node_NN_type
                            + "; currently only support 'mlp', 'mlp_per_node' or 'conv' (can be set with config['NeuralNetwork']['Architecture']['output_heads']['node']['type'], e.g., ./examples/ci_multihead.json)"
                        )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def forward(self, data, node_features):
        # Take only the type-0 irreps for graph aggregation
        if data.batch is None:
            graph_features = node_features[:, : self.input_scalar_dim].mean(
                dim=0, keepdim=True
            )
            data.batch = data.x * 0
        else:
            graph_features = global_mean_pool(
                node_features[:, : self.input_scalar_dim],
                data.batch.to(node_features.device),
            )
        # if no dataset_name, set it to be 0
        if not hasattr(data, "dataset_name"):
            setattr(data, "dataset_name", data.batch.unique() * 0)
        datasetIDs = data.dataset_name.unique()
        unique, node_counts = torch.unique_consecutive(data.batch, return_counts=True)
        outputs = []

        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                head = torch.zeros(
                    (len(data.dataset_name), head_dim), device=graph_features.device
                )
                if self.num_branches == 1:
                    x_graph_head = self.graph_shared["branch-0"](graph_features)
                    output_head = headloc["branch-0"](x_graph_head)
                    head = output_head[:, :head_dim]
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask = mask[:, 0]
                        branchtype = f"branch-{ID.item()}"
                        x_graph_head = self.graph_shared[branchtype](
                            graph_features[mask, :]
                        )
                        output_head = headloc[branchtype](x_graph_head)
                        head[mask] = output_head[:, :head_dim]
                outputs.append(head)
            else:  # Node-level output
                # assuming all node types are the same
                node_NN_type = self.config_heads["node"][0]["architecture"]["type"]
                head = torch.zeros((data.x.shape[0], head_dim), device=data.x.device)
                if self.num_branches == 1:
                    branchtype = "branch-0"
                    if node_NN_type == "conv":
                        raise ValueError(
                            "Node-level convolutional layers are not supported in MACE"
                        )
                    else:
                        x_node = headloc[branchtype](node_features, data.batch)
                    head = x_node[:, :head_dim]
                else:
                    for ID in datasetIDs:
                        mask = data.dataset_name == ID
                        mask_nodes = torch.repeat_interleave(mask, node_counts)
                        branchtype = f"branch-{ID.item()}"
                        # print("Pei debugging:", branchtype, data.dataset_name, mask, data.dataset_name[mask])
                        if node_NN_type == "conv":
                            raise ValueError(
                                "Node-level convolutional layers are not supported in MACE"
                            )
                        else:
                            x_node = headloc[branchtype](
                                node_features[mask_nodes, :], data.batch[mask_nodes]
                            )
                        head[mask_nodes] = x_node[:, :head_dim]
                outputs.append(head)
        return outputs


@compile_mode("script")
class LinearMLPNode(torch.nn.Module):
    def __init__(
        self,
        input_irreps,
        output_dim,
        # No longer need hidden_dim_node because there is only one layer
        num_mlp,
        node_type,
        activation_function,
        num_nodes,
    ):
        super().__init__()
        self.input_irreps = input_irreps
        self.output_dim = output_dim
        self.num_mlp = num_mlp
        self.node_type = node_type
        self.activation_function = activation_function
        self.num_nodes = num_nodes

        self.mlp = ModuleList()
        for _ in range(self.num_mlp):
            denselayers = []
            output_irreps = o3.Irreps(create_irreps_string(output_dim, 0))
            denselayers.append(
                o3.Linear(input_irreps, output_irreps)
            )  # First layer is o3.Linear and takes all irreps down to scalars
            self.mlp.append(Sequential(*denselayers))

    def node_features_reshape(self, x, batch):
        """reshape x from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = x.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=x.dtype,
            device=x.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = x[inode_index, :]
        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        if self.node_type == "mlp":
            outs = self.mlp[0](x)
        else:
            outs = torch.zeros(
                (x.shape[0], self.output_dim),
                dtype=x.dtype,
                device=x.device,
            )
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"


@compile_mode("script")
class NonLinearMLPNode(torch.nn.Module):
    def __init__(
        self,
        input_irreps,
        output_dim,
        num_mlp,
        hidden_dim_node,
        node_type,
        activation_function,
        num_nodes,
    ):
        super().__init__()
        self.input_irreps = input_irreps
        self.output_dim = output_dim
        self.num_mlp = num_mlp
        self.node_type = node_type
        self.activation_function = activation_function
        self.num_nodes = num_nodes

        self.mlp = ModuleList()
        for _ in range(self.num_mlp):
            denselayers = []
            hidden_irreps = o3.Irreps(create_irreps_string(hidden_dim_node[0], 0))
            denselayers.append(
                o3.Linear(input_irreps, hidden_irreps)
            )  # First layer is o3.Linear and takes all irreps down to scalars
            denselayers.append(self.activation_function)
            for ilayer in range(len(hidden_dim_node) - 1):
                denselayers.append(
                    Linear(hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1])
                )
                denselayers.append(self.activation_function)
            denselayers.append(Linear(hidden_dim_node[-1], output_dim))
            self.mlp.append(Sequential(*denselayers))

    def node_features_reshape(self, x, batch):
        """reshape x from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = x.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=x.dtype,
            device=x.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = x[inode_index, :]
        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        if self.node_type == "mlp":
            outs = self.mlp[0](x)
        else:
            outs = torch.zeros(
                (x.shape[0], self.output_dim),
                dtype=x.dtype,
                device=x.device,
            )
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"


@compile_mode("script")
class CombineBlock(torch.nn.Module):
    def __init__(self):
        super(CombineBlock, self).__init__()

    def forward(self, inv_node_features, equiv_node_features):
        return torch.cat([inv_node_features, equiv_node_features], dim=1)


@compile_mode("script")
class SplitBlock(torch.nn.Module):
    def __init__(self, irreps):
        super(SplitBlock, self).__init__()
        self.dim = irreps.count(o3.Irrep(0, 1))

    def forward(self, node_features):
        return node_features[:, : self.dim], node_features[:, self.dim :]
