import torch
from torch_geometric.nn import GATConv, HeteroConv

from hydragnn.globalAtt.HeteroGPS import (
    HeteroGPSConv,
    StructuralCoordinatePerformerAttention,
)
from hydragnn.globalAtt.structural import StructuralAttentionContext


class DummyEquivariantLocalConv(torch.nn.Module):
    def forward(
        self,
        inv_node_feat_dict,
        equiv_node_feat_dict,
        edge_index_dict,
        edge_attr_dict=None,
    ):
        inv_out = {node_type: x + 1.0 for node_type, x in inv_node_feat_dict.items()}
        equiv_out = {
            node_type: v + 2.0 for node_type, v in equiv_node_feat_dict.items()
        }
        return inv_out, equiv_out


class DummyInvariantLocalConv(torch.nn.Module):
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return {node_type: x + 1.0 for node_type, x in x_dict.items()}


def test_hetero_gps_supports_equivariant_local_branch_and_preserves_equivariant_output():
    torch.manual_seed(0)

    metadata = (["a", "b"], [("a", "r", "a")])
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=DummyEquivariantLocalConv(),
        heads=1,
        dropout=0.0,
        attn_type="multihead",
    )
    conv.eval()

    x_dict = {
        "a": torch.randn(3, 4),
        "b": torch.randn(2, 4),
    }
    equiv_dict = {
        "a": torch.randn(3, 3, 4),
        "b": torch.randn(2, 3, 4),
    }
    batch_dict = {
        "a": torch.tensor([0, 0, 1], dtype=torch.long),
        "b": torch.tensor([0, 1], dtype=torch.long),
    }

    inv_out, equiv_out = conv(
        inv_node_feat_dict=x_dict,
        equiv_node_feat_dict=equiv_dict,
        edge_index_dict={},
        batch_dict=batch_dict,
    )

    assert set(inv_out.keys()) == set(x_dict.keys())
    assert set(equiv_out.keys()) == set(equiv_dict.keys())
    assert torch.allclose(equiv_out["a"], equiv_dict["a"] + 2.0)
    assert torch.allclose(equiv_out["b"], equiv_dict["b"] + 2.0)


def test_hetero_gps_backward_compatible_with_invariant_local_conv():
    torch.manual_seed(1)

    metadata = (["a", "b"], [("a", "r", "a")])
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=DummyInvariantLocalConv(),
        heads=1,
        dropout=0.0,
        attn_type="multihead",
    )
    conv.eval()

    x_dict = {
        "a": torch.randn(2, 4),
        "b": torch.randn(2, 4),
    }
    batch_dict = {
        "a": torch.tensor([0, 1], dtype=torch.long),
        "b": torch.tensor([0, 1], dtype=torch.long),
    }

    inv_out, equiv_out = conv(x_dict, {}, batch_dict)

    assert set(inv_out.keys()) == set(x_dict.keys())
    assert equiv_out is None


def test_hetero_gps_forwards_edge_attributes_to_pyg_hetero_conv():
    torch.manual_seed(2)

    edge_type = ("a", "r", "a")
    metadata = (["a"], [edge_type])
    local_conv = HeteroConv(
        {
            edge_type: GATConv(
                in_channels=4,
                out_channels=4,
                heads=1,
                edge_dim=3,
                add_self_loops=False,
            )
        },
        aggr="sum",
    )
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=local_conv,
        heads=1,
        dropout=0.0,
        attn_type="multihead",
    )
    conv.eval()

    x_dict = {"a": torch.randn(3, 4)}
    edge_index_dict = {
        edge_type: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    }
    edge_attr_dict = {edge_type: torch.randn(3, 3)}
    batch_dict = {"a": torch.zeros(3, dtype=torch.long)}

    inv_out, equiv_out = conv(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        batch_dict=batch_dict,
        edge_attr_dict=edge_attr_dict,
    )

    assert inv_out["a"].shape == x_dict["a"].shape
    assert equiv_out is None


def test_hetero_gps_attention_isolation_across_graphs():
    torch.manual_seed(2)

    metadata = (["a", "b"], [("a", "r", "a")])
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=None,
        heads=1,
        dropout=0.0,
        attn_type="multihead",
    )
    conv.eval()

    # Graph 0 has nodes a[0], b[0]; Graph 1 has nodes a[1], b[1].
    x_dict = {
        "a": torch.randn(2, 4),
        "b": torch.randn(2, 4),
    }
    batch_dict = {
        "a": torch.tensor([0, 1], dtype=torch.long),
        "b": torch.tensor([0, 1], dtype=torch.long),
    }

    out_ref, _ = conv(
        inv_node_feat_dict=x_dict,
        equiv_node_feat_dict=None,
        edge_index_dict={},
        batch_dict=batch_dict,
    )

    x_perturbed = {
        "a": x_dict["a"].clone(),
        "b": x_dict["b"].clone(),
    }
    # Perturb graph 1 only.
    x_perturbed["a"][1] += 10.0
    x_perturbed["b"][1] += 10.0

    out_perturbed, _ = conv(
        inv_node_feat_dict=x_perturbed,
        equiv_node_feat_dict=None,
        edge_index_dict={},
        batch_dict=batch_dict,
    )

    assert torch.allclose(out_ref["a"][0], out_perturbed["a"][0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_ref["b"][0], out_perturbed["b"][0], atol=1e-6, rtol=1e-6)


def test_hetero_gps_attention_node_types_default_to_all_types():
    metadata = (["a", "b"], [("a", "r", "a")])
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=None,
        heads=1,
        dropout=0.0,
        attn_type="multihead",
    )

    assert conv.attn_node_types == ["a", "b"]


def test_hetero_gps_attention_node_types_can_be_configured_independently():
    metadata = (["a", "b"], [("a", "r", "a")])
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=None,
        heads=1,
        dropout=0.0,
        attn_type="multihead",
        attn_node_types=["b"],
    )
    x_dict = {
        "a": torch.randn(2, 4),
        "b": torch.randn(3, 4),
    }
    batch_dict = {
        "a": torch.zeros(2, dtype=torch.long),
        "b": torch.zeros(3, dtype=torch.long),
    }

    _, _, split_sizes, pack_node_types = conv._pack_x_dict(x_dict, batch_dict)

    assert conv.attn_node_types == ["b"]
    assert pack_node_types == ["b"]
    assert split_sizes == [3]


def test_hetero_gps_rejects_unknown_attention_node_types():
    metadata = (["a", "b"], [("a", "r", "a")])

    try:
        HeteroGPSConv(
            channels=4,
            metadata=metadata,
            conv=None,
            heads=1,
            dropout=0.0,
            attn_type="multihead",
            attn_node_types=["missing"],
        )
    except ValueError as exc:
        assert "Unknown attention node types" in str(exc)
    else:
        raise AssertionError("Expected unknown attention node type to be rejected.")


def test_pairwise_features_are_not_tied_to_an_opf_node_name():
    metadata = (["a", "entity"], [("entity", "r", "entity")])
    conv = HeteroGPSConv(
        channels=4,
        metadata=metadata,
        conv=None,
        heads=1,
        dropout=0.0,
        attn_type="multihead",
        attn_node_types=["entity"],
        pairwise_feature_dim=2,
    )

    assert conv.attn_node_types == ["entity"]
    assert conv.pairwise_feature_dim == 2


def test_structural_coordinate_performer_has_domain_neutral_api():
    attention = StructuralCoordinatePerformerAttention(
        channels=4,
        heads=1,
        coordinate_dim=3,
        num_random_features=4,
    )

    assert attention.coordinate_dim == 3


def test_attention_accepts_structural_context():
    conv = HeteroGPSConv(
        channels=4,
        metadata=(["entity"], []),
        conv=None,
        heads=1,
        dropout=0.0,
        attn_type="multihead",
        attn_node_types=["entity"],
        pairwise_feature_dim=1,
    )
    context = StructuralAttentionContext(pairwise_features=[torch.zeros(3, 3, 1)])

    output, _ = conv(
        inv_node_feat_dict={"entity": torch.randn(3, 4)},
        edge_index_dict={},
        batch_dict={"entity": torch.zeros(3, dtype=torch.long)},
        structural_context=context,
    )

    assert output["entity"].shape == (3, 4)
