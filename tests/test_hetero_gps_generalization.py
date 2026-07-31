import torch

from hydragnn.globalAtt.HeteroGPS import HeteroGPSConv


class DummyEquivariantLocalConv(torch.nn.Module):
    def forward(
        self,
        inv_node_feat_dict,
        equiv_node_feat_dict,
        edge_index_dict,
        edge_attr_dict=None,
    ):
        inv_out = {
            node_type: x + 1.0 for node_type, x in inv_node_feat_dict.items()
        }
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
