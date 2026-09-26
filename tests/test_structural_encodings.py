import io

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from hydragnn.globalAtt.HeteroGPS import HeteroGPSConv
from hydragnn.globalAtt.structural import (
    StructuralAttentionContext,
    StructuralEncodingProvider,
)
from hydragnn.models.heterogeneous import HeteroSAGEStack


def _attention(
    *,
    node_types=("entity",),
    attn_node_types=("entity",),
    attn_type="multihead",
    pairwise_feature_dim=0,
    qk_coordinate_dim=0,
):
    return HeteroGPSConv(
        channels=4,
        metadata=(list(node_types), []),
        conv=None,
        heads=1,
        dropout=0.0,
        norm=None,
        attn_type=attn_type,
        attn_node_types=list(attn_node_types),
        pairwise_feature_dim=pairwise_feature_dim,
        qk_coordinate_dim=qk_coordinate_dim,
    )


def _hetero_stack(structural_encoding):
    return HeteroSAGEStack(
        edge_dim=0,
        input_dim=3,
        hidden_dim=4,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        config_heads={
            "node": [
                {
                    "type": "branch-0",
                    "architecture": {
                        "num_headlayers": 1,
                        "dim_headlayers": [4],
                        "type": "mlp",
                    },
                }
            ]
        },
        activation_function_type="relu",
        loss_function_type="mse",
        equivariance=False,
        num_conv_layers=1,
        node_target_type="entity",
        metadata=(["entity"], []),
        node_input_dims={"entity": 3},
        structural_encoding=structural_encoding,
    )


def test_non_opf_pairwise_attention_runs_forward_and_backward():
    torch.manual_seed(4)
    conv = _attention(pairwise_feature_dim=2)
    x = torch.randn(4, 4, requires_grad=True)
    pairwise = torch.randn(4, 4, 2)
    context = StructuralAttentionContext(pairwise_features=[pairwise])

    output, _ = conv(
        inv_node_feat_dict={"entity": x},
        edge_index_dict={},
        batch_dict={"entity": torch.zeros(4, dtype=torch.long)},
        structural_context=context,
    )
    output["entity"].square().sum().backward()

    assert output["entity"].shape == (4, 4)
    assert torch.isfinite(output["entity"]).all()
    assert x.grad is not None
    assert conv.rpe_mlp[0].weight.grad is not None


def test_pairwise_attention_isolated_across_variable_size_graphs():
    torch.manual_seed(5)
    conv = _attention(pairwise_feature_dim=1).eval()
    x = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 1, 1, 1])
    context = StructuralAttentionContext(
        pairwise_features=[torch.randn(2, 2, 1), torch.randn(3, 3, 1)]
    )

    reference, _ = conv(
        inv_node_feat_dict={"entity": x},
        edge_index_dict={},
        batch_dict={"entity": batch},
        structural_context=context,
    )
    changed = x.clone()
    changed[batch == 1] += 100.0
    perturbed, _ = conv(
        inv_node_feat_dict={"entity": changed},
        edge_index_dict={},
        batch_dict={"entity": batch},
        structural_context=context,
    )

    assert torch.allclose(
        reference["entity"][batch == 0],
        perturbed["entity"][batch == 0],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_structural_performer_coefficient_receives_gradient():
    torch.manual_seed(6)
    conv = _attention(attn_type="performer", qk_coordinate_dim=2)
    coefficient = torch.nn.Parameter(torch.tensor(0.2))
    coordinates = torch.randn(3, 2)
    context = StructuralAttentionContext(
        qk_coordinates=coordinates,
        qk_coefficient=coefficient,
    )

    output, _ = conv(
        inv_node_feat_dict={"entity": torch.randn(3, 4)},
        edge_index_dict={},
        structural_context=context,
    )
    output["entity"].square().sum().backward()

    assert coefficient.grad is not None
    assert torch.isfinite(coefficient.grad)
    assert conv.attn.q.weight.grad is not None


def test_structural_input_supports_graph_broadcast_and_sign_flip():
    first = HeteroData()
    first["entity"].x = torch.randn(2, 3)
    first["entity"].vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first["entity"].values = torch.tensor([[5.0, 6.0]])
    second = HeteroData()
    second["entity"].x = torch.randn(3, 3)
    second["entity"].vectors = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    second["entity"].values = torch.tensor([[13.0, 14.0]])
    data = Batch.from_data_list([first, second])
    model = _hetero_stack(
        {
            "target_node_type": "entity",
            "node_inputs": [
                {"attribute": "vectors", "dim": 2, "random_sign_flip": True},
                {"attribute": "values", "dim": 2, "broadcast": "graph"},
            ],
        }
    )

    torch.manual_seed(9)
    model.train()
    embedded, batches = model._prepare_node_features(data)
    collected = model._collect_structural_input(
        data,
        batches["entity"],
        device=embedded["entity"].device,
        dtype=embedded["entity"].dtype,
    )

    assert embedded["entity"].shape == (5, 4)
    assert collected.shape == (5, 4)
    assert torch.equal(collected[:2, 2:], torch.tensor([[5.0, 6.0]]).expand(2, -1))
    assert torch.equal(collected[2:, 2:], torch.tensor([[13.0, 14.0]]).expand(3, -1))
    assert torch.equal(collected[:2, :2].abs(), first["entity"].vectors.abs())
    assert torch.equal(collected[2:, :2].abs(), second["entity"].vectors.abs())


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"target_node_type": "entity", "node_inputs": [{"attribute": "x"}]},
            "positive dim",
        ),
        (
            {
                "target_node_type": "entity",
                "pairwise": {"attribute": "pairs", "dim": 1},
                "qk_coordinates": {
                    "attribute": "coords",
                    "dim": 2,
                    "placement": "qk",
                },
            },
            "Only one structural attention input",
        ),
        (
            {
                "target_node_type": "entity",
                "qk_coordinates": {
                    "attribute": "coords",
                    "dim": 2,
                    "placement": "invalid",
                },
            },
            "placement",
        ),
    ],
)
def test_structural_configuration_validation(config, message):
    with pytest.raises(ValueError, match=message):
        _hetero_stack(config)


def test_missing_structural_tensor_reports_attribute_name():
    data = HeteroData()
    data["entity"].x = torch.randn(2, 3)
    model = _hetero_stack(
        {
            "target_node_type": "entity",
            "node_inputs": [{"attribute": "missing_feature", "dim": 2}],
        }
    )

    with pytest.raises(ValueError, match="entity.missing_feature"):
        model._prepare_node_features(data)


def test_attention_rejects_pairwise_features_for_multiple_node_types():
    with pytest.raises(ValueError, match="exactly one attention node type"):
        _attention(
            node_types=("left", "right"),
            attn_node_types=("left", "right"),
            pairwise_feature_dim=1,
        )


def test_attention_rejects_wrong_coordinate_shape():
    conv = _attention(attn_type="performer", qk_coordinate_dim=2)
    context = StructuralAttentionContext(
        qk_coordinates=torch.randn(3, 3),
        qk_coefficient=torch.tensor(0.0),
    )

    with pytest.raises(ValueError, match="structural coordinates"):
        conv(
            inv_node_feat_dict={"entity": torch.randn(3, 4)},
            edge_index_dict={},
            structural_context=context,
        )


def test_structural_attention_checkpoint_round_trip():
    torch.manual_seed(12)
    original = _attention(pairwise_feature_dim=1).eval()
    x = torch.randn(3, 4)
    context = StructuralAttentionContext(pairwise_features=[torch.randn(3, 3, 1)])
    expected, _ = original(
        inv_node_feat_dict={"entity": x},
        edge_index_dict={},
        structural_context=context,
    )
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = _attention(pairwise_feature_dim=1).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual, _ = restored(
        inv_node_feat_dict={"entity": x},
        edge_index_dict={},
        structural_context=context,
    )

    assert torch.equal(expected["entity"], actual["entity"])


def test_domain_provider_protocol_is_structural():
    class ExampleProvider:
        def __call__(self, graph, topology_id=None):
            graph.topology_id = topology_id
            return graph

    provider = ExampleProvider()
    graph = HeteroData()

    assert isinstance(provider, StructuralEncodingProvider)
    assert provider(graph, "generic").topology_id == "generic"
