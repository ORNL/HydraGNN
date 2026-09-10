import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from hydragnn.models.create import create_model
from hydragnn.utils.model.model import update_multibranch_heads

GENERIC_MPNN_TYPES = (
    "PAINN",
    "PNAEq",
    "DimeNet",
    "SchNet",
    "EGNN",
    "PNAPlus",
    "PNA",
    "GAT",
    "GIN",
    "SAGE",
    "CGCNN",
    "MFC",
)
CUSTOM_EMBEDDING_MPNN_TYPES = ("PAINN", "PNAEq", "DimeNet", "SchNet", "PNAPlus")
NATIVE_SPECIES_MPNN_TYPES = {"MACE", "UMA", "AllScAIP"}
ATOMISTIC_EXAMPLE_DIRECTORIES = (
    "LennardJones",
    "lsms",
    "alexandria",
    "ani1_x",
    "csce",
    "dftb_uv_spectrum",
    "eam",
    "md17",
    "mptrj",
    "multibranch",
    "multibranch_hpo",
    "multidataset",
    "multidataset_deepspeed",
    "multidataset_hpo",
    "multidataset_hpo_sc26",
    "nabla2_dft",
    "open_catalyst_2020",
    "open_catalyst_2022",
    "open_catalyst_2025",
    "open_direct_air_capture_2023",
    "open_materials_2024",
    "open_molecules_2025",
    "open_polymers_2026",
    "qcml",
    "qm7x",
    "qm9",
    "qm9_hpo",
    "transition1x",
)


def _embedding_encoding(name="atomic_numbers", start=0, embedding_dim=8):
    return {
        "name": name,
        "start": start,
        "dim": 1,
        "type": "embedding",
        "num_categories": 118,
        "embedding_dim": embedding_dim,
        "min_value": 1,
    }


def _model(mpnn_type="EGNN", atomistic=False, encodings=None, input_dim=1):
    heads = update_multibranch_heads(
        {
            "node": {
                "num_sharedlayers": 1,
                "dim_sharedlayers": 8,
                "num_headlayers": 1,
                "dim_headlayers": [8],
                "type": "mlp",
            }
        }
    )
    return create_model(
        mpnn_type=mpnn_type,
        input_dim=input_dim,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="",
        global_attn_type="",
        global_attn_heads=1,
        output_type=["node"],
        output_heads=heads,
        activation_function="elu",
        loss_function_type="mse",
        task_weights=[1.0],
        num_conv_layers=2,
        num_nodes=3,
        max_neighbours=8,
        edge_dim=1 if mpnn_type == "CGCNN" else None,
        pna_deg=torch.tensor([0, 1, 2, 1]),
        num_radial=4,
        num_spherical=3,
        num_gaussians=8,
        num_filters=8,
        radius=5.0,
        envelope_exponent=5,
        basis_emb_size=4,
        int_emb_size=8,
        out_emb_size=8,
        num_before_skip=1,
        num_after_skip=1,
        enable_interatomic_potential=atomistic,
        input_node_encodings=encodings,
        energy_weight=0.1,
        force_weight=1.0,
        use_gpu=False,
    )


def _data(values=None):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]],
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    return Data(
        x=values if values is not None else torch.tensor([[7.0], [1.0], [8.0]]),
        pos=pos,
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.shape[1], 1),
        edge_shifts=torch.zeros(edge_index.shape[1], 3),
        batch=torch.zeros(3, dtype=torch.long),
        energy=torch.zeros(1),
        forces=torch.zeros_like(pos),
    )


@pytest.mark.parametrize("mpnn_type", GENERIC_MPNN_TYPES)
def test_unencoded_generic_input_is_unchanged(mpnn_type):
    model = _model(mpnn_type=mpnn_type)
    data = _data()
    features, _, _ = model._embedding(data)
    assert not model.input_feature_encoders
    assert torch.equal(features, data.x)


@pytest.mark.parametrize("mpnn_type", GENERIC_MPNN_TYPES)
def test_embedding_encoded_input_has_configured_width(mpnn_type):
    model = _model(mpnn_type=mpnn_type, encodings=[_embedding_encoding()], input_dim=8)
    data = _data()
    features, _, _ = model._embedding(data)
    embedding = model.input_feature_encoders["0"]
    assert embedding.num_embeddings == 118
    assert features.shape == (3, 8)
    assert torch.equal(features, embedding(data.x[:, 0].long() - 1))


def test_encoding_is_generic_and_preserves_continuous_features():
    values = torch.tensor([[0.1, 7.0, 1.0], [0.2, 1.0, 2.0], [0.3, 8.0, 3.0]])
    model = _model(
        encodings=[_embedding_encoding(name="site_type", start=1, embedding_dim=4)],
        input_dim=6,
    )
    features = model._input_node_features(_data(values))
    expected = torch.cat(
        (
            values[:, :1],
            model.input_feature_encoders["0"](values[:, 1].long() - 1),
            values[:, 2:],
        ),
        dim=1,
    )
    assert torch.equal(features, expected)


def test_one_hot_encoding_is_supported_for_arbitrary_node_feature():
    encoding = {
        "name": "site_type",
        "start": 0,
        "dim": 1,
        "type": "one_hot",
        "num_categories": 4,
        "embedding_dim": None,
        "min_value": 0,
    }
    model = _model(encodings=[encoding], input_dim=4)
    data = _data(torch.tensor([[0.0], [2.0], [3.0]]))
    expected = torch.nn.functional.one_hot(data.x[:, 0].long(), num_classes=4).float()
    assert torch.equal(model._input_node_features(data), expected)


def test_qm9_property_example_provides_canonical_atomic_numbers():
    from examples.qm9.qm9 import qm9_pre_transform

    data = Data(
        z=torch.tensor([6, 1, 1], dtype=torch.long),
        y=torch.zeros(1, 19),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )

    def add_test_pe(sample):
        sample.pe = torch.zeros(sample.num_nodes, 2)
        return sample

    transformed = qm9_pre_transform(data, add_test_pe)
    assert torch.equal(transformed.atomic_numbers, data.z)
    assert transformed.atomic_numbers.dtype == torch.long
    assert transformed.atomic_numbers.ndim == 1


def test_atomistic_examples_use_canonical_atomic_number_input():
    examples = Path(__file__).resolve().parents[1] / "examples"
    checked = []
    for directory in ATOMISTIC_EXAMPLE_DIRECTORIES:
        for config_path in (examples / directory).glob("*.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architecture = config.get("NeuralNetwork", {}).get("Architecture", {})
            if "mpnn_type" not in architecture:
                continue
            if architecture["mpnn_type"] in NATIVE_SPECIES_MPNN_TYPES:
                continue
            checked.append(config_path)
            atomic_numbers = [
                variable
                for variable in config["Variables"]["inputs"]
                if variable["name"] == "atomic_numbers"
            ]
            assert len(atomic_numbers) == 1
            encoding = atomic_numbers[0].get("encoding")
            if encoding is not None:
                assert encoding["type"] == "embedding"
                assert encoding["min_value"] == 1
                assert encoding["num_categories"] == 118
    assert checked


def test_multibranch_examples_preserve_continuous_atomic_number_input():
    examples = Path(__file__).resolve().parents[1] / "examples" / "multibranch"
    for config_path in examples.glob("*.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architecture = config.get("NeuralNetwork", {}).get("Architecture", {})
        if "mpnn_type" not in architecture:
            continue
        atomic_numbers = next(
            variable
            for variable in config["Variables"]["inputs"]
            if variable["name"] == "atomic_numbers"
        )
        assert "encoding" not in atomic_numbers


@pytest.mark.parametrize("mpnn_type", CUSTOM_EMBEDDING_MPNN_TYPES)
def test_custom_embedding_paths_use_configured_input_encoder(mpnn_type):
    model = _model(
        mpnn_type=mpnn_type,
        atomistic=True,
        encodings=[_embedding_encoding()],
        input_dim=8,
    ).model
    data = _data()
    features, _, _ = model._embedding(data)
    expected = model.input_feature_encoders["0"](data.x[:, 0].long() - 1)
    assert torch.equal(features, expected)


@pytest.mark.parametrize("mpnn_type", CUSTOM_EMBEDDING_MPNN_TYPES)
def test_custom_embedding_paths_receive_gradients(mpnn_type, monkeypatch):
    if mpnn_type == "SchNet":
        import torch_geometric.nn.models.schnet as schnet_module

        def complete_radius_graph(pos, r, batch, max_num_neighbors):
            del r, max_num_neighbors
            nodes = torch.arange(pos.shape[0], device=pos.device)
            row = nodes.repeat_interleave(pos.shape[0])
            col = nodes.repeat(pos.shape[0])
            mask = (row != col) & (batch[row] == batch[col])
            return torch.stack((row[mask], col[mask]))

        monkeypatch.setattr(schnet_module, "radius_graph", complete_radius_graph)
    model = _model(
        mpnn_type=mpnn_type,
        atomistic=True,
        encodings=[_embedding_encoding()],
        input_dim=8,
    )
    data = _data()
    predictions = model(data)
    sum(prediction.sum() for prediction in predictions).backward()
    gradient = model.model.input_feature_encoders["0"].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[(data.x[:, 0] - 1).long()].abs().sum() > 0


@pytest.mark.parametrize("value", [0.0, 119.0, 1.5])
def test_invalid_encoded_categories_are_rejected(value):
    model = _model(encodings=[_embedding_encoding()], input_dim=8)
    with pytest.raises((TypeError, ValueError)):
        model._input_node_features(_data(torch.tensor([[value], [1.0], [8.0]])))


def test_atomistic_forward_backward_force_path():
    model = _model(atomistic=True, encodings=[_embedding_encoding()], input_dim=8)
    data = _data()
    predictions = model(data)
    energy = predictions[0].sum()
    forces = -torch.autograd.grad(energy, data.pos, create_graph=True)[0]
    assert forces.shape == data.pos.shape
    assert torch.isfinite(forces).all()
    forces.square().sum().backward()
    assert model.model.input_feature_encoders["0"].weight.grad is not None
