import importlib.util
import math
from pathlib import Path

import torch
from torch_geometric.data import Batch, HeteroData

from hydragnn.globalAtt.HeteroGPS import (
    EffectiveResistancePerformerAttention,
    HeteroGPSConv,
)
from hydragnn.models.heterogeneous.HeteroBase import HeteroBase
from hydragnn.models.heterogeneous.HeteroSAGEStack import HeteroSAGEStack


_MODULE_PATH = Path(__file__).parents[1] / "examples" / "opf" / "opf_svd_rpe.py"
_SPEC = importlib.util.spec_from_file_location("opf_svd_rpe", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
add_ybus_svd_rpe = _MODULE.add_ybus_svd_rpe
build_ybus = _MODULE.build_ybus
build_topological_laplacian = _MODULE.build_topological_laplacian
compute_topological_laplacian_pe = _MODULE.compute_topological_laplacian_pe
compute_effective_resistance_pe = _MODULE.compute_effective_resistance_pe
compute_effective_resistance_matrix = _MODULE.compute_effective_resistance_matrix
compute_effective_resistance_rpe = _MODULE.compute_effective_resistance_rpe
compute_effective_resistance_qk = _MODULE.compute_effective_resistance_qk
compute_effective_impedance_matrix = _MODULE.compute_effective_impedance_matrix
compute_effective_impedance_pe = _MODULE.compute_effective_impedance_pe
compute_effective_impedance_rpe = _MODULE.compute_effective_impedance_rpe
OPFSpectralPEPreprocessor = _MODULE.OPFSpectralPEPreprocessor
resolve_opf_positional_encoding_config = (
    _MODULE.resolve_opf_positional_encoding_config
)


def _two_bus_transformer():
    data = HeteroData()
    data["bus"].x = torch.zeros(2, 4)
    data["shunt"].x = torch.empty(0, 2)
    data["bus", "ac_line", "bus"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data["bus", "ac_line", "bus"].edge_attr = torch.empty((0, 9))
    data["bus", "transformer", "bus"].edge_index = torch.tensor([[0], [1]])
    # angmin, angmax, r, x, rates, tap, shift, g_sh, b_sh
    data["bus", "transformer", "bus"].edge_attr = torch.tensor(
        [[-0.5, 0.5, 0.01, 0.1, 2.0, 2.0, 2.0, 1.1, 0.2, 0.03, 0.04]]
    )
    data["shunt", "shunt_link", "bus"].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    return data


def _three_bus_path():
    data = HeteroData()
    data["bus"].x = torch.zeros(3, 4)
    data["shunt"].x = torch.empty(0, 2)
    data["bus", "ac_line", "bus"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    # Unit reactance makes effective resistance equal path distance.
    data["bus", "ac_line", "bus"].edge_attr = torch.tensor(
        [
            [-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
            [-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        ]
    )
    data["bus", "transformer", "bus"].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    data["bus", "transformer", "bus"].edge_attr = torch.empty((0, 11))
    data["shunt", "shunt_link", "bus"].edge_index = torch.empty(
        (2, 0), dtype=torch.long
    )
    return data


def pytest_ybus_keeps_complex_transformer_asymmetry():
    ybus = build_ybus(_two_bus_transformer())
    assert ybus.dtype == torch.complex128
    assert not torch.allclose(ybus, ybus.transpose(0, 1))
    assert not torch.allclose(ybus, ybus.conj().transpose(0, 1))


def pytest_ybus_svd_preprocessing_stores_padded_real_factors():
    data = add_ybus_svd_rpe(_two_bus_transformer(), k=3)
    bus = data["bus"]
    for name in ("svd_u_real", "svd_u_imag", "svd_v_real", "svd_v_imag", "svd_s"):
        value = getattr(bus, name)
        assert value.shape == (2, 3)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    # A two-bus matrix has at most two nonzero components, so k=3 pads one.
    assert torch.count_nonzero(bus.svd_s[:, 2]) == 0
    assert torch.all(bus.svd_s[:, :-1] >= 0)
    assert torch.all(bus.svd_s[:, :-1] <= 1)


def pytest_heterogps_svd_rpe_changes_attention_output():
    torch.manual_seed(7)
    conv = HeteroGPSConv(
        channels=4,
        metadata=(["bus"], [("bus", "line", "bus")]),
        conv=None,
        heads=2,
        dropout=0.0,
        attn_type="multihead",
        attn_node_types=["bus"],
        pe_dim=2,
    )
    conv.eval()
    x = {"bus": torch.randn(3, 4)}
    batch = {"bus": torch.zeros(3, dtype=torch.long)}
    factors = {
        "bus": {
            "u_real": torch.randn(3, 2),
            "u_imag": torch.randn(3, 2),
            "v_real": torch.randn(3, 2),
            "v_imag": torch.randn(3, 2),
            "s": torch.rand(3, 2),
        }
    }
    out_a, _ = conv(x, {}, batch, svd_rpe_dict=factors)
    changed = {"bus": {name: value.clone() for name, value in factors["bus"].items()}}
    changed["bus"]["v_imag"] += 2.0
    out_b, _ = conv(x, {}, batch, svd_rpe_dict=changed)
    assert not torch.allclose(out_a["bus"], out_b["bus"])


def pytest_topological_laplacian_stores_smallest_nonzero_eigenpairs():
    data = _three_bus_path()
    laplacian = build_topological_laplacian(data)
    artifact = compute_topological_laplacian_pe(data, k=3, compute_device="cpu")
    vectors = artifact["lap_eigvec"].double()
    values = artifact["lap_eigval"].squeeze(0).double()

    assert torch.allclose(
        values, torch.tensor([1.0, 3.0, 0.0], dtype=torch.float64), atol=1.0e-6
    )
    assert torch.allclose(
        laplacian @ vectors[:, :2],
        vectors[:, :2] * values[:2],
        atol=1.0e-6,
    )
    assert torch.allclose(
        vectors[:, :2].transpose(0, 1) @ vectors[:, :2],
        torch.eye(2, dtype=torch.float64),
        atol=1.0e-6,
    )


def pytest_effective_resistance_summary_excludes_diagonal():
    stats = compute_effective_resistance_pe(
        _three_bus_path(), compute_device="cpu"
    )["effective_resistance_pe"]
    expected = torch.tensor(
        [
            [1.0, 2.0, 0.5, 1.5, 1.5],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 0.5, 1.5, 1.5],
        ]
    )
    assert torch.allclose(stats, expected, atol=1.0e-5)

    two_bus = _three_bus_path()
    two_bus["bus"].x = two_bus["bus"].x[:2]
    two_bus["bus", "ac_line", "bus"].edge_index = torch.tensor([[0], [1]])
    two_bus["bus", "ac_line", "bus"].edge_attr = two_bus[
        "bus", "ac_line", "bus"
    ].edge_attr[:1]
    two_bus_stats = compute_effective_resistance_pe(
        two_bus, compute_device="cpu"
    )["effective_resistance_pe"]
    assert torch.allclose(two_bus_stats[:, 3], torch.tensor([1.0, 1.0]))


def pytest_effective_resistance_rpe_is_raw_pairwise_distance():
    matrix = compute_effective_resistance_matrix(
        _three_bus_path(), compute_device="cpu"
    )
    rpe = compute_effective_resistance_rpe(
        _three_bus_path(), compute_device="cpu", resistance=matrix
    )["pairwise_rpe"]
    expected = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    assert rpe.shape == (3, 3, 1)
    assert torch.allclose(rpe[..., 0], expected, atol=1.0e-5)


def pytest_effective_resistance_qk_coordinates_reconstruct_truncated_distance():
    data = _three_bus_path()
    coordinates = compute_effective_resistance_qk(
        data, k=2, compute_device="cpu"
    )["effective_resistance_qk"]
    reconstructed = torch.cdist(coordinates, coordinates).square()
    exact = compute_effective_resistance_matrix(data, compute_device="cpu").float()

    assert coordinates.shape == (3, 2)
    assert torch.allclose(reconstructed, exact, atol=1.0e-5)


def pytest_resistance_qk_augmentation_matches_explicit_softmax_bias():
    attention = EffectiveResistancePerformerAttention(
        channels=4,
        heads=1,
        head_channels=4,
        resistance_dim=2,
        num_random_features=16,
        dropout=0.0,
    )
    q = torch.tensor(
        [[[[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 0.5, -1.0]]]]
    )
    k = torch.tensor(
        [[[[0.5, -1.0, 1.5, 0.0], [1.0, 0.5, 0.0, -0.5]]]]
    )
    coordinates = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
    coefficient = torch.tensor(0.3, requires_grad=True)

    q_augmented, k_augmented = attention._augment_qk(
        q, k, coordinates, coefficient
    )
    augmented_scores = q_augmented @ k_augmented.transpose(-2, -1)
    content_scores = (q @ k.transpose(-2, -1)) / math.sqrt(4.0)
    resistance = torch.cdist(coordinates, coordinates).square().unsqueeze(1)
    explicit_scores = content_scores + coefficient * resistance

    # The omitted ||x_i||^2 term is query-only, so row softmax is identical.
    assert torch.allclose(
        torch.softmax(augmented_scores, dim=-1),
        torch.softmax(explicit_scores, dim=-1),
        atol=1.0e-6,
    )
    assert q_augmented.size(-1) == 7
    augmented_scores.sum().backward()
    assert coefficient.grad is not None
    assert coefficient.grad.abs() > 0


def pytest_heterogps_resistance_performer_updates_global_coefficient():
    torch.manual_seed(23)
    conv = HeteroGPSConv(
        channels=4,
        metadata=(["bus"], [("bus", "line", "bus")]),
        conv=None,
        heads=1,
        dropout=0.0,
        attn_type="performer",
        attn_node_types=["bus"],
        resistance_qk_dim=2,
        attn_kwargs={"head_channels": 4, "num_random_features": 32},
    )
    conv.eval()
    x = {"bus": torch.randn(3, 4)}
    batch = {"bus": torch.zeros(3, dtype=torch.long)}
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    coefficient = torch.nn.Parameter(torch.tensor(-0.1))

    output, _ = conv(
        x,
        {},
        batch,
        resistance_qk=coordinates,
        resistance_coefficient=coefficient,
    )
    output["bus"].square().sum().backward()

    assert output["bus"].shape == (3, 4)
    assert torch.isfinite(output["bus"]).all()
    assert coefficient.grad is not None
    assert torch.isfinite(coefficient.grad)


def pytest_effective_impedance_summary_and_rpe_use_real_imaginary_parts():
    data = _three_bus_path()
    impedance = compute_effective_impedance_matrix(data, compute_device="cpu")
    expected_distance = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(
        impedance.real, torch.zeros_like(expected_distance), atol=1.0e-6
    )
    assert torch.allclose(impedance.imag, expected_distance, atol=1.0e-6)

    stats = compute_effective_impedance_pe(
        data, compute_device="cpu", impedance=impedance
    )["effective_impedance_pe"]
    expected_imag_stats = torch.tensor(
        [
            [1.0, 2.0, 0.5, 1.5, 1.5],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 0.5, 1.5, 1.5],
        ]
    )
    assert stats.shape == (3, 10)
    assert torch.allclose(stats[:, :5], torch.zeros((3, 5)), atol=1.0e-6)
    assert torch.allclose(stats[:, 5:], expected_imag_stats, atol=1.0e-5)

    rpe = compute_effective_impedance_rpe(
        data, compute_device="cpu", impedance=impedance
    )["pairwise_rpe"]
    assert rpe.shape == (3, 3, 2)
    assert torch.allclose(rpe[..., 0], torch.zeros((3, 3)), atol=1.0e-6)
    assert torch.allclose(rpe[..., 1], expected_distance.float(), atol=1.0e-6)


def pytest_direct_pairwise_rpe_changes_attention_and_has_zero_self_bias():
    torch.manual_seed(19)
    conv = HeteroGPSConv(
        channels=4,
        metadata=(["bus"], [("bus", "line", "bus")]),
        conv=None,
        heads=2,
        dropout=0.0,
        attn_type="multihead",
        attn_node_types=["bus"],
        direct_rpe_dim=2,
        rpe_hidden_dim=4,
        rpe_zero_diagonal=True,
    )
    conv.eval()
    x = {"bus": torch.randn(3, 4)}
    batch = {"bus": torch.zeros(3, dtype=torch.long)}
    pairwise = torch.zeros(3, 3, 2)
    pairwise[..., 0] = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    out_a, _ = conv(x, {}, batch, direct_pairwise_rpe=[pairwise])
    changed = pairwise.clone()
    changed[0, 2, 1] = 4.0
    changed[2, 0, 1] = 4.0
    out_b, _ = conv(x, {}, batch, direct_pairwise_rpe=[changed])
    assert not torch.allclose(out_a["bus"], out_b["bus"])

    learned_bias = conv.rpe_mlp(pairwise)
    diagonal = torch.diagonal(learned_bias, dim1=0, dim2=1)
    assert torch.count_nonzero(diagonal) == 0


def pytest_spectral_preprocessor_caches_and_batches_graph_eigenvalues(tmp_path):
    architecture = {
        "positional_encodings": {
            "precompute": ["laplacian", "effective_resistance"],
            "use": ["laplacian"],
            "cache_by_case": True,
            "compute_device": "cpu",
            "laplacian": {"dim": 2},
        }
    }
    preprocessor = OPFSpectralPEPreprocessor(
        architecture, cache_dir=str(tmp_path)
    )
    first = preprocessor(_three_bus_path(), case_name="path")
    second = preprocessor(_three_bus_path(), case_name="path")

    assert len(list(tmp_path.glob("*.pt"))) == 2
    assert first["bus"].lap_eigvec.shape == (3, 2)
    assert first["bus"].lap_eigval.shape == (1, 2)
    assert first["bus"].effective_resistance_pe.shape == (3, 5)
    assert "generator" not in first.node_types

    batched = Batch.from_data_list([first, second])
    assert batched["bus"].lap_eigvec.shape == (6, 2)
    assert batched["bus"].lap_eigval.shape == (2, 2)


def pytest_pairwise_rpe_artifacts_are_referenced_instead_of_embedded(tmp_path):
    architecture = {
        "positional_encodings": {
            "precompute": [
                "effective_resistance_rpe",
                "effective_impedance_rpe",
            ],
            "use": ["effective_impedance_rpe"],
            "cache_by_case": True,
            "compute_device": "cpu",
        }
    }
    data = OPFSpectralPEPreprocessor(
        architecture, cache_dir=str(tmp_path)
    )(_three_bus_path(), case_name="path")
    bus = data["bus"]
    assert not hasattr(bus, "effective_resistance_rpe")
    assert not hasattr(bus, "effective_impedance_rpe")
    assert Path(bus.effective_resistance_rpe_path).is_file()
    assert Path(bus.effective_impedance_rpe_path).is_file()
    assert len(list(tmp_path.glob("*.pt"))) == 2

    artifact = torch.load(
        bus.effective_impedance_rpe_path,
        map_location="cpu",
        weights_only=True,
    )
    assert artifact["pairwise_rpe"].shape == (3, 3, 2)


def pytest_heterobase_loads_topology_level_rpe_cache(tmp_path):
    architecture = {
        "positional_encodings": {
            "precompute": ["effective_impedance_rpe"],
            "use": ["effective_impedance_rpe"],
            "cache_by_case": True,
            "compute_device": "cpu",
            "effective_impedance_rpe": {
                "feature_dim": 2,
                "mlp_hidden_dim": 4,
                "zero_diagonal_bias": True,
            },
        }
    }
    data = OPFSpectralPEPreprocessor(
        architecture, cache_dir=str(tmp_path)
    )(_three_bus_path(), case_name="path")
    # Keep the forward smoke test focused on bus attention; pooling an empty
    # auxiliary node store is independently unsupported by HeteroBase.
    del data["shunt", "shunt_link", "bus"]
    del data["shunt"]
    model = HeteroSAGEStack(
        input_dim=4,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="GPS",
        global_attn_type="multihead",
        global_attn_heads=2,
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
        node_target_type="bus",
        metadata=data.metadata(),
        node_input_dims={"bus": 4},
        attn_node_types=["bus"],
        positional_encodings=architecture["positional_encodings"],
    )
    embedded, batch = model._prepare_node_features(data)
    matrices = model._get_direct_pairwise_rpe(
        data,
        batch["bus"],
        device=embedded["bus"].device,
        dtype=embedded["bus"].dtype,
    )

    assert model.direct_rpe_source == "effective_impedance_rpe"
    assert model.direct_rpe_dim == 2
    assert len(matrices) == 1
    assert matrices[0].shape == (3, 3, 2)

    model.eval()
    outputs = model(data)
    assert isinstance(outputs, list)
    assert outputs[0].shape == (3, 1)


def pytest_heterobase_shares_one_resistance_coefficient_across_performer_layers():
    architecture = {
        "positional_encodings": {
            "precompute": ["effective_resistance_qk"],
            "use": ["effective_resistance_qk"],
            "compute_device": "cpu",
            "effective_resistance_qk": {
                "dim": 2,
                "coefficient_init": -0.05,
            },
        }
    }
    data = OPFSpectralPEPreprocessor(architecture)(
        _three_bus_path(), "path"
    )
    del data["shunt", "shunt_link", "bus"]
    del data["shunt"]
    model = HeteroSAGEStack(
        input_dim=4,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="GPS",
        global_attn_type="performer",
        global_attn_heads=2,
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
        num_conv_layers=2,
        node_target_type="bus",
        metadata=data.metadata(),
        node_input_dims={"bus": 4},
        attn_node_types=["bus"],
        positional_encodings=architecture["positional_encodings"],
    )
    model.eval()
    outputs = model(data)
    outputs[0].square().sum().backward()

    coefficient_names = [
        name
        for name, _ in model.named_parameters()
        if "resistance_qk_coefficient" in name
    ]
    assert coefficient_names == ["resistance_qk_coefficient"]
    assert model.resistance_qk_coefficient.shape == ()
    assert model.resistance_qk_coefficient.grad is not None
    assert all(conv.resistance_qk_dim == 2 for conv in model.graph_convs)
    assert outputs[0].shape == (3, 1)


def pytest_resistance_coordinates_can_be_fused_before_qkv_projection():
    architecture = {
        "positional_encodings": {
            "precompute": ["effective_resistance_qk"],
            "use": ["effective_resistance_qk"],
            "compute_device": "cpu",
            "effective_resistance_qk": {
                "dim": 2,
                "placement": "input",
            },
        }
    }
    data = OPFSpectralPEPreprocessor(architecture)(
        _three_bus_path(), "path"
    )
    del data["shunt", "shunt_link", "bus"]
    del data["shunt"]
    model = HeteroSAGEStack(
        input_dim=4,
        hidden_dim=8,
        output_dim=[1],
        pe_dim=0,
        global_attn_engine="GPS",
        global_attn_type="performer",
        global_attn_heads=2,
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
        node_target_type="bus",
        metadata=data.metadata(),
        node_input_dims={"bus": 4},
        attn_node_types=["bus"],
        positional_encodings=architecture["positional_encodings"],
    )
    embedded, _ = model._prepare_node_features(data)
    model.eval()
    outputs = model(data)

    assert model.resistance_qk_placement == "input"
    assert model.resistance_qk_input_dim == 2
    assert model.resistance_qk_attention_dim == 0
    assert model.resistance_qk_coefficient is None
    assert model.bus_input_pe_dim == 2
    assert embedded["bus"].shape == (3, 8)
    assert outputs[0].shape == (3, 1)


def pytest_laplacian_sign_flip_is_graphwise_and_train_only():
    class _SignFlipHarness:
        training = True
        laplacian_random_sign_flip = True

    harness = _SignFlipHarness()
    vectors = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    batch = torch.tensor([0, 0, 1, 1])
    torch.manual_seed(11)
    flipped = HeteroBase._apply_laplacian_sign_flip(harness, vectors, batch)

    assert torch.equal(flipped.abs(), vectors.abs())
    ratios = flipped / vectors
    assert torch.equal(ratios[0], ratios[1])
    assert torch.equal(ratios[2], ratios[3])

    harness.training = False
    assert torch.equal(
        HeteroBase._apply_laplacian_sign_flip(harness, vectors, batch), vectors
    )


def pytest_legacy_svd_configuration_remains_supported():
    resolved = resolve_opf_positional_encoding_config(
        {"pe_encoder": "svd_ybus", "pe_dim": 4, "svd_rpe_tolerance": 1.0e-8}
    )
    assert resolved["use"] == ["ybus_svd"]
    assert resolved["precompute"] == ["ybus_svd"]
    assert resolved["ybus_svd"]["dim"] == 4
    assert resolved["ybus_svd"]["relative_tolerance"] == 1.0e-8


def pytest_laplacian_and_resistance_are_fused_only_into_bus_input():
    architecture = {
        "positional_encodings": {
            "precompute": ["laplacian", "effective_resistance"],
            "use": ["laplacian", "effective_resistance"],
            "compute_device": "cpu",
            "laplacian": {"dim": 2, "random_sign_flip": False},
        }
    }
    data = OPFSpectralPEPreprocessor(architecture)(_three_bus_path(), "path")
    model = HeteroSAGEStack(
        input_dim=4,
        hidden_dim=8,
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
        node_target_type="bus",
        metadata=data.metadata(),
        node_input_dims={"bus": 4, "shunt": 2},
        positional_encodings=architecture["positional_encodings"],
    )
    embedded, _ = model._prepare_node_features(data)

    assert model.bus_input_pe_dim == 9  # 2 eigenvectors + 2 values + 5 stats.
    assert model.bus_pe_fuser is not None
    assert embedded["bus"].shape == (3, 8)
    assert embedded["shunt"].shape == (0, 8)


def pytest_ten_dimensional_impedance_summary_is_fused_into_bus_input():
    architecture = {
        "positional_encodings": {
            "precompute": ["effective_impedance"],
            "use": ["effective_impedance"],
            "compute_device": "cpu",
        }
    }
    data = OPFSpectralPEPreprocessor(architecture)(_three_bus_path(), "path")
    model = HeteroSAGEStack(
        input_dim=4,
        hidden_dim=8,
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
        node_target_type="bus",
        metadata=data.metadata(),
        node_input_dims={"bus": 4, "shunt": 2},
        positional_encodings=architecture["positional_encodings"],
    )
    embedded, _ = model._prepare_node_features(data)

    assert model.bus_input_pe_dim == 10
    assert embedded["bus"].shape == (3, 8)
