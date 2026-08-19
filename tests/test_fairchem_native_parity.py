"""Optional numerical parity tests against a native FAIR-Chem installation.

These tests compare backbone and normalization boundaries with identical
weights. HydraGNN task decoders are intentionally outside the parity contract.
"""

from __future__ import annotations

import copy

import pytest
import torch

fairchem = pytest.importorskip("fairchem", reason="native FAIR-Chem is optional")

from fairchem.core.models.allscaip.AllScAIP import (  # noqa: E402
    AllScAIPBackbone as NativeAllScAIPBackbone,
)
from fairchem.core.models.allscaip.utils.allscaip_radius_graph import (  # noqa: E402
    biknn_radius_graph as native_biknn_radius_graph,
)
from fairchem.core.models.escaip.utils.nn_utils import (  # noqa: E402
    NormalizationType as NativeNormalizationType,
)
from fairchem.core.models.escaip.utils.nn_utils import (  # noqa: E402
    get_normalization_layer as native_get_normalization_layer,
)
from fairchem.core.models.uma.escn_md import (  # noqa: E402
    eSCNMDBackbone as NativeUMABackbone,
)

from hydragnn.models.AllScAIPStack import _FairChemAdapter  # noqa: E402
from hydragnn.models.UMAStack import _UMADataDict  # noqa: E402
from hydragnn.utils.model.allscaip.AllScAIP import (  # noqa: E402
    AllScAIPBackbone,
)
from hydragnn.utils.model.allscaip.utils.allscaip_radius_graph import (  # noqa: E402
    biknn_radius_graph,
)
from hydragnn.utils.model.escaip.utils.nn_utils import (  # noqa: E402
    NormalizationType,
    get_normalization_layer,
)
from hydragnn.utils.model.uma._vendored.fairchem.core.models.uma.escn_md import (  # noqa: E402,E501
    eSCNMDBackbone as VendoredUMABackbone,
)

pytestmark = pytest.mark.fairchem

# FAIR-Chem source used while authoring this suite. CI/reproducible runs should
# install this commit (or deliberately update the reference and tolerances).
FAIRCHEM_REFERENCE_COMMIT = "de5db01588da57665bde96966091d268a9b6b8f7"


def _allscaip_config() -> dict:
    return {
        "regress_forces": False,
        "direct_forces": False,
        "regress_stress": False,
        "hidden_size": 64,
        "num_layers": 1,
        "activation": "gelu",
        "use_residual_scaling": True,
        "use_node_path": False,
        "use_compile": False,
        "use_padding": False,
        "dataset_list": [],
        "max_num_elements": 119,
        "max_atoms": 8,
        "max_batch_size": 2,
        "max_radius": 3.0,
        "knn_k": 4,
        "knn_soft": False,
        "knn_sigmoid_scale": 0.2,
        "knn_lse_scale": 0.1,
        "knn_use_low_mem": True,
        "use_chunked_graph": False,
        "graph_chunk_size": 3,
        "distance_function": "gaussian",
        "use_envelope": True,
        "atten_name": "math",
        "atten_num_heads": 1,
        "freequency_list": [20, 10, 4, 10, 20],
        "use_freq_mask": True,
        "use_sincx_mask": False,
        "normalization": "rmsnorm",
        "mlp_dropout": 0.0,
        "atten_dropout": 0.0,
    }


def _allscaip_adapter(*, batched: bool, periodic: bool) -> _FairChemAdapter:
    torch.manual_seed(31)
    sizes = [4, 3] if batched else [5]
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    pos = torch.rand(sum(sizes), 3)
    cell_scale = 2.5 if periodic else 6.0
    cell = torch.eye(3).repeat(len(sizes), 1, 1) * cell_scale
    pbc = torch.full((len(sizes), 3), periodic, dtype=torch.bool)
    return _FairChemAdapter(
        pos=pos,
        atomic_numbers=torch.arange(sum(sizes)).remainder(8).add(1),
        batch=batch,
        num_graphs=len(sizes),
        cell=cell,
        pbc=pbc,
        charge=torch.zeros(len(sizes), dtype=torch.long),
        spin=torch.zeros(len(sizes), dtype=torch.long),
    )


@pytest.mark.parametrize(
    "batched,periodic", [(False, False), (False, True), (True, False)]
)
def pytest_allscaip_backbone_node_representations(batched, periodic):
    torch.manual_seed(11)
    ours = AllScAIPBackbone(**_allscaip_config()).eval()
    native = NativeAllScAIPBackbone(**_allscaip_config()).eval()
    native.load_state_dict(copy.deepcopy(ours.state_dict()), strict=True)

    with torch.no_grad():
        ours_out = ours(_allscaip_adapter(batched=batched, periodic=periodic))
        native_out = native(_allscaip_adapter(batched=batched, periodic=periodic))

    assert ours_out["node_reps"].dtype == native_out["node_reps"].dtype
    assert torch.allclose(
        ours_out["node_reps"], native_out["node_reps"], atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def pytest_allscaip_output_normalization_matches_native(dtype):
    torch.manual_seed(13)
    ours = get_normalization_layer(NormalizationType.RMSNorm)(64)
    native = native_get_normalization_layer(NativeNormalizationType.RMSNorm)(64)
    native.load_state_dict(copy.deepcopy(ours.state_dict()), strict=True)
    ours = ours.to(dtype)
    native = native.to(dtype)
    node_reps = torch.randn(9, 64, dtype=dtype)
    assert torch.allclose(ours(node_reps), native(node_reps), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def pytest_allscaip_neighbor_graph_matches_native(periodic, dtype):
    ours_data = _allscaip_adapter(batched=True, periodic=periodic)
    native_data = _allscaip_adapter(batched=True, periodic=periodic)
    for data in (ours_data, native_data):
        data.pos = data.pos.to(dtype)
        data.cell = data.cell.to(dtype)
    args = (3.0, 4, False, 0.2, 0.1, True, None, ours_data.pos.device)
    ours = biknn_radius_graph(ours_data, *args, compute_dist_pairwise=True)
    if dtype == torch.float64:
        # FAIR-Chem 2.22.0 constructs PBC image identifiers in the process
        # default dtype, so its FP64 graph path raises before comparison.
        # HydraGNN's successful FP64 result is the intentional parity fix.
        assert all(
            value.dtype == dtype for value in ours if value.dtype.is_floating_point
        )
        with pytest.raises(RuntimeError, match="same dtype"):
            native_biknn_radius_graph(native_data, *args, compute_dist_pairwise=True)
        return
    native = native_biknn_radius_graph(native_data, *args, compute_dist_pairwise=True)
    for ours_value, native_value in zip(ours, native):
        if ours_value.dtype.is_floating_point:
            # HydraGNN deliberately propagates position dtype where native
            # FAIR-Chem still allocates some graph buffers in the default
            # dtype. Compare values in HydraGNN's requested dtype.
            assert ours_value.dtype == dtype
            assert torch.allclose(
                ours_value, native_value.to(dtype), atol=1e-7, rtol=1e-6
            )
        else:
            assert torch.equal(ours_value, native_value)


def _uma_config() -> dict:
    return {
        "max_num_elements": 100,
        "sphere_channels": 8,
        "lmax": 2,
        "mmax": 2,
        "grid_resolution": None,
        "otf_graph": False,
        "max_neighbors": 8,
        "cutoff": 4.0,
        "edge_channels": 8,
        "distance_function": "gaussian",
        "num_distance_basis": 8,
        "direct_forces": False,
        "regress_forces": False,
        "direct_stress": False,
        "regress_stress": False,
        "regress_hessian": False,
        "num_layers": 1,
        "hidden_channels": 8,
        "norm_type": "rms_norm_sh",
        "act_type": "gate",
        "ff_type": "spectral",
        "activation_checkpointing": False,
        "use_dataset_embedding": False,
        "dataset_list": None,
        "dataset_mapping": {"_hydragnn_default": "_hydragnn_default"},
        "use_quaternion_wigner": False,
        "always_use_pbc": False,
    }


def _uma_data_dict(*, batched: bool, periodic: bool) -> _UMADataDict:
    torch.manual_seed(37)
    sizes = [4, 3] if batched else [5]
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    edge_blocks = []
    offset = 0
    for size in sizes:
        index = torch.arange(size)
        src, dst = torch.meshgrid(index, index, indexing="ij")
        mask = src != dst
        edge_blocks.append(torch.stack([src[mask] + offset, dst[mask] + offset]))
        offset += size
    edge_index = torch.cat(edge_blocks, dim=1)
    num_edges = [size * (size - 1) for size in sizes]
    cell_offsets = torch.zeros(edge_index.shape[1], 3)
    if periodic:
        cell_offsets[0, 0] = 1.0
    return _UMADataDict(
        {
            "pos": torch.rand(sum(sizes), 3),
            "atomic_numbers": torch.arange(sum(sizes)).remainder(8).add(1),
            "atomic_numbers_full": torch.arange(sum(sizes)).remainder(8).add(1),
            "batch": batch,
            "batch_full": batch,
            "natoms": torch.tensor(sizes),
            "nedges": torch.tensor(num_edges),
            "cell": torch.eye(3).repeat(len(sizes), 1, 1) * 6.0,
            "cell_offsets": cell_offsets,
            "pbc": torch.full((len(sizes), 3), periodic, dtype=torch.bool),
            "edge_index": edge_index,
            "charge": torch.zeros(len(sizes), dtype=torch.long),
            "spin": torch.zeros(len(sizes), dtype=torch.long),
        }
    )


@pytest.mark.parametrize(
    "batched,periodic", [(False, False), (False, True), (True, False)]
)
def pytest_uma_spherical_backbone_embeddings(batched, periodic):
    torch.manual_seed(17)
    ours = VendoredUMABackbone(**_uma_config()).eval()
    native = NativeUMABackbone(**_uma_config()).eval()
    native.load_state_dict(copy.deepcopy(ours.state_dict()), strict=True)

    with torch.no_grad():
        ours_embedding = ours(_uma_data_dict(batched=batched, periodic=periodic))[
            "node_embedding"
        ]
        native_embedding = native(_uma_data_dict(batched=batched, periodic=periodic))[
            "node_embedding"
        ]

    # Compare the complete spherical embedding. L=0 and L=1 are explicit
    # slices below so failures identify scalar versus vector-irrep drift.
    assert torch.allclose(ours_embedding, native_embedding, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        ours_embedding[:, 0], native_embedding[:, 0], atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        ours_embedding[:, 1:4], native_embedding[:, 1:4], atol=1e-6, rtol=1e-5
    )


def pytest_uma_backbone_position_gradients():
    torch.manual_seed(19)
    ours = VendoredUMABackbone(**_uma_config()).eval()
    native = NativeUMABackbone(**_uma_config()).eval()
    native.load_state_dict(copy.deepcopy(ours.state_dict()), strict=True)
    ours_data = _uma_data_dict(batched=True, periodic=False)
    native_data = _uma_data_dict(batched=True, periodic=False)
    ours_data["pos"].requires_grad_(True)
    native_data["pos"].requires_grad_(True)

    ours_value = ours(ours_data)["node_embedding"][:, 0].sum()
    native_value = native(native_data)["node_embedding"][:, 0].sum()
    ours_grad = torch.autograd.grad(ours_value, ours_data["pos"])[0]
    native_grad = torch.autograd.grad(native_value, native_data["pos"])[0]
    assert torch.allclose(ours_grad, native_grad, atol=2e-6, rtol=2e-5)
