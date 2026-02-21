import os
import tempfile
from importlib import import_module

import pytest
import torch
import torch.distributed as dist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from hydragnn.models.create import create_model
from hydragnn.utils.distributed import get_distributed_model
from hydragnn.utils.model.model import update_multibranch_heads

tvt = import_module("hydragnn.train.train_validate_test")


pytestmark = pytest.mark.mpi_skip()


def _build_tiny_interatomic_batch(num_nodes=4):
    pos = torch.randn(num_nodes, 3, dtype=torch.float32)
    x = torch.randint(1, 5, (num_nodes, 1), dtype=torch.int64).float()
    batch = torch.zeros(num_nodes, dtype=torch.long)
    energy = torch.randn(1, 1, dtype=torch.float32)
    forces = torch.randn(num_nodes, 3, dtype=torch.float32)
    pe = torch.randn(num_nodes, 6, dtype=torch.float32)

    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        batch=batch,
        energy=energy,
        forces=forces,
        pe=pe,
        edge_shifts=torch.zeros(edge_index.size(1), 3, dtype=torch.float32),
    )


def _bf16_supported():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).major >= 7
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return True
    return bool(getattr(torch.backends.cpu, "has_bf16", False))


@pytest.mark.gpu()
@pytest.mark.parametrize("precision", ["fp32", "fp64", "bf16"])
def pytest_fsdp2_enhanced_wrapper_force_grad_regression(monkeypatch, precision):
    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not (has_cuda or has_xpu):
        pytest.skip("FSDP2 force-grad regression requires CUDA or XPU")
    if precision == "bf16" and not _bf16_supported():
        pytest.skip("bf16 precision is not supported on this device")

    backend = "gloo"
    if dist.is_initialized():
        pytest.skip("Distributed process group already initialized")

    with tempfile.NamedTemporaryFile() as init_file:
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{init_file.name}",
            rank=0,
            world_size=1,
        )

        try:
            monkeypatch.setenv("HYDRAGNN_USE_FSDP", "1")
            monkeypatch.setenv("HYDRAGNN_FSDP_VERSION", "2")
            monkeypatch.setenv("HYDRAGNN_FSDP_STRATEGY", "SHARD_GRAD_OP")
            monkeypatch.setenv("HYDRAGNN_AGGR_BACKEND", "torch")
            monkeypatch.setenv("HYDRAGNN_USE_ddstore", "0")

            output_heads = {
                "node": {
                    "num_sharedlayers": 1,
                    "dim_sharedlayers": 16,
                    "num_headlayers": 1,
                    "dim_headlayers": [1],
                    "type": "mlp",
                }
            }

            model = create_model(
                mpnn_type="EGNN",
                input_dim=1,
                hidden_dim=32,
                output_dim=[1],
                pe_dim=6,
                global_attn_engine="",
                global_attn_type="",
                global_attn_heads=1,
                output_type=["node"],
                output_heads=update_multibranch_heads(output_heads),
                activation_function="relu",
                loss_function_type="mse",
                task_weights=[1.0],
                num_conv_layers=2,
                num_nodes=16,
                enable_interatomic_potential=True,
                energy_weight=1.0,
                energy_peratom_weight=1.0,
                force_weight=1.0,
                use_gpu=True,
            )

            assert (
                model.__class__.__name__ == "EnhancedModelWrapper"
            ), "Model must use HydraGNN EnhancedModelWrapper"

            model = get_distributed_model(model, verbosity=0)

            _, param_dtype, _ = tvt.resolve_precision(precision)
            model = model.to(dtype=param_dtype)

            reshard_calls = []
            original_setter = tvt._set_reshard_after_backward

            def _spy_setter(model_obj, enabled):
                reshard_calls.append(enabled)
                return original_setter(model_obj, enabled)

            monkeypatch.setattr(tvt, "_set_reshard_after_backward", _spy_setter)
            monkeypatch.setattr(tvt, "print_peak_memory", lambda *args, **kwargs: None)

            data = _build_tiny_interatomic_batch()
            loader = DataLoader([data], batch_size=1, shuffle=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

            train_error, task_errors = tvt.train(
                loader,
                model,
                optimizer,
                verbosity=0,
                num_tasks=3,
                compute_grad_energy=True,
                precision=precision,
            )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    assert torch.isfinite(train_error)
    assert len(task_errors) == 3
    assert torch.isfinite(task_errors).all()
    assert reshard_calls, "Expected FSDP2 reshard toggling calls"
    assert reshard_calls[0] is False
    assert True in reshard_calls
